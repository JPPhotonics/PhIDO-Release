"""Streamlit app for reviewing queued KB changes in Neo4j."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import streamlit as st

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover - fallback for older streamlit
    get_script_run_ctx = None

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.config import Neo4jConfig
from PhotonicsAI.KnowledgeBase.Neo4j.schema_registry import SchemaRegistry
from PhotonicsAI.KnowledgeBase.agents.dia_agent import DIAAgent
from PhotonicsAI.KnowledgeBase.agents.vsa_agent.models import ProposedNode, ProposedEdge


def _connect_client() -> Neo4jClient:
    config = Neo4jConfig()
    client = Neo4jClient(config=config)
    client.connect()
    return client


def _fetch_review_items(
    client: Neo4jClient, status: str, priority: Optional[str]
) -> List[Dict[str, Any]]:
    with client.driver.session() as session:
        query = """
        MATCH (r:ReviewItem)
        WHERE r.status = $status
        RETURN r
        ORDER BY r.created_at DESC
        """
        records = session.run(query, status=status)
        items = [dict(record["r"]) for record in records]
    if priority and priority != "any":
        items = [item for item in items if item.get("priority") == priority]
    return items


def _update_review_status(
    client: Neo4jClient,
    item_id: str,
    status: str,
    reviewer: str,
    notes: str,
    action: str,
) -> None:
    with client.driver.session() as session:
        query = """
        MATCH (r:ReviewItem {id: $id})
        SET r.status = $status,
            r.reviewed_at = datetime(),
            r.review_action = $action,
            r.review_notes = $notes,
            r.reviewed_by = $reviewer
        """
        session.run(
            query,
            id=item_id,
            status=status,
            notes=notes,
            reviewer=reviewer,
            action=action,
        )


def _load_payload(payload_json: str) -> Dict[str, Any]:
    try:
        return json.loads(payload_json)
    except Exception:
        return {"_raw": payload_json}


def _is_node_payload(payload: Dict[str, Any]) -> bool:
    return all(k in payload for k in ("collection", "name", "operation"))


def _is_edge_payload(payload: Dict[str, Any]) -> bool:
    return all(k in payload for k in ("edge_collection", "from_node", "to_node"))


def _commit_node(agent: DIAAgent, payload: Dict[str, Any], source_document: str) -> None:
    node = ProposedNode(**payload)
    agent._commit_node(node, source_document)


def _commit_edge(agent: DIAAgent, payload: Dict[str, Any]) -> None:
    edge = ProposedEdge(**payload)
    agent._commit_edge(edge)


def main() -> None:
    if get_script_run_ctx is not None and get_script_run_ctx() is None:
        print("This app must be run with: streamlit run review_queue_app.py")
        return

    st.set_page_config(page_title="KG Review Queue", layout="wide")
    st.title("Knowledge Graph Review Queue")

    reviewer = st.text_input("Reviewer name", value="expert")
    status_filter = st.selectbox("Status", options=["pending", "approved", "rejected"], index=0)
    priority_filter = st.selectbox("Priority", options=["any", "high", "medium", "normal", "low"], index=0)

    if st.button("Refresh"):
        st.experimental_rerun()

    try:
        client = _connect_client()
    except Exception as exc:
        st.error(f"Failed to connect to Neo4j: {exc}")
        return

    agent = DIAAgent(kb_client=client)

    items = _fetch_review_items(client, status_filter, priority_filter)
    st.write(f"Found {len(items)} review items.")

    for item in items:
        item_id = item.get("id", "")
        payload = _load_payload(item.get("payload_json", ""))
        created_at = item.get("created_at", "")
        source_doc = item.get("source_document", "")
        reason = item.get("reason", "")
        item_type = item.get("item_type", "")
        priority = item.get("priority", "")

        with st.expander(f"{item_type.upper()} | {reason} | {priority} | {item_id}"):
            st.write(f"**Created:** {created_at}")
            if source_doc:
                st.write(f"**Source document:** {source_doc}")

            st.json(payload)

            edge_description = None
            edge_evidence = None
            edge_confidence = None
            edge_provenance = None
            if item_type == "edge" and _is_edge_payload(payload):
                edge_description = st.text_area(
                    "Edge description",
                    value=payload.get("description", "") or "",
                    key=f"edge_desc_{item_id}",
                )
                evidence_text = st.text_area(
                    "Evidence quotes (one per line)",
                    value="\n".join(payload.get("evidence_quotes", []) or []),
                    key=f"edge_evidence_{item_id}",
                )
                edge_evidence = [line.strip() for line in evidence_text.splitlines() if line.strip()]
                default_conf = payload.get("confidence")
                if default_conf is None:
                    default_conf = payload.get("weight", 0.0) or 0.0
                edge_confidence = st.number_input(
                    "Confidence (0.0 - 1.0)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(default_conf),
                    step=0.01,
                    key=f"edge_conf_{item_id}",
                )
                edge_provenance = st.text_input(
                    "Provenance",
                    value=payload.get("provenance", "") or "",
                    key=f"edge_prov_{item_id}",
                )

            notes = st.text_area(
                "Review notes",
                key=f"notes_{item_id}",
                placeholder="Why you approved/rejected this item...",
            )

            can_commit_node = item_type == "node" and _is_node_payload(payload)
            can_commit_edge = item_type == "edge" and _is_edge_payload(payload)
            can_commit = can_commit_node or can_commit_edge

            selected_edge_type = None
            if can_commit_edge:
                # Load active edge types from SchemaRegistry (dynamic)
                try:
                    registry = SchemaRegistry(client.driver)
                    edge_type_options = registry.get_active_type_names()
                    # Filter out EXTRACTED_FROM as it's not a user-facing edge type
                    edge_type_options = [t for t in edge_type_options if t != "EXTRACTED_FROM"]
                except Exception:
                    edge_type_options = [
                        "PERFORMS_FUNCTION",
                        "BASED_ON_PRINCIPLE",
                        "HAS_PROPERTY",
                        "USES_COMPONENT",
                        "RELATED_TO",
                    ]
                current_edge_type = payload.get("edge_collection")
                if current_edge_type and current_edge_type not in edge_type_options:
                    edge_type_options.append(current_edge_type)
                default_index = edge_type_options.index(current_edge_type) if current_edge_type in edge_type_options else 0
                selected_edge_type = st.selectbox(
                    "Select edge type to commit",
                    options=edge_type_options,
                    index=default_index,
                    key=f"edge_type_{item_id}",
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Approve & Commit", key=f"approve_{item_id}", disabled=not can_commit):
                    try:
                        if can_commit_node:
                            _commit_node(agent, payload, source_doc)
                        elif can_commit_edge:
                            commit_payload = dict(payload)
                            if selected_edge_type:
                                commit_payload["edge_collection"] = selected_edge_type
                            if edge_description is not None:
                                commit_payload["description"] = edge_description
                            if edge_evidence is not None:
                                commit_payload["evidence_quotes"] = edge_evidence
                            if edge_confidence is not None:
                                commit_payload["confidence"] = edge_confidence
                                commit_payload["weight"] = edge_confidence
                            if edge_provenance is not None:
                                commit_payload["provenance"] = edge_provenance
                            _commit_edge(agent, commit_payload)
                        _update_review_status(
                            client,
                            item_id,
                            status="approved",
                            reviewer=reviewer,
                            notes=notes,
                            action="commit",
                        )
                        st.success("Committed and marked approved.")
                    except Exception as exc:
                        st.error(f"Commit failed: {exc}")

            with col2:
                if st.button("Approve (no commit)", key=f"approve_no_commit_{item_id}"):
                    _update_review_status(
                        client,
                        item_id,
                        status="approved",
                        reviewer=reviewer,
                        notes=notes,
                        action="approve_no_commit",
                    )
                    st.success("Marked approved without committing.")

            with col3:
                if st.button("Reject", key=f"reject_{item_id}"):
                    _update_review_status(
                        client,
                        item_id,
                        status="rejected",
                        reviewer=reviewer,
                        notes=notes,
                        action="reject",
                    )
                    st.warning("Marked rejected.")

            if not can_commit:
                st.info("This item does not include a full node/edge payload; commit is disabled.")


if __name__ == "__main__":
    main()
