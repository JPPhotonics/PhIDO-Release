"""Schema Evolution Agent (SEA).

Analyses accumulated RelationshipObservation nodes, clusters them, applies
statistical promotion gates, validates with an LLM, and recategorizes existing
RELATED_TO edges using a hybrid auto-commit / review-queue strategy.
"""

import datetime
import json
from collections import Counter
from statistics import mean as _mean
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np

from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient
from PhotonicsAI.KnowledgeBase.Neo4j.schema_registry import SchemaRegistry
from PhotonicsAI.Photon import llm_api

from .models import (
    EvolutionReport,
    ObservationCluster,
    PromotionCandidate,
    RecategorizationAction,
    RecategorizationVerification,
    SchemaValidationResult,
)
from .prompts import (
    EDGE_BLOCK_TEMPLATE,
    RECATEGORIZATION_SYS_PROMPT,
    SCHEMA_VALIDATION_SYS_PROMPT,
    build_recategorization_user_prompt,
    build_schema_validation_user_prompt,
)


class SchemaEvolutionAgent:
    """
    Schema Evolution Agent (SEA).

    Discovers new relationship types from accumulated cross-document
    observations, promotes them through statistical and LLM gates, and
    recategorizes existing ``RELATED_TO`` edges.
    """

    # Tunable knobs --------------------------------------------------------
    PROMOTION_THRESHOLD: int = 3        # min distinct documents
    CLUSTER_SIMILARITY: float = 0.85    # cosine threshold for grouping
    MIN_CONFIDENCE_MEAN: float = 0.70   # avg confidence floor
    RECAT_AUTO_THRESHOLD: float = 0.85  # auto-commit recategorization above this
    RECAT_BATCH_SIZE: int = 15          # edges per LLM recategorization call
    DIRECTIONAL_CONSISTENCY: float = 0.6  # 60% agreement on source/target types

    def __init__(
        self,
        kb_client: Neo4jClient,
        schema_registry: SchemaRegistry,
        llm_model: str = "gemini-2.5-pro",
    ) -> None:
        self.kb_client = kb_client
        self.registry = schema_registry
        self.llm_model = llm_model

    # ======================================================================
    # Main entry point
    # ======================================================================

    def evolve(self) -> EvolutionReport:
        """Run one full schema evolution pass and return a report."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report = EvolutionReport(generated_at=now)

        # Phase 1 – Cluster observations
        print("SEA Phase 1: Clustering observations...")
        clusters = self._cluster_observations()
        print(f"  Found {len(clusters)} clusters.")

        if not clusters:
            return report

        # Phase 2 – Statistical gates
        print("SEA Phase 2: Evaluating promotion candidates...")
        candidates = self._evaluate_candidates(clusters)
        report.clusters_below_threshold = len(clusters) - len(candidates)
        report.candidates_pending = [
            c.canonical_name
            for c in clusters
            if c.cluster_id not in {cand.cluster.cluster_id for cand in candidates}
        ]
        print(f"  {len(candidates)} candidates passed statistical gates.")

        if not candidates:
            return report

        # Phase 3 – LLM validation
        print("SEA Phase 3: LLM validation...")
        validated = self._validate_with_llm(candidates)
        print(f"  {len(validated)} candidates validated by LLM.")

        if not validated:
            return report

        # Phase 4 – Promote & recategorize
        print("SEA Phase 4: Promoting and recategorizing...")
        auto_count, review_count, actions = self._promote_and_recategorize(validated)
        report.edges_recategorized_auto = auto_count
        report.edges_queued_for_review = review_count
        report.recategorization_actions = actions
        report.new_types_promoted = [
            v.llm_canonical_name or v.cluster.canonical_name for v in validated
        ]

        print(f"SEA: Done. Promoted {len(report.new_types_promoted)} types, "
              f"recategorized {auto_count} edges, queued {review_count} for review.")
        return report

    # ======================================================================
    # Phase 1 – Cluster observations
    # ======================================================================

    def _cluster_observations(self) -> List[ObservationCluster]:
        """Fetch un-clustered observations, embed & cluster them."""
        observations = self._fetch_unclustered_observations()
        if not observations:
            return []

        # Ensure every observation has an embedding
        self._backfill_embeddings(observations)

        # Collect embeddings into a matrix
        valid_obs = [o for o in observations if o.get("embedding")]
        if not valid_obs:
            return []

        embeddings = np.array([o["embedding"] for o in valid_obs], dtype=np.float32)

        # Agglomerative clustering with cosine distance
        labels = self._run_clustering(embeddings)

        # Group observations by cluster label
        cluster_map: Dict[int, List[Dict[str, Any]]] = {}
        for label, obs in zip(labels, valid_obs):
            cluster_map.setdefault(int(label), []).append(obs)

        # Build ObservationCluster objects
        clusters: List[ObservationCluster] = []
        for label_id, obs_list in cluster_map.items():
            cluster = self._build_cluster(label_id, obs_list)
            clusters.append(cluster)

        # Write cluster_id back to Neo4j
        self._persist_cluster_ids(clusters)

        return clusters

    def _fetch_unclustered_observations(self) -> List[Dict[str, Any]]:
        """Query RelationshipObservation nodes that have not yet been clustered."""
        query = """
        MATCH (o:RelationshipObservation)
        WHERE o.cluster_id IS NULL
        RETURN o
        """
        results: List[Dict[str, Any]] = []
        try:
            with self.kb_client.driver.session() as session:
                for record in session.run(query):
                    node = record["o"]
                    data = dict(node)
                    data["_element_id"] = node.element_id if hasattr(node, "element_id") else None
                    results.append(data)
        except Exception as e:
            print(f"  [SEA] Failed to fetch observations: {e}")
        return results

    def _backfill_embeddings(self, observations: List[Dict[str, Any]]) -> None:
        """Generate and store embeddings for observations that lack them."""
        importer = self.kb_client.importer
        if importer is None:
            return
        for obs in observations:
            if obs.get("embedding"):
                continue
            text = f"{obs.get('proposed_type', '')}: {obs.get('description', '')}"
            emb = importer.generate_embedding(text)
            if emb:
                obs["embedding"] = emb
                # Persist to Neo4j
                eid = obs.get("_element_id")
                if eid:
                    try:
                        with self.kb_client.driver.session() as session:
                            session.run(
                                "MATCH (o) WHERE elementId(o) = $eid SET o.embedding = $emb",
                                eid=eid,
                                emb=emb,
                            )
                    except Exception:
                        pass

    def _run_clustering(self, embeddings: np.ndarray) -> List[int]:
        """Run agglomerative clustering on embedding vectors."""
        n_samples = embeddings.shape[0]
        if n_samples == 1:
            return [0]

        try:
            from sklearn.cluster import AgglomerativeClustering

            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=1.0 - self.CLUSTER_SIMILARITY,
            )
            labels = clustering.fit_predict(embeddings).tolist()
        except ImportError:
            # Fallback: treat each observation as its own cluster
            print("  [SEA] sklearn not available; falling back to single-observation clusters.")
            labels = list(range(n_samples))

        return labels

    def _build_cluster(
        self, label_id: int, obs_list: List[Dict[str, Any]]
    ) -> ObservationCluster:
        """Aggregate observations into an ObservationCluster."""
        cluster_id = str(uuid4())

        # Canonical name = most frequent proposed_type
        type_counts = Counter(o.get("proposed_type", "") for o in obs_list)
        canonical_name = type_counts.most_common(1)[0][0] if type_counts else "UNKNOWN"

        # Merged description
        unique_descs = list({o.get("description", "") for o in obs_list if o.get("description")})
        merged_description = " | ".join(unique_descs[:5])  # cap to avoid huge strings

        # Distinct documents
        docs = {o.get("source_document", "") for o in obs_list if o.get("source_document")}
        distinct_documents = len(docs)

        # Mean confidence
        confidences = [float(o.get("confidence", 0.0) or 0.0) for o in obs_list]
        mean_conf = _mean(confidences) if confidences else 0.0

        # Dominant source/target types
        src_counts = Counter(o.get("from_entity_type", "") for o in obs_list)
        tgt_counts = Counter(o.get("to_entity_type", "") for o in obs_list)
        dominant_src = self._dominant_types(src_counts, len(obs_list))
        dominant_tgt = self._dominant_types(tgt_counts, len(obs_list))

        return ObservationCluster(
            cluster_id=cluster_id,
            canonical_name=canonical_name,
            merged_description=merged_description,
            observations=obs_list,
            distinct_documents=distinct_documents,
            mean_confidence=mean_conf,
            dominant_source_types=dominant_src,
            dominant_target_types=dominant_tgt,
        )

    def _dominant_types(self, counts: Counter, total: int) -> List[str]:
        """Return entity types that represent >= DIRECTIONAL_CONSISTENCY of observations."""
        if total == 0:
            return []
        return [
            t for t, c in counts.most_common()
            if t and c / total >= self.DIRECTIONAL_CONSISTENCY
        ]

    def _persist_cluster_ids(self, clusters: List[ObservationCluster]) -> None:
        """Write cluster_id back to each observation node in Neo4j."""
        try:
            with self.kb_client.driver.session() as session:
                for cluster in clusters:
                    obs_ids = [
                        o.get("id") for o in cluster.observations if o.get("id")
                    ]
                    if obs_ids:
                        session.run(
                            """
                            UNWIND $ids AS oid
                            MATCH (o:RelationshipObservation {id: oid})
                            SET o.cluster_id = $cluster_id
                            """,
                            ids=obs_ids,
                            cluster_id=cluster.cluster_id,
                        )
        except Exception as e:
            print(f"  [SEA] Failed to persist cluster IDs: {e}")

    # ======================================================================
    # Phase 2 – Statistical gates
    # ======================================================================

    def _evaluate_candidates(
        self, clusters: List[ObservationCluster]
    ) -> List[PromotionCandidate]:
        """Apply statistical gates to filter clusters into promotion candidates."""
        candidates: List[PromotionCandidate] = []
        for cluster in clusters:
            # Gate 1: document diversity
            if cluster.distinct_documents < self.PROMOTION_THRESHOLD:
                continue

            # Gate 2: confidence floor
            if cluster.mean_confidence < self.MIN_CONFIDENCE_MEAN:
                continue

            # Gate 3: directional consistency
            if not cluster.dominant_source_types or not cluster.dominant_target_types:
                continue

            # Gate 4: not already an active type
            if self.registry.is_active_type(cluster.canonical_name):
                continue

            candidates.append(PromotionCandidate(cluster=cluster))

        return candidates

    # ======================================================================
    # Phase 3 – LLM validation
    # ======================================================================

    def _validate_with_llm(
        self, candidates: List[PromotionCandidate]
    ) -> List[PromotionCandidate]:
        """Use LLM to confirm each candidate is genuinely distinct."""
        schema_block = self.registry.get_prompt_schema_block()
        validated: List[PromotionCandidate] = []

        for cand in candidates:
            cluster = cand.cluster

            # Build sample observations text
            samples = cluster.observations[:5]
            sample_lines = []
            for s in samples:
                sample_lines.append(
                    f"- {s.get('from_entity', '?')} ({s.get('from_entity_type', '?')}) "
                    f"-> {s.get('to_entity', '?')} ({s.get('to_entity_type', '?')}): "
                    f"{s.get('description', 'N/A')}"
                )
            sample_text = "\n".join(sample_lines)

            user_prompt = build_schema_validation_user_prompt(
                existing_schema_block=schema_block,
                candidate_name=cluster.canonical_name,
                candidate_description=cluster.merged_description,
                dominant_source_types=", ".join(cluster.dominant_source_types),
                dominant_target_types=", ".join(cluster.dominant_target_types),
                sample_observations=sample_text,
            )

            try:
                result: SchemaValidationResult = llm_api.callgoogle_pydantic(
                    prompt=user_prompt,
                    sys_prompt=SCHEMA_VALIDATION_SYS_PROMPT,
                    pydantic_model=SchemaValidationResult,
                )

                if result.is_distinct:
                    cand.llm_validated = True
                    cand.llm_canonical_name = result.canonical_name
                    cand.llm_description = result.description
                    cand.llm_source_types = result.source_types
                    cand.llm_target_types = result.target_types
                    cand.llm_reason = result.reason
                    validated.append(cand)
                    print(f"  [SEA] Validated: {result.canonical_name} — {result.reason}")
                else:
                    print(f"  [SEA] Rejected: {cluster.canonical_name} — {result.reason}")
            except Exception as e:
                print(f"  [SEA] LLM validation failed for {cluster.canonical_name}: {e}")

        return validated

    # ======================================================================
    # Phase 4 – Promote & recategorize
    # ======================================================================

    def _promote_and_recategorize(
        self, validated: List[PromotionCandidate]
    ) -> Tuple[int, int, List[RecategorizationAction]]:
        """Promote validated types and recategorize matching RELATED_TO edges."""
        total_auto = 0
        total_review = 0
        all_actions: List[RecategorizationAction] = []

        for cand in validated:
            name = cand.llm_canonical_name or cand.cluster.canonical_name
            description = cand.llm_description or cand.cluster.merged_description
            source_types = cand.llm_source_types or cand.cluster.dominant_source_types
            target_types = cand.llm_target_types or cand.cluster.dominant_target_types

            # 1. Promote
            self.registry.promote_candidate(
                name=name,
                description=description,
                allowed_sources=source_types,
                allowed_targets=target_types,
            )
            print(f"  [SEA] Promoted type: {name}")

            # 2. Recategorize matching RELATED_TO edges
            auto, review, actions = self._recategorize_related_to_edges(
                name, description, source_types, target_types
            )
            total_auto += auto
            total_review += review
            all_actions.extend(actions)

        return total_auto, total_review, all_actions

    def _recategorize_related_to_edges(
        self,
        new_type_name: str,
        new_type_description: str,
        source_types: List[str],
        target_types: List[str],
    ) -> Tuple[int, int, List[RecategorizationAction]]:
        """Find and recategorize RELATED_TO edges matching the new type."""
        # Query matching RELATED_TO edges
        query = """
        MATCH (a)-[r:RELATED_TO]->(b)
        WHERE any(label IN labels(a) WHERE label IN $source_types)
          AND any(label IN labels(b) WHERE label IN $target_types)
        RETURN elementId(a) AS from_id, a.name AS from_name, labels(a) AS from_labels,
               elementId(b) AS to_id, b.name AS to_name, labels(b) AS to_labels,
               r.description AS desc,
               r.evidence_quotes AS evidence,
               elementId(r) AS rel_id
        LIMIT 200
        """
        edges: List[Dict[str, Any]] = []
        try:
            with self.kb_client.driver.session() as session:
                for record in session.run(
                    query, source_types=source_types, target_types=target_types
                ):
                    edges.append({
                        "from_id": record["from_id"],
                        "from_name": record["from_name"],
                        "from_type": record["from_labels"][0] if record["from_labels"] else "",
                        "to_id": record["to_id"],
                        "to_name": record["to_name"],
                        "to_type": record["to_labels"][0] if record["to_labels"] else "",
                        "desc": record["desc"] or "",
                        "evidence": record["evidence"] or [],
                        "rel_id": record["rel_id"],
                    })
        except Exception as e:
            print(f"  [SEA] Failed to query RELATED_TO edges: {e}")
            return 0, 0, []

        if not edges:
            return 0, 0, []

        print(f"  [SEA] Found {len(edges)} RELATED_TO edges to evaluate for {new_type_name}.")

        # Batch LLM verification
        auto_count = 0
        review_count = 0
        actions: List[RecategorizationAction] = []

        sys_prompt = RECATEGORIZATION_SYS_PROMPT.format(
            new_type_name=new_type_name,
            new_type_description=new_type_description,
            source_types=", ".join(source_types),
            target_types=", ".join(target_types),
        )

        batches = [
            edges[i : i + self.RECAT_BATCH_SIZE]
            for i in range(0, len(edges), self.RECAT_BATCH_SIZE)
        ]

        for batch_idx, batch in enumerate(batches):
            # Build prompt
            block_parts = []
            edge_id_map: Dict[str, Dict[str, Any]] = {}
            for idx, e in enumerate(batch):
                eid = f"edge_{batch_idx}_{idx}"
                evidence_str = "; ".join(e["evidence"][:3]) if e["evidence"] else "N/A"
                block_parts.append(
                    EDGE_BLOCK_TEMPLATE.format(
                        edge_id=eid,
                        from_name=e["from_name"],
                        from_type=e["from_type"],
                        to_name=e["to_name"],
                        to_type=e["to_type"],
                        description=e["desc"][:300],
                        evidence=evidence_str[:300],
                    )
                )
                edge_id_map[eid] = e

            user_prompt = build_recategorization_user_prompt("\n".join(block_parts))

            try:
                result: RecategorizationVerification = llm_api.callgoogle_pydantic(
                    prompt=user_prompt,
                    sys_prompt=sys_prompt,
                    pydantic_model=RecategorizationVerification,
                )

                for item in result.results:
                    edge_data = edge_id_map.get(item.edge_id)
                    if not edge_data:
                        continue

                    if not item.should_retype:
                        continue

                    if item.confidence >= self.RECAT_AUTO_THRESHOLD:
                        # Auto-commit: delete old RELATED_TO, create new typed edge
                        success = self._retype_edge(
                            edge_data["rel_id"],
                            edge_data["from_id"],
                            edge_data["to_id"],
                            new_type_name,
                        )
                        if success:
                            auto_count += 1
                            actions.append(
                                RecategorizationAction(
                                    edge_from=edge_data["from_name"],
                                    edge_to=edge_data["to_name"],
                                    old_type="RELATED_TO",
                                    new_type=new_type_name,
                                    confidence=item.confidence,
                                    action="committed",
                                )
                            )
                    else:
                        # Queue for human review
                        self._queue_recategorization_review(
                            edge_data, new_type_name, item.confidence, item.reasoning
                        )
                        review_count += 1
                        actions.append(
                            RecategorizationAction(
                                edge_from=edge_data["from_name"],
                                edge_to=edge_data["to_name"],
                                old_type="RELATED_TO",
                                new_type=new_type_name,
                                confidence=item.confidence,
                                action="queued_for_review",
                            )
                        )
            except Exception as e:
                print(f"  [SEA] Recategorization batch {batch_idx} failed: {e}")

        return auto_count, review_count, actions

    # ------------------------------------------------------------------
    # Graph mutation helpers
    # ------------------------------------------------------------------

    def _retype_edge(
        self,
        rel_element_id: str,
        from_element_id: str,
        to_element_id: str,
        new_type: str,
    ) -> bool:
        """Delete old RELATED_TO edge and create a new typed edge.

        Neo4j doesn't support changing relationship types in-place, so we
        copy properties, delete the old relationship, and create a new one.
        """
        try:
            with self.kb_client.driver.session() as session:
                # Read properties from old relationship
                props_result = session.run(
                    """
                    MATCH ()-[r]->()
                    WHERE elementId(r) = $rel_id
                    RETURN properties(r) AS props
                    """,
                    rel_id=rel_element_id,
                )
                record = props_result.single()
                props = dict(record["props"]) if record else {}
                # Remove internal keys
                props.pop("embedding", None)

                # Delete old + create new in one transaction
                session.run(
                    f"""
                    MATCH (a) WHERE elementId(a) = $from_id
                    MATCH (b) WHERE elementId(b) = $to_id
                    MATCH (a)-[r:RELATED_TO]->(b) WHERE elementId(r) = $rel_id
                    DELETE r
                    CREATE (a)-[r2:{new_type}]->(b)
                    SET r2 = $props
                    """,
                    from_id=from_element_id,
                    to_id=to_element_id,
                    rel_id=rel_element_id,
                    props=props,
                )
                return True
        except Exception as e:
            print(f"  [SEA] Failed to retype edge {rel_element_id}: {e}")
            return False

    def _queue_recategorization_review(
        self,
        edge_data: Dict[str, Any],
        new_type: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Create a ReviewItem node for a low-confidence recategorization."""
        if not hasattr(self.kb_client, "driver") or self.kb_client.driver is None:
            return

        item_id = str(uuid4())
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()

        payload = {
            "edge_collection": new_type,
            "from_node": edge_data.get("from_name", ""),
            "from_collection": edge_data.get("from_type", ""),
            "to_node": edge_data.get("to_name", ""),
            "to_collection": edge_data.get("to_type", ""),
            "operation": "MERGE",
            "description": f"Schema recategorization from RELATED_TO to {new_type}: {reasoning}",
            "confidence": confidence,
            "provenance": "SEA",
        }

        props = {
            "id": item_id,
            "item_type": "edge",
            "reason": "schema_recategorization",
            "priority": "medium",
            "status": "pending",
            "created_at": now,
            "source_document": "",
            "payload_json": json.dumps(payload, default=str),
        }

        try:
            with self.kb_client.driver.session() as session:
                session.run(
                    "MERGE (r:ReviewItem {id: $id}) SET r += $props",
                    id=item_id,
                    props=props,
                )
        except Exception as e:
            print(f"  [SEA] Failed to queue review item: {e}")
