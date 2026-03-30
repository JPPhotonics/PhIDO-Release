"""Bidirectional enrichment between PDK_Cell and Component entities.

Direction B (PDK → Component): After PDK ingestion, propagate PDK-discovered
relationships to the abstract Component nodes they implement.

Direction A (Component → PDK): After paper processing, propagate paper-discovered
relationships to the concrete PDK_Cell nodes that implement those Components.

Architecture Template Upgrade: When a PDK_Cell has a high-confidence topology_template
and is linked via IMPLEMENTS to a Component that is linked from an Architecture,
propagate the PDK template to supplement the Architecture's paper-derived template.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient


def enrich_components_from_pdk(kb_client: "Neo4jClient") -> int:
    """Direction B: Propagate PDK-discovered edges to linked Components.

    For each (PDK_Cell)-[IMPLEMENTS]->(Component), find edges from the PDK_Cell
    that the Component does not already have and create them with inferred provenance.

    Returns count of edges created.
    """
    query = """\
    MATCH (pdk:PDK_Cell)-[:IMPLEMENTS]->(comp:Component)
    MATCH (pdk)-[r]->(target)
    WHERE type(r) IN ['EXHIBITS', 'PERFORMS_FUNCTION', 'FABRICATED_WITH']
      AND NOT (comp)-[]->(target)
    RETURN pdk.module_name AS pdk_module,
           comp.name AS comp_name,
           type(r) AS rel_type,
           target.name AS target_name,
           labels(comp)[0] AS comp_label,
           labels(target)[0] AS target_label,
           elementId(comp) AS comp_id,
           elementId(target) AS target_id
    """

    # Map PDK edge types to their Component equivalents
    pdk_to_comp_edge = {
        "EXHIBITS": "HAS_PROPERTY",
        "PERFORMS_FUNCTION": "PERFORMS_FUNCTION",
        "FABRICATED_WITH": "BASED_ON_PRINCIPLE",
    }

    count = 0
    with kb_client.driver.session() as session:
        records = list(session.run(query))

    for rec in records:
        comp_edge_type = pdk_to_comp_edge.get(rec["rel_type"])
        if not comp_edge_type:
            continue

        props = {
            "provenance": "inferred_from_pdk",
            "confidence": 0.6,
            "source_pdk_cell": rec["pdk_module"],
        }

        try:
            kb_client.create_edge(
                edge_type=comp_edge_type,
                from_key=rec["comp_id"],
                from_collection=rec["comp_label"],
                to_key=rec["target_id"],
                to_collection=rec["target_label"],
                props=props,
            )
            count += 1
            print(f"  [Enrich B] {rec['comp_name']} --[{comp_edge_type}]--> {rec['target_name']}")
        except Exception as e:
            print(f"  [Enrich B] Failed: {rec['comp_name']} -> {rec['target_name']}: {e}")

    return count


def enrich_pdk_from_components(kb_client: "Neo4jClient") -> int:
    """Direction A: Propagate Component-discovered edges to linked PDK_Cells.

    For each (PDK_Cell)-[IMPLEMENTS]->(Component), find edges from the Component
    that the PDK_Cell does not already have and create them with inferred provenance.

    Returns count of edges created.
    """
    query = """\
    MATCH (comp:Component)-[r]->(target)
    MATCH (pdk:PDK_Cell)-[:IMPLEMENTS]->(comp)
    WHERE type(r) IN ['HAS_PROPERTY', 'PERFORMS_FUNCTION', 'BASED_ON_PRINCIPLE']
      AND NOT (pdk)-[]->(target)
    RETURN comp.name AS comp_name,
           pdk.module_name AS pdk_module,
           type(r) AS rel_type,
           target.name AS target_name,
           labels(pdk)[0] AS pdk_label,
           labels(target)[0] AS target_label,
           elementId(pdk) AS pdk_id,
           elementId(target) AS target_id
    """

    # Map Component edge types to their PDK equivalents
    comp_to_pdk_edge = {
        "HAS_PROPERTY": "EXHIBITS",
        "PERFORMS_FUNCTION": "PERFORMS_FUNCTION",
        "BASED_ON_PRINCIPLE": "FABRICATED_WITH",
    }

    count = 0
    with kb_client.driver.session() as session:
        records = list(session.run(query))

    for rec in records:
        pdk_edge_type = comp_to_pdk_edge.get(rec["rel_type"])
        if not pdk_edge_type:
            continue

        props = {
            "provenance": "inferred_from_literature",
            "confidence": 0.6,
            "source_component": rec["comp_name"],
        }

        try:
            kb_client.create_edge(
                edge_type=pdk_edge_type,
                from_key=rec["pdk_id"],
                from_collection=rec["pdk_label"],
                to_key=rec["target_id"],
                to_collection=rec["target_label"],
                props=props,
            )
            count += 1
            print(f"  [Enrich A] {rec['pdk_module']} --[{pdk_edge_type}]--> {rec['target_name']}")
        except Exception as e:
            print(f"  [Enrich A] Failed: {rec['pdk_module']} -> {rec['target_name']}: {e}")

    return count


def upgrade_architecture_templates(kb_client: "Neo4jClient") -> int:
    """Propagate high-confidence PDK topology templates to Architecture nodes.

    When a PDK_Cell has a topology_template with confidence >= 0.8 and is linked
    via IMPLEMENTS to a Component that is linked from an Architecture via
    USES_COMPONENT, check if the Architecture's template can be supplemented.

    Returns count of Architecture nodes updated.
    """
    query = """\
    MATCH (pdk:PDK_Cell)-[:IMPLEMENTS]->(comp:Component)<-[:USES_COMPONENT]-(arch:Architecture)
    WHERE pdk.topology_template IS NOT NULL
    RETURN arch.name AS arch_name,
           elementId(arch) AS arch_id,
           arch.topology_template AS arch_template,
           pdk.module_name AS pdk_module,
           pdk.topology_template AS pdk_template
    """

    count = 0
    with kb_client.driver.session() as session:
        records = list(session.run(query))

    for rec in records:
        pdk_template_raw = rec["pdk_template"]
        if not pdk_template_raw:
            continue

        try:
            pdk_template = json.loads(pdk_template_raw) if isinstance(pdk_template_raw, str) else pdk_template_raw
        except (json.JSONDecodeError, TypeError):
            continue

        pdk_confidence = pdk_template.get("confidence", 0.0)
        if pdk_confidence < 0.8:
            continue

        arch_template_raw = rec["arch_template"]
        arch_confidence = 0.0
        if arch_template_raw:
            try:
                arch_template = json.loads(arch_template_raw) if isinstance(arch_template_raw, str) else arch_template_raw
                arch_confidence = arch_template.get("confidence", 0.0)
            except (json.JSONDecodeError, TypeError):
                pass

        if pdk_confidence > arch_confidence:
            enriched = pdk_template.copy()
            enriched["source"] = f"pdk:{rec['pdk_module']}"
            try:
                with kb_client.driver.session() as session:
                    session.run(
                        """\
                        MATCH (a) WHERE elementId(a) = $arch_id
                        SET a.topology_template = $template,
                            a.topology_source = $source
                        """,
                        arch_id=rec["arch_id"],
                        template=json.dumps(enriched),
                        source=f"pdk:{rec['pdk_module']}",
                    )
                count += 1
                print(f"  [TemplateUpgrade] {rec['arch_name']} <- {rec['pdk_module']} (conf {pdk_confidence:.2f})")
            except Exception as e:
                print(f"  [TemplateUpgrade] Failed for {rec['arch_name']}: {e}")

    return count
