"""Vector+LLM cascade for resolving PDK_Cell relationships.

Handles:
- IMPLEMENTS  (PDK_Cell → Component)
- PERFORMS_FUNCTION  (PDK_Cell → Design_Function)
- FABRICATED_WITH  (PDK_Cell → Physical_Principle)
- EXHIBITS  (PDK_Cell → Property)
- Topology completion for delegated composition patterns
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from PhotonicsAI.Photon import llm_api

from .models import (
    BatchImplementsResolution,
    BatchRelationshipResolution,
    ImplementsResolution,
    LayoutComposition,
    PDKCellNode,
    RelationshipResolution,
)

if TYPE_CHECKING:
    from PhotonicsAI.KnowledgeBase.Neo4j.client import Neo4jClient

from pydantic import BaseModel, ValidationError


def _try_parse_bare_list(exc: Exception, item_model: type[BaseModel]) -> Optional[list]:
    """When callgoogle_pydantic fails because the LLM returned a bare JSON array
    instead of ``{"results": [...]}``, extract the list from the ValidationError
    input and parse each item individually.
    """
    if not isinstance(exc, ValidationError):
        return None
    for error in exc.errors():
        raw = error.get("input")
        if isinstance(raw, list):
            items = []
            for entry in raw:
                try:
                    items.append(item_model.model_validate(entry))
                except Exception:
                    continue
            if items:
                return items
    return None


# ---------------------------------------------------------------------------
# Reference table for GDSFactory compound component internal topologies
# ---------------------------------------------------------------------------
GF_COMPOUND_TOPOLOGIES = {
    "mzi": {
        "description": "Mach-Zehnder interferometer with splitter, combiner, and two arms",
        "roles": ["splitter", "combiner", "straight_x_top", "straight_x_bot"],
        "internal_connections": [
            {"from_role": "splitter", "from_port": "o3", "to_role": "straight_x_top", "to_port": "o1"},
            {"from_role": "splitter", "from_port": "o4", "to_role": "straight_x_bot", "to_port": "o1"},
            {"from_role": "straight_x_top", "from_port": "o2", "to_role": "combiner", "to_port": "o3"},
            {"from_role": "straight_x_bot", "from_port": "o2", "to_role": "combiner", "to_port": "o4"},
        ],
    },
    "mzi2x2_2x2": {
        "description": "2x2 MZI with 2x2 splitter and combiner",
        "roles": ["splitter", "combiner", "straight_x_top", "straight_x_bot"],
        "internal_connections": [
            {"from_role": "splitter", "from_port": "o3", "to_role": "straight_x_top", "to_port": "o1"},
            {"from_role": "splitter", "from_port": "o4", "to_role": "straight_x_bot", "to_port": "o1"},
            {"from_role": "straight_x_top", "from_port": "o2", "to_role": "combiner", "to_port": "o3"},
            {"from_role": "straight_x_bot", "from_port": "o2", "to_role": "combiner", "to_port": "o4"},
        ],
    },
    "mzi1x2_2x2": {
        "description": "1x2 MZI with 1x2 splitter and 2x2 combiner",
        "roles": ["splitter", "combiner", "straight_x_top", "straight_x_bot"],
        "internal_connections": [
            {"from_role": "splitter", "from_port": "o2", "to_role": "straight_x_top", "to_port": "o1"},
            {"from_role": "splitter", "from_port": "o3", "to_role": "straight_x_bot", "to_port": "o1"},
            {"from_role": "straight_x_top", "from_port": "o2", "to_role": "combiner", "to_port": "o3"},
            {"from_role": "straight_x_bot", "from_port": "o2", "to_role": "combiner", "to_port": "o4"},
        ],
    },
    "ring_double": {
        "description": "Double-bus ring resonator",
        "roles": [],
        "internal_connections": [],
    },
    "ring_single": {
        "description": "Single-bus ring resonator",
        "roles": [],
        "internal_connections": [],
    },
}


# ---------------------------------------------------------------------------
# Topology completion (Phase 2b)
# ---------------------------------------------------------------------------

_TOPOLOGY_SYS_PROMPT = """\
You are a photonic integrated circuit (PIC) expert. Given a GDSFactory cell function's
source code and partial AST extraction, produce a complete ArchitectureTemplate JSON.

The ArchitectureTemplate has this schema:
{
  "name": "<component name>",
  "topology_class": "<tree|cascade|mesh|ring-bus|mzi|...>",
  "component_roles": [
    {
      "role_id": "<unique id>",
      "component_type": "<what fills this role>",
      "port_config": "<NxM or null>",
      "quantity": <int>,
      "alternatives": []
    }
  ],
  "connections": [
    {
      "from_role": "<role_id>",
      "from_port": "<port name>",
      "to_role": "<role_id>",
      "to_port": "<port name>"
    }
  ],
  "scaling_rules": null,
  "variants": [],
  "source": "pdk:<pdk_name>",
  "confidence": <0.0-1.0>
}

Reference table for GDSFactory compound components:
"""


def complete_topology(
    cell_source: str,
    composition: LayoutComposition,
    pdk_name: str,
) -> Optional[dict]:
    """Use LLM to produce a complete ArchitectureTemplate for a delegated composition.

    Returns the template as a dict, or None on failure.
    """
    if composition.composition_pattern not in ("delegated", "explicit"):
        return None

    ref_table = json.dumps(GF_COMPOUND_TOPOLOGIES, indent=2)
    ast_summary = composition.model_dump_json(indent=2)

    prompt = f"""Analyze this GDSFactory cell function and produce a complete ArchitectureTemplate JSON.

## Cell source code:
```python
{cell_source}
```

## Partial AST extraction:
```json
{ast_summary}
```

Produce the ArchitectureTemplate JSON. Use the role_id values from the AST extraction
where available. For delegated patterns, use the reference table to fill in internal
connections. Set source to "pdk:{pdk_name}" and confidence between 0.85-0.95.
"""
    sys_prompt = _TOPOLOGY_SYS_PROMPT + ref_table

    try:
        from PhotonicsAI.KnowledgeBase.models.architecture_template import ArchitectureTemplate
        result = llm_api.callgoogle_pydantic(
            prompt=prompt,
            sys_prompt=sys_prompt,
            pydantic_model=ArchitectureTemplate,
        )
        return result.model_dump()
    except Exception as e:
        print(f"  [TopologyCompletion] Failed for {composition.module_name}: {e}")
        return None


# ---------------------------------------------------------------------------
# IMPLEMENTS resolution (Phase 3)
# ---------------------------------------------------------------------------

_IMPLEMENTS_SYS_PROMPT = """\
You are a photonic integrated circuit expert. For each PDK cell below, determine
whether it implements any of the candidate abstract Component concepts from the
knowledge graph.

A PDK cell "implements" a Component if the cell is a concrete realization of
that abstract concept. For example, an "mzi_2x2_heater_tin_cband" implements
"MZI" (Mach-Zehnder Interferometer).

Return a JSON object with a "results" key containing an array of results, one per
candidate. Set does_implement=true only when there is a clear conceptual match.
Provide confidence (0.0-1.0) and brief reasoning.
"""


def resolve_implements(
    cell: PDKCellNode,
    kb_client: "Neo4jClient",
    threshold: float = 0.5,
    top_k: int = 5,
) -> list[ImplementsResolution]:
    """Vector search + LLM cascade for IMPLEMENTS resolution."""
    query = f"{cell.display_name}. {cell.description}"
    if cell.aka:
        query += f" Also known as: {cell.aka}."

    candidates = kb_client.semantic_search(
        query_text=query,
        collection="Components",
        limit=top_k,
        threshold=threshold,
    )

    if not candidates:
        return []

    # Build prompt
    cell_info = (
        f"PDK Cell: {cell.module_name}\n"
        f"Display Name: {cell.display_name}\n"
        f"Description: {cell.description}\n"
        f"Labels: {', '.join(cell.labels)}\n"
        f"AKA: {cell.aka}\n"
        f"Technology: {cell.technology}\n"
        f"Ports: {cell.ports}\n"
    )

    candidate_lines = []
    for c in candidates:
        c_name = c.get("name", "")
        c_desc = (c.get("description", "") or "")[:300]
        c_score = c.get("score", 0.0)
        candidate_lines.append(f"- {c_name} (similarity: {c_score:.3f}): {c_desc}")

    prompt = (
        f"## PDK Cell:\n{cell_info}\n\n"
        f"## Candidate Components:\n" + "\n".join(candidate_lines) + "\n\n"
        "For each candidate, respond with does_implement, confidence, and reasoning."
    )

    try:
        result = llm_api.callgoogle_pydantic(
            prompt=prompt,
            sys_prompt=_IMPLEMENTS_SYS_PROMPT,
            pydantic_model=BatchImplementsResolution,
        )
        return result.results
    except Exception as e:
        # LLM may return a bare list instead of {"results": [...]}
        parsed = _try_parse_bare_list(e, ImplementsResolution)
        if parsed is not None:
            return parsed
        print(f"  [IMPLEMENTS] LLM resolution failed for {cell.module_name}: {e}")
        return []


# ---------------------------------------------------------------------------
# Generic relationship resolution (Phase 4)
# ---------------------------------------------------------------------------

_RELATIONSHIP_SYS_PROMPT = """\
You are a photonic integrated circuit expert. For each candidate pair below,
determine whether a relationship exists between the PDK cell and the candidate
knowledge graph entity.

Return a JSON object with a "results" key containing an array of results.
For each candidate, provide:
- candidate_name: the name of the candidate entity
- is_related: true/false
- confidence: 0.0-1.0
- edge_type: the relationship type (one of: {edge_types})
- reasoning: brief explanation

Only confirm relationships that are clearly valid. When in doubt, set is_related=false.
"""


def resolve_relationships(
    cell: PDKCellNode,
    kb_client: "Neo4jClient",
    target_collection: str,
    edge_type: str,
    query_text: str,
    threshold: float = 0.5,
    top_k: int = 5,
) -> list[RelationshipResolution]:
    """Generic vector+LLM cascade for relationship resolution.

    Used for PERFORMS_FUNCTION, FABRICATED_WITH, and EXHIBITS.
    """
    candidates = kb_client.semantic_search(
        query_text=query_text,
        collection=target_collection,
        limit=top_k,
        threshold=threshold,
    )

    if not candidates:
        return []

    cell_info = (
        f"PDK Cell: {cell.module_name}\n"
        f"Display Name: {cell.display_name}\n"
        f"Description: {cell.description}\n"
        f"Labels: {', '.join(cell.labels)}\n"
        f"Technology: {cell.technology}\n"
    )

    candidate_lines = []
    for c in candidates:
        c_name = c.get("name", "")
        c_desc = (c.get("description", "") or "")[:300]
        c_score = c.get("score", 0.0)
        candidate_lines.append(f"- {c_name} (similarity: {c_score:.3f}): {c_desc}")

    prompt = (
        f"## PDK Cell:\n{cell_info}\n\n"
        f"## Candidate {target_collection}:\n" + "\n".join(candidate_lines) + "\n\n"
        f"Determine which candidates have a {edge_type} relationship with this PDK cell."
    )

    sys_prompt = _RELATIONSHIP_SYS_PROMPT.format(edge_types=edge_type)

    try:
        result = llm_api.callgoogle_pydantic(
            prompt=prompt,
            sys_prompt=sys_prompt,
            pydantic_model=BatchRelationshipResolution,
        )
        return result.results
    except Exception as e:
        parsed = _try_parse_bare_list(e, RelationshipResolution)
        if parsed is not None:
            return parsed
        print(f"  [{edge_type}] LLM resolution failed for {cell.module_name}: {e}")
        return []


def resolve_performs_function(
    cell: PDKCellNode, kb_client: "Neo4jClient"
) -> list[RelationshipResolution]:
    """Resolve PERFORMS_FUNCTION edges."""
    # Filter labels that aren't useful for function search
    skip_labels = {"active", "passive", "1x1", "1x2", "2x2", "1x4", "2x4"}
    useful_labels = [l for l in cell.labels if l.lower() not in skip_labels]
    query = f"{cell.display_name}. {' '.join(useful_labels)}"
    if cell.description:
        query += f" {cell.description[:200]}"

    return resolve_relationships(
        cell=cell,
        kb_client=kb_client,
        target_collection="Design_Functions",
        edge_type="PERFORMS_FUNCTION",
        query_text=query,
    )


def resolve_fabricated_with(
    cell: PDKCellNode, kb_client: "Neo4jClient"
) -> list[RelationshipResolution]:
    """Resolve FABRICATED_WITH edges."""
    if not cell.technology:
        return []
    query = f"{cell.technology}. {cell.display_name}"
    return resolve_relationships(
        cell=cell,
        kb_client=kb_client,
        target_collection="Physical_Principles",
        edge_type="FABRICATED_WITH",
        query_text=query,
    )


def resolve_exhibits(
    cell: PDKCellNode, kb_client: "Neo4jClient"
) -> list[RelationshipResolution]:
    """Resolve EXHIBITS edges using parsed numeric specs."""
    if not cell.numeric_specs:
        return []

    all_results: list[RelationshipResolution] = []
    for spec in cell.numeric_specs:
        query = spec.field_name
        results = resolve_relationships(
            cell=cell,
            kb_client=kb_client,
            target_collection="Properties",
            edge_type="EXHIBITS",
            query_text=query,
            top_k=3,
        )
        # Attach spec metadata to matching results
        for r in results:
            if r.is_related:
                r.reasoning += f" [value={spec.value} {spec.unit}, raw='{spec.raw_text}']"
        all_results.extend(results)

    return all_results
