"""Phase A.7 – Dedicated architecture decomposition.

This module runs *after* the general entity-description phase (A.5) and the
architecture quality gate (A.6).  It takes Architecture entities that still
lack structural detail and makes a focused, second-pass LLM call whose sole
job is to produce a structured ``ArchitectureTemplate`` for each one.

Key differences from the general describer (Phase A.5):
  * Evidence gathering scans the **entire** filtered text, not just a small
    neighbour window.
  * The prompt is specialised for decomposition and explicitly targets
    ``ComponentRole`` + ``AbstractConnection`` JSON output.
  * It can optionally incorporate KG priors (existing ``USES_COMPONENT`` edges)
    to anchor the decomposition on known knowledge.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from PhotonicsAI.Photon import llm_api
from PhotonicsAI.KnowledgeBase.models.architecture_template import (
    AbstractConnection,
    ArchitectureTemplate,
    ComponentRole,
)
from .models import RawEntity


# ---------------------------------------------------------------------------
# Pydantic wrapper for batch LLM output
# ---------------------------------------------------------------------------

class _ArchTemplateItem(BaseModel):
    """Single item returned by the decomposition LLM call."""
    name: str = Field(..., description="Architecture entity name (must match input)")
    topology_class: Optional[str] = Field(None, description="Topology family (tree, cascade, mesh, ring-bus, …)")
    component_roles: list[ComponentRole] = Field(default_factory=list)
    connections: list[AbstractConnection] = Field(default_factory=list)
    scaling_rules: Optional[dict] = Field(None)
    variants: list[str] = Field(default_factory=list)

class _ArchTemplateList(BaseModel):
    architectures: list[_ArchTemplateItem]


# ---------------------------------------------------------------------------
# Evidence gathering helpers
# ---------------------------------------------------------------------------

def _gather_broad_evidence(
    arch_name: str,
    component_names: list[str],
    filtered_text: str,
    *,
    max_chars: int = 16_000,
) -> str:
    """Scan the full paper text for paragraphs that mention the architecture
    or any of its candidate sub-components, then return a bounded evidence block."""

    targets = [arch_name] + component_names
    target_res = [re.compile(re.escape(t), re.IGNORECASE) for t in targets if t]

    paragraphs = re.split(r"\n{2,}", filtered_text)
    relevant: list[str] = []
    total = 0

    for para in paragraphs:
        if any(r.search(para) for r in target_res):
            relevant.append(para.strip())
            total += len(relevant[-1])
            if total >= max_chars:
                break

    return "\n\n".join(relevant)[:max_chars]


def _fetch_kg_prior(
    arch_name: str,
    kb_client,
) -> Optional[dict]:
    """If the architecture already exists in the KG, retrieve its
    USES_COMPONENT neighbours to seed the decomposition."""
    try:
        node = kb_client.find_by_name(arch_name, "architectures")
        if node is None:
            return None
        element_id = node.get("_key")
        if not element_id:
            return None

        query = """
        MATCH (a)-[r:USES_COMPONENT]->(c)
        WHERE elementId(a) = $eid
        RETURN c.name AS component, properties(r) AS props
        """
        components: list[dict] = []
        with kb_client.driver.session() as session:
            result = session.run(query, eid=element_id)
            for record in result:
                components.append({
                    "name": record["component"],
                    **(record["props"] or {}),
                })
        if components:
            return {"architecture": arch_name, "known_components": components}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def decompose_architectures(
    arch_entities: list[RawEntity],
    all_component_entities: list[RawEntity],
    filtered_text: str,
    descriptions: dict[str, dict],
    kb_client=None,
) -> Dict[str, ArchitectureTemplate]:
    """Run focused decomposition for each architecture entity.

    Parameters
    ----------
    arch_entities:
        Architecture-typed ``RawEntity`` objects (may include ones already
        marked valid by Phase A.5 — they are re-processed to upgrade them
        to structured templates).
    all_component_entities:
        All Component-typed ``RawEntity`` objects from the same paper.
    filtered_text:
        Full preprocessed paper text.
    descriptions:
        Output of ``EntityDescriber.describe_entities`` (name → dict).
    kb_client:
        Optional ``Neo4jClient`` for KG prior retrieval.

    Returns
    -------
    Mapping of architecture name → ``ArchitectureTemplate``.
    """
    if not arch_entities:
        return {}

    component_names = sorted({e.name for e in all_component_entities})

    prompt_items: list[dict] = []
    for arch in arch_entities:
        evidence = _gather_broad_evidence(
            arch.name, component_names, filtered_text,
        )

        kg_prior = None
        if kb_client is not None:
            kg_prior = _fetch_kg_prior(arch.name, kb_client)

        existing_desc = descriptions.get(arch.name, {})

        prompt_items.append({
            "name": arch.name,
            "existing_description": existing_desc.get("description", ""),
            "existing_components": existing_desc.get("components", []),
            "existing_connectivity": existing_desc.get("connectivity", []),
            "candidate_components": component_names,
            "evidence": evidence,
            "kg_prior": kg_prior,
        })

    sys_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(prompt_items)

    templates: Dict[str, ArchitectureTemplate] = {}

    try:
        result = llm_api.callgoogle_pydantic(user_prompt, sys_prompt, _ArchTemplateList)
        items = result.architectures if result and hasattr(result, "architectures") else []
    except Exception as e:
        print(f"Warning: Architecture decomposition LLM call failed: {e}")
        items = []

    for item in items:
        validated = _validate_template(item, component_names, kb_client)
        if validated is not None:
            templates[item.name] = validated

    return templates


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert photonic integrated circuit (PIC) architect.

Your task: decompose each given architecture entity into its **structural blueprint**.

For each architecture you MUST produce:

1. **component_roles** — a list of functional slots.  Each slot has:
   - `role_id`: a short, unique identifier (snake_case) describing the *function*
     of the component in this architecture (e.g. "input_splitter", "phase_arm",
     "drop_ring_1").
   - `component_type`: the type of component filling this role (should match one
     of the `candidate_components` when possible, but you may name a generic type
     if the evidence clearly supports it).
   - `port_config`: port configuration hint if known (e.g. "2x2", "1x2").
   - `quantity`: number of instances of this role (default 1).
   - `alternatives`: list of alternative component types that could fill this role.

2. **connections** — directed port-level links between roles:
   - `from_role` / `from_port` → `to_role` / `to_port`
   - Use descriptive port names (e.g. "out1", "cross", "through", "in", "add", "drop").
   - Every role must appear in at least one connection.

3. **topology_class** — one of: tree, cascade, mesh, ring-bus, star, parallel, series, point-to-point, or a short descriptive term.

4. **scaling_rules** (optional) — a dict of parameter-to-formula pairs if the
   architecture scales (e.g. {"stages": "log2(N)", "splitters": "N-1"}).

5. **variants** (optional) — known named variants of this architecture.

Rules:
- Ground your answer ONLY in the provided evidence and KG priors.
- If you lack evidence for connections, still provide your best structural guess
  and set plausible port names; the confidence field is available for the caller
  to track certainty.
- Prefer specific component types from the candidate list over generic terms.
- The `name` field in your output MUST exactly match the input architecture name.

Output must be valid JSON matching the requested schema."""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT


def _build_user_prompt(items: list[dict]) -> str:
    return f"""\
Decompose each architecture below into a structural blueprint.

Input architectures:
{json.dumps(items, indent=2, default=str)}

Return JSON:
{{
  "architectures": [
    {{
      "name": "...",
      "topology_class": "...",
      "component_roles": [
        {{"role_id": "...", "component_type": "...", "port_config": "...", "quantity": 1, "alternatives": []}}
      ],
      "connections": [
        {{"from_role": "...", "from_port": "...", "to_role": "...", "to_port": "..."}}
      ],
      "scaling_rules": {{}},
      "variants": []
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_template(
    item: _ArchTemplateItem,
    paper_components: list[str],
    kb_client=None,
) -> Optional[ArchitectureTemplate]:
    """Convert an LLM output item into a validated ArchitectureTemplate.

    Returns None only if the template is completely empty (no roles at all).
    Otherwise, it flags unrecognised component types by lowering confidence.
    """
    if not item.component_roles:
        return None

    paper_lower = {c.lower() for c in paper_components}

    unrecognised: list[str] = []
    for role in item.component_roles:
        ct_lower = role.component_type.lower()
        if ct_lower not in paper_lower:
            found_in_kg = False
            if kb_client is not None:
                try:
                    found_in_kg = kb_client.find_by_name(role.component_type, "components") is not None
                except Exception:
                    pass
            if not found_in_kg:
                unrecognised.append(role.component_type)

    role_ids = {r.role_id for r in item.component_roles}
    connected_roles = set()
    for conn in item.connections:
        connected_roles.add(conn.from_role)
        connected_roles.add(conn.to_role)
    orphaned = role_ids - connected_roles

    confidence = 1.0
    if unrecognised:
        confidence -= 0.15 * min(len(unrecognised), 3)
    if orphaned:
        confidence -= 0.1 * min(len(orphaned), 3)
    confidence = max(confidence, 0.1)

    return ArchitectureTemplate(
        name=item.name,
        topology_class=item.topology_class,
        component_roles=item.component_roles,
        connections=item.connections,
        scaling_rules=item.scaling_rules,
        variants=item.variants,
        source="paper:decomposer",
        confidence=round(confidence, 2),
    )
