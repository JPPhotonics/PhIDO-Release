"""
Pipeline Orchestrator — deterministic controller for the full PhIDO pipeline.

Chains: Interpreter -> Component Selection -> Schematic Building
with validation gates and backward feedback loops.

Not an MCP server — called directly by the Streamlit UI or scripts.

Stages:
  Phases 0-3   : Interpreter (explore_and_ask + finalize_stream)
  Phase 4      : Component Selection (LLM matches ComponentIntents to PDK modules)
  Phase 4.75   : Selection Validation Gate (programmatic checks)
  Phase 4.8    : LLM Compliance Check (verify functional match of selections)
  Phase 5      : Circuit DSL Construction + DOT generation (deterministic)
  Phase 5b     : Edge Routing (LLM adds port-level edges to DOT, with planarity retries)
  Phase 6      : Layout computation (Graphviz) + footprints (GDSFactory)
  Phase 6.5    : Schematic Validation Gate (programmatic checks)
  Phase 7      : Open ports + GDSFactory netlist export
"""

import json
import re
from typing import Any, Generator, Optional

_CODE_FENCE_RE = re.compile(r"```(?:dot|graphviz)?\s*\n?", re.IGNORECASE)

from openai import OpenAI

from mcp_servers.models import (
    DesignIntent,
    ComponentMapping,
    ComponentSelection,
    ComponentSelectionLLM,
    ComplianceVerdict,
    PipelineFeedback,
    FeedbackClassification,
    EdgePatch,
    ComponentSwap,
)
from mcp_servers.interpreter_agent import explore_and_ask, finalize_stream
from mcp_servers.pdk_catalog_server import (
    search_components,
    get_component_details,
    get_component_footprint,
    get_module_params,
    get_port_names,
    validate_selection,
)
from mcp_servers.schematic_builder_server import (
    circuit_dsl_to_dot,
    check_planarity,
    compute_layout,
    find_open_ports,
    export_gf_netlist,
)

PipelineEvent = dict[str, Any]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SELECTOR_PROMPT = """\
You are a photonic component selector. Given a DesignIntent with abstract component
descriptions, match each component to a concrete PDK module.

MATCHING PRIORITY (in strict order):
1. FUNCTIONAL MATCH IS PARAMOUNT. The component's `description` and `role` define what
   the device IS. A PDK module must be the same *type* of device. For example:
   - "microring modulator" must map to a ring-based module, NEVER to an MZI module.
   - "directional coupler" must map to a coupler module, not a splitter or MMI.
   - "MZI switch" must map to an MZI-based module, not a ring resonator.
   Read the PDK module's name, description, and labels carefully to verify the device type.
2. PORT CONFIGURATION is secondary. A module with the correct function but slightly
   different port count is a better match than one with the right port count but
   completely wrong function. Mark port mismatches in notes.
3. SPECS (bandwidth, arm length, etc.) are tertiary — note any spec mismatches.

For EACH component in the DesignIntent:
1. Read the component's description, role, and specs carefully.
2. From the PDK search results, identify modules whose *device type* matches the
   component's description and role. Ignore modules whose function is fundamentally
   different, even if they score high on keyword overlap.
3. Among functionally matching modules, pick the one with the best port config and
   spec fit.
4. Set match_quality:
   - "exact": function, port config, and key specs all match
   - "partial": function matches but port config or some specs differ
   - "poor": best available but significant mismatch — explain in notes
5. Map any user-specified specs to GDSFactory parameter names where possible
   and place them in user_overrides. Leave user_overrides empty if no specs
   can be mapped.
6. Leave resolved_settings EMPTY (as an empty list). Default settings will be
   fetched from GDSFactory automatically — do NOT guess or fabricate them.

If NO module in the PDK is a functional match for a component, add its ID to the
unmapped list. Do NOT force a functionally wrong module just to avoid unmapped entries.

Return a JSON object matching the ComponentSelectionLLM schema exactly.
"""

EDGE_ROUTING_PROMPT = """\
You are an assistant to a photonic engineer.
You have two input DOT graphs:
- Graph1 has the correct definition of nodes including their ports, but the edges are missing.
- Graph2 has the correct definition of edges, but the nodes definition is incomplete.

Follow these instructions:
- Add the edges from Graph2 to Graph1.
- Add the port numbers to the edge definitions (e.g. C1:o3 -- C2:o2;). Do not label the edges.
- Do not change the node definitions (the labels and the ports) in Graph1.
- Ports are labelled o1, o2, o3 etc, ordered counter-clockwise around the rectangle node.
  For example a 2x3 node has o1 (left-bottom), o2 (left-top), o3 (right-top), o4 (right-middle), o5 (right-bottom).
- It is important that the edges don't cross. Reason about the spatial location of ports and ensure non-crossing edges.
- Each port can only take one edge.
- Define only one edge between any two nodes unless explicitly stated.
- Do not connect a node to itself unless explicitly stated.
- If there is only one node with no connections, output Graph1 unchanged.
- Do not add additional nodes.

Output ONLY the dot code. No explanation, no ```dot fences.
"""

EDGE_ROUTING_RETRY_PROMPT = """\
You are an assistant to a photonic engineer.
You have three input DOT graphs:
- Graph1 has the correct definition of nodes including their ports, but the edges are missing.
- Graph2 has the correct definition of edges, but the nodes definition is incomplete.
- Graph3 has the correct definition of nodes but a definition of edges that FAILED a test \
for crossings.

Follow these instructions:
- Add the edges from Graph2 to Graph1.
- Add the port numbers to the edge definitions (e.g. C1:o3 -- C2:o2;). Do not label the edges.
- Do not change the node definitions (the labels and the ports) in Graph1.
- Ports are labelled o1, o2, o3 etc, ordered counter-clockwise around the rectangle node.
  For example a 2x3 node has o1 (left-bottom), o2 (left-top), o3 (right-top), o4 (right-middle), \
o5 (right-bottom).
- It is important that the edges don't cross. You should reason about the spatial location of \
ports around the rectangular nodes and make sure the edges are not crossing. \
Refer to Graph3, which is a failed attempt.
- Each port can only take one edge.
- Define only one edge between any two nodes unless explicitly stated.
- Do not connect a node to itself unless explicitly stated.
- If there is only one node with no connections output Graph1.
- Do not under any circumstances add new nodes.

Output ONLY the dot code. No explanation, no ```dot fences.
"""

EDGE_VERIFY_PROMPT = """\
Input is a DOT graph.
Each node has some ports labelled as o1, o2 etc.
Check this condition:
The graph should have only one edge between any two nodes, counting all of their ports. \
Edges are defined using the syntax (C1:o3 -- C2:o2).
If there is only one node there should most often be no edges.

If necessary, modify the input DOT graph to satisfy these conditions. \
Do not add additional nodes under any circumstances.
Do not output any text beyond the DOT code to generate a single valid DOT graph. \
Output should not contain any preamble like ```dot.
"""

COMPLIANCE_PROMPT = """\
You are a photonic design reviewer. For each component mapping below, judge whether
the selected PDK module is a faithful FUNCTIONAL match for the original design intent.

Focus on whether the PDK module is the correct *type* of device:
- A microring modulator and an MZI modulator are fundamentally different devices
  (resonance vs. interference). Selecting one for the other is NON-COMPLIANT.
- A directional coupler and an MMI coupler may be acceptable substitutes depending
  on spec requirements — note this in your reason.
- An MMI splitter selected for an MMI coupler role is likely compliant if port
  configs align.

For each mapping, return:
- compliant: true if the PDK module's device type matches the described component
- reason: brief explanation of why it is or isn't compliant
- suggested_alternative: if non-compliant, suggest what type of PDK module to search
  for instead (e.g. "Search for ring_modulator or microring modules")

Return a JSON object matching the ComplianceVerdict schema exactly.
"""

FEEDBACK_CLASSIFIER_PROMPT = """\
You are a schematic feedback classifier for a photonic circuit design tool.

The user has reviewed a generated schematic and provided feedback. Your job is to
classify the feedback into one of three tiers so the system can re-run only the
minimum necessary pipeline stages.

TIERS:
1. "edge_edit" — The user wants to add, remove, or change specific port-to-port
   connections WITHOUT changing which components exist or which PDK modules are used.
   Examples: "connect o3 of C1 to o2 of C2", "remove the link between C1 and C3",
   "swap the connection from C1:o1-C2:o2 to C1:o1-C3:o1".
   For this tier you MUST populate edge_patches with structured EdgePatch objects.

2. "component_swap" — The user wants to replace one or more components with
   different PDK modules (or ask for a different type of device) WITHOUT changing
   the overall circuit topology or adding/removing components.
   Examples: "use a ring modulator instead of MZI for C2", "change C3 to a 1x4 MMI".
   For this tier you MUST populate component_swaps with ComponentSwap objects.

3. "architectural" — The user wants to add new components, remove existing
   components, change the overall topology, or make changes that require
   re-interpreting the design intent from scratch.
   Examples: "add a photodetector at the output", "split the MZI into separate arms",
   "redesign with 4 channels instead of 2".

CURRENT SCHEMATIC CONTEXT:
{context}

Classify the feedback and extract structured actions where applicable.
Return a JSON object matching the FeedbackClassification schema.
"""

UPSTREAM_FEEDBACK_PROMPT = """\
DOWNSTREAM VALIDATION FEEDBACK
===============================
A later stage in the pipeline attempted to build your DesignIntent into a
fabricable circuit and encountered problems. These are NOT hypothetical — they
are concrete failures from PDK validation and schematic construction.

{feedback_items}

You MUST address these problems using your tools. They indicate that your
DesignIntent needs revision — either different components, different decomposition,
or different connections. Re-investigate with search_pdk and the knowledge graph,
then say DONE when you have enough information to produce a corrected DesignIntent.
"""


# ---------------------------------------------------------------------------
# Feedback Classification (tiered schematic edits)
# ---------------------------------------------------------------------------

def _build_schematic_context(
    circuit_dsl: dict,
    selection_dict: dict,
    dot_string: str,
) -> str:
    """Build a concise text summary of the current schematic for the classifier."""
    parts: list[str] = []

    nodes = circuit_dsl.get("nodes", {})
    parts.append("NODES:")
    for nid, ndata in nodes.items():
        comp = ndata.get("component", "?")
        ports_cfg = ndata.get("properties", {}).get("ports", "?")
        port_names = get_port_names(comp)
        parts.append(f"  {nid}: {comp} (ports={ports_cfg}, port_names={port_names})")

    edges = circuit_dsl.get("edges", {})
    if edges:
        parts.append("\nEDGES:")
        for eid, edata in edges.items():
            parts.append(f"  {eid}: {edata.get('link', '?')}")
    else:
        edge_pattern = re.compile(r"(\w+):(\w+)\s*--\s*(\w+):(\w+)")
        matches = edge_pattern.findall(dot_string)
        if matches:
            parts.append("\nEDGES (from DOT):")
            for i, (sn, sp, tn, tp) in enumerate(matches):
                parts.append(f"  E{i+1}: {sn},{sp} -> {tn},{tp}")

    mappings = selection_dict.get("mappings", [])
    if mappings:
        parts.append("\nCOMPONENT MAPPINGS:")
        for m in mappings:
            parts.append(
                f"  {m['component_id']}: {m['pdk_module']} "
                f"(match={m.get('match_quality','?')}, ports={m.get('port_config','?')})"
            )

    return "\n".join(parts)


def _classify_feedback(
    client: OpenAI,
    feedback_text: str,
    circuit_dsl: dict,
    selection_dict: dict,
    dot_string: str,
    model: str = "o3-mini",
) -> FeedbackClassification:
    """Use an LLM to classify user schematic feedback into a tier."""
    context = _build_schematic_context(circuit_dsl, selection_dict, dot_string)
    prompt = FEEDBACK_CLASSIFIER_PROMPT.replace("{context}", context)

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": feedback_text},
        ],
        response_format=FeedbackClassification,
    )
    return response.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Phase 4: Component Selection
# ---------------------------------------------------------------------------

def _run_component_selection(
    client: OpenAI,
    design_intent: DesignIntent,
    model: str = "o3-mini",
) -> Generator[PipelineEvent, None, None]:
    """Match ComponentIntents to PDK modules via LLM structured output.

    Yields streaming events. The final event has type ``"selection_done"``
    and carries the ``ComponentSelection`` in ``event["_result"]``.
    """
    yield {"type": "phase", "phase": "component_selection",
           "detail": "Matching components to PDK modules..."}

    pdk_context_parts = []
    for comp in design_intent.components:
        query = f"{comp.role or ''} {comp.description} {comp.port_config or ''}".strip()
        results = search_components(query)
        pdk_context_parts.append(
            f"## {comp.id}: {comp.description}\n"
            f"PDK search results:\n{results}\n"
        )
        yield {"type": "tool_call", "name": "search_pdk", "args": {"query": query}}
        yield {"type": "tool_result", "name": "search_pdk", "result": results}

    pdk_context = "\n".join(pdk_context_parts)

    messages = [
        {"role": "system", "content": SELECTOR_PROMPT},
        {"role": "user", "content": (
            f"## DesignIntent\n\n"
            f"```json\n{json.dumps(design_intent.summary(), indent=2)}\n```\n\n"
            f"## PDK Search Results\n{pdk_context}"
        )},
    ]

    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=ComponentSelectionLLM,
    )

    llm_result: ComponentSelectionLLM = response.choices[0].message.parsed

    selection = ComponentSelection(
        design_intent=design_intent,
        mappings=llm_result.mappings,
        unmapped=llm_result.unmapped,
        ambiguities=llm_result.ambiguities,
    )

    yield {"type": "selection_done",
           "selection": selection.model_dump(),
           "_result": selection}


# ---------------------------------------------------------------------------
# Phase 4.5: LLM Compliance Check
# ---------------------------------------------------------------------------

def _llm_compliance_check(
    client: OpenAI,
    selection: ComponentSelection,
    design_intent: DesignIntent,
    model: str = "o3-mini",
) -> list[PipelineFeedback]:
    """Ask an LLM to verify each mapping is a faithful functional match.

    Returns PipelineFeedback items for any non-compliant mappings.
    """
    review_parts = []
    for mapping in selection.mappings:
        intent_comp = next(
            (c for c in design_intent.components if c.id == mapping.component_id),
            None,
        )
        if intent_comp is None:
            continue

        pdk_meta = get_component_details(mapping.pdk_module)

        review_parts.append(
            f"### {mapping.component_id}\n"
            f"**Design Intent:** description=\"{intent_comp.description}\", "
            f"role=\"{intent_comp.role or 'unspecified'}\", "
            f"port_config=\"{intent_comp.port_config or 'unspecified'}\"\n"
            f"**Selected PDK Module:** {mapping.pdk_module} "
            f"(match_quality={mapping.match_quality})\n"
            f"**PDK Catalog Entry:**\n{pdk_meta}\n"
        )

    if not review_parts:
        return []

    messages = [
        {"role": "system", "content": COMPLIANCE_PROMPT},
        {"role": "user", "content": "\n".join(review_parts)},
    ]

    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=ComplianceVerdict,
        )
        verdict: ComplianceVerdict = response.choices[0].message.parsed
    except Exception:
        return []

    feedback: list[PipelineFeedback] = []
    for result in verdict.results:
        if not result.compliant:
            feedback.append(PipelineFeedback(
                source_phase="compliance_check",
                target_phase="component_selector",
                severity="major",
                description=(
                    f"Compliance check FAILED for {result.component_id}: "
                    f"{result.reason}"
                ),
                affected_components=[result.component_id],
                suggested_action=result.suggested_alternative or "Re-search PDK for correct device type",
                context={"compliance_result": result.model_dump()},
            ))

    return feedback


# ---------------------------------------------------------------------------
# Phase 5b: LLM Edge Routing
# ---------------------------------------------------------------------------

def _sanitise_dot(raw: str) -> str:
    """Strip markdown code fences, inline comments, and leading/trailing whitespace."""
    cleaned = _CODE_FENCE_RE.sub("", raw).strip()
    cleaned = re.sub(r"//.*", "", cleaned)
    if not re.match(r"(?:di)?graph\s", cleaned, re.IGNORECASE):
        match = re.search(r"((?:di)?graph\s[\s\S]+\})", cleaned)
        if match:
            cleaned = match.group(1)
    return cleaned


def _run_edge_routing(
    client: OpenAI,
    dot_with_ports: str,
    preschematic_dot: str,
    model: str = "o3-mini",
    max_retries: int = 2,
) -> Generator[PipelineEvent, None, None]:
    """Add port-level edges to a DOT graph using LLM, with planarity retries.

    1. Initial LLM edge routing
    2. Planarity check — if crossings found, retry with the failed graph as feedback
    3. Verification pass — ensure one edge per node pair, no extra nodes

    Final event has type ``"edge_routing_done"`` with ``event["_result"]``
    containing the DOT string.
    """
    yield {"type": "phase", "phase": "edge_routing",
           "detail": "LLM routing port-level edges..."}

    # --- Step 1: Initial edge routing ---
    prompt = (
        f"{EDGE_ROUTING_PROMPT}\n\n"
        f"INPUT graph1:\n{dot_with_ports}\n\n"
        f"INPUT graph2:\n{preschematic_dot}\n"
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    dot_with_edges = _sanitise_dot(response.choices[0].message.content)

    # --- Step 2: Planarity check + retry loop ---
    for attempt in range(max_retries):
        try:
            planarity = json.loads(check_planarity(dot_with_edges))
        except (json.JSONDecodeError, TypeError):
            break  # can't parse result, skip retry

        if planarity.get("planar", True):
            yield {"type": "phase", "phase": "edge_routing",
                   "detail": f"Planarity OK (attempt {attempt + 1})"}
            break

        n_crossings = planarity.get("num_crossings", "?")
        yield {"type": "phase", "phase": "edge_routing",
               "detail": f"Planarity FAILED — {n_crossings} crossing(s), retrying (attempt {attempt + 2})..."}

        retry_prompt = (
            f"{EDGE_ROUTING_RETRY_PROMPT}\n\n"
            f"INPUT graph1:\n{dot_with_ports}\n\n"
            f"INPUT graph2:\n{preschematic_dot}\n\n"
            f"INPUT graph3 (FAILED attempt):\n{dot_with_edges}\n"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": retry_prompt}],
        )
        dot_with_edges = _sanitise_dot(response.choices[0].message.content)

    # --- Step 3: Verification pass ---
    yield {"type": "phase", "phase": "edge_routing",
           "detail": "Verifying edge constraints..."}

    verify_prompt = f"{EDGE_VERIFY_PROMPT}\n\n{dot_with_edges}"
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": verify_prompt}],
    )
    dot_verified = _sanitise_dot(response.choices[0].message.content)

    yield {"type": "edge_routing_done", "dot": dot_verified, "_result": dot_verified}


# ---------------------------------------------------------------------------
# Validation Gates (programmatic, no LLM)
# ---------------------------------------------------------------------------

def _validate_selection(
    selection: ComponentSelection,
    design_intent: DesignIntent,
) -> list[PipelineFeedback]:
    """Programmatic validation after Phase 4."""
    feedback: list[PipelineFeedback] = []

    mapped_ids = {m.component_id for m in selection.mappings}

    for comp in design_intent.components:
        if comp.id not in mapped_ids and comp.id not in selection.unmapped:
            feedback.append(PipelineFeedback(
                source_phase="component_selection",
                target_phase="interpreter",
                severity="fundamental",
                description=(
                    f"Component {comp.id} ({comp.description}) has no mapping "
                    f"and is not listed as unmapped"
                ),
                affected_components=[comp.id],
                suggested_action="Decompose into sub-components or revise description",
                context={"component": comp.model_dump()},
            ))

    for uid in selection.unmapped:
        feedback.append(PipelineFeedback(
            source_phase="component_selection",
            target_phase="interpreter",
            severity="fundamental",
            description=f"No PDK module found for {uid}",
            affected_components=[uid],
            suggested_action="Decompose into sub-components the PDK supports",
            context={},
        ))

    for mapping in selection.mappings:
        intent_comp = next(
            (c for c in design_intent.components if c.id == mapping.component_id),
            None,
        )
        if (intent_comp and intent_comp.port_config
                and intent_comp.port_config != mapping.port_config):
            feedback.append(PipelineFeedback(
                source_phase="component_selection",
                target_phase="interpreter",
                severity="major",
                description=(
                    f"Interpreter claims {mapping.component_id} is "
                    f"{intent_comp.port_config} but best PDK match has "
                    f"{mapping.port_config}"
                ),
                affected_components=[mapping.component_id],
                suggested_action="Verify port config with PDK tools or revise",
                context={"pdk_module": mapping.pdk_module,
                         "pdk_ports": mapping.port_config},
            ))

    for mapping in selection.mappings:
        if mapping.match_quality == "poor":
            feedback.append(PipelineFeedback(
                source_phase="component_selection",
                target_phase="interpreter",
                severity="major",
                description=(
                    f"Poor PDK match for {mapping.component_id}: "
                    f"'{mapping.pdk_module}'"
                ),
                affected_components=[mapping.component_id],
                suggested_action="Consider alternative components or decomposition",
                context={"notes": mapping.notes},
            ))

    batch = json.dumps([
        {"component_id": m.component_id, "pdk_module": m.pdk_module,
         "expected_ports": m.port_config}
        for m in selection.mappings
    ])
    gds_result = json.loads(validate_selection(batch))
    for issue in gds_result.get("issues", []):
        feedback.append(PipelineFeedback(
            source_phase="component_selection",
            target_phase=(
                "interpreter" if issue["severity"] == "fundamental"
                else "component_selector"
            ),
            severity=issue["severity"],
            description=issue["issue"],
            affected_components=[issue["component_id"]],
            suggested_action="Check PDK module name and port config",
            context=issue,
        ))

    return feedback


def _validate_schematic(
    dot_with_edges: str,
    design_intent: DesignIntent,
) -> list[PipelineFeedback]:
    """Programmatic validation after Phases 5-6."""
    feedback: list[PipelineFeedback] = []

    try:
        planarity = json.loads(check_planarity(dot_with_edges))
        if not planarity.get("planar", True):
            feedback.append(PipelineFeedback(
                source_phase="schematic_builder",
                target_phase="local",
                severity="minor",
                description=(
                    f"DOT graph has {planarity['num_crossings']} edge crossing(s)"
                ),
                affected_components=["global"],
                suggested_action="Re-run edge routing with crossing feedback",
                context=planarity,
            ))
    except Exception:
        pass

    try:
        ports_info = json.loads(find_open_ports(dot_with_edges))
        if ports_info["connected_count"] == 0 and len(design_intent.connections) > 0:
            feedback.append(PipelineFeedback(
                source_phase="schematic_builder",
                target_phase="schematic_builder",
                severity="major",
                description="No ports are connected despite DesignIntent having connections",
                affected_components=["global"],
                suggested_action="Edge routing likely failed — re-run",
                context=ports_info,
            ))
    except Exception:
        pass

    return feedback


# ---------------------------------------------------------------------------
# Helpers (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _get_ground_truth_params(pdk_module: str) -> dict[str, str]:
    """Fetch real GDSFactory default settings for a PDK module.

    Returns a dict of param_name -> value (stringified). Falls back to
    an empty dict if GDSFactory instantiation fails.
    """
    try:
        result = json.loads(get_module_params(pdk_module))
        if "error" in result:
            return {}
        settings = result.get("settings", {})
        return {k: str(v) for k, v in settings.items()}
    except Exception:
        return {}


def _build_circuit_dsl(
    design_intent: DesignIntent,
    selection: ComponentSelection,
) -> dict:
    """Assemble a circuit DSL dict from DesignIntent + ComponentSelection.

    For each mapped component, fetches real GDSFactory defaults via
    ``get_module_params`` and overlays any ``user_overrides`` from the
    component selector on top.
    """
    mappings_by_id = {m.component_id: m for m in selection.mappings}
    nodes = {}
    for comp in design_intent.components:
        mapping = mappings_by_id.get(comp.id)
        if mapping:
            params = _get_ground_truth_params(mapping.pdk_module)
            for override in mapping.user_overrides:
                if override.key in params:
                    params[override.key] = override.value
            display = (comp.role.title() if comp.role else comp.description)
            nodes[comp.id] = {
                "component": mapping.pdk_module,
                "properties": {"ports": mapping.port_config},
                "params": params,
                "label": f"{comp.id}: {display}\\n({mapping.pdk_module})",
            }
    return {
        "doc": {
            "name": design_intent.title,
            "description": design_intent.brief_summary,
        },
        "nodes": nodes,
        "edges": {},
        "ports": {},
    }


def _enrich_circuit_dsl(
    circuit_dsl: dict,
    selection: ComponentSelection,
    positions: dict,
    footprints: dict,
    circuit_ports: dict,
) -> None:
    """Mutate circuit_dsl in-place: add placements, footprints, and ports."""
    for node_id, node in circuit_dsl["nodes"].items():
        pos = positions.get(node_id, {})
        fp = footprints.get(node_id, [0, 0])
        dx = fp[0] if isinstance(fp, (list, tuple)) else 0
        dy = fp[1] if isinstance(fp, (list, tuple)) else 0
        node["placement"] = {
            "x": pos.get("x", 0) - dx / 2,
            "y": pos.get("y", 0),
            "rotation": 0,
        }
        node["properties"]["dx"] = dx
        node["properties"]["dy"] = dy

    circuit_dsl["ports"] = circuit_ports


def _edges_dot_to_dsl(dot_string: str, circuit_dsl: dict) -> dict:
    """Parse edges from a DOT string and add them to the circuit DSL."""
    edge_pattern = re.compile(r"(\w+):(\w+)\s*--\s*(\w+):(\w+)\s*;")
    matches = edge_pattern.findall(dot_string)
    edges = {}
    for i, (src_node, src_port, tgt_node, tgt_port) in enumerate(matches):
        edges[f"E{i + 1}"] = {"link": f"{src_node},{src_port}: {tgt_node},{tgt_port}"}
    circuit_dsl["edges"] = edges
    return circuit_dsl


# ---------------------------------------------------------------------------
# Tier 1: Edge-only patching
# ---------------------------------------------------------------------------

def _apply_edge_patches(dot_string: str, patches: list[EdgePatch]) -> str:
    """Apply structured edge patches to a DOT string and return the modified DOT."""
    lines = dot_string.strip().splitlines()
    result_lines: list[str] = []
    closing_brace_idx = None

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "}":
            closing_brace_idx = i
        result_lines.append(line)

    if closing_brace_idx is None:
        closing_brace_idx = len(result_lines)

    for patch in patches:
        if patch.action == "remove" or patch.action == "replace":
            old_sn = patch.old_src_node if patch.action == "replace" else patch.src_node
            old_sp = patch.old_src_port if patch.action == "replace" else patch.src_port
            old_tn = patch.old_tgt_node if patch.action == "replace" else patch.tgt_node
            old_tp = patch.old_tgt_port if patch.action == "replace" else patch.tgt_port
            if old_sn and old_sp and old_tn and old_tp:
                pat = re.compile(
                    rf"\b{re.escape(old_sn)}:{re.escape(old_sp)}\s*--\s*"
                    rf"{re.escape(old_tn)}:{re.escape(old_tp)}\b"
                )
                pat_rev = re.compile(
                    rf"\b{re.escape(old_tn)}:{re.escape(old_tp)}\s*--\s*"
                    rf"{re.escape(old_sn)}:{re.escape(old_sp)}\b"
                )
                result_lines = [
                    l for l in result_lines
                    if not (pat.search(l) or pat_rev.search(l))
                ]

        if patch.action in ("add", "replace"):
            if patch.src_node and patch.src_port and patch.tgt_node and patch.tgt_port:
                new_edge = f"  {patch.src_node}:{patch.src_port} -- {patch.tgt_node}:{patch.tgt_port};"
                closing = None
                for idx in range(len(result_lines) - 1, -1, -1):
                    if result_lines[idx].strip() == "}":
                        closing = idx
                        break
                if closing is not None:
                    result_lines.insert(closing, new_edge)
                else:
                    result_lines.append(new_edge)

    return "\n".join(result_lines)


def run_pipeline_patch_edges(
    circuit_dsl: dict,
    selection_dict: dict,
    design_intent_dict: dict,
    dot_string: str,
    footprints: dict,
    edge_patches: list[EdgePatch],
    model: str = "o3-mini",
) -> Generator[PipelineEvent, None, None]:
    """Tier 1: apply edge patches to the existing schematic without re-running
    the interpreter or component selection.

    Re-runs only: edge patching -> planarity check -> layout -> validation -> export.
    """
    yield {"type": "pipeline_phase", "phase": "edge_patching"}
    yield {"type": "phase", "phase": "edge_patching",
           "detail": f"Applying {len(edge_patches)} edge patch(es)..."}

    patched_dot = _apply_edge_patches(dot_string, edge_patches)

    try:
        planarity = json.loads(check_planarity(patched_dot))
        if planarity.get("planar", True):
            yield {"type": "phase", "phase": "edge_patching",
                   "detail": "Planarity OK after patching."}
        else:
            n = planarity.get("num_crossings", "?")
            yield {"type": "phase", "phase": "edge_patching",
                   "detail": f"Warning: {n} crossing(s) after patching."}
    except Exception:
        pass

    yield {"type": "edge_routing_done", "dot": patched_dot, "_result": patched_dot}

    # Re-run layout
    yield {"type": "pipeline_phase", "phase": "layout"}
    layout_result_raw = compute_layout(patched_dot, json.dumps(footprints))
    try:
        layout_result = json.loads(layout_result_raw)
    except (json.JSONDecodeError, TypeError):
        layout_result = {"positions": {}, "error": "Layout computation returned invalid JSON"}

    positions = layout_result.get("positions", {})
    yield {"type": "layout_done", "positions": positions}

    # Validate
    design_intent = DesignIntent(**design_intent_dict)
    sch_feedback = _validate_schematic(patched_dot, design_intent)
    if sch_feedback:
        yield {"type": "feedback", "source": "schematic_builder",
               "issues": [f.model_dump() for f in sch_feedback]}

    # Export
    yield {"type": "pipeline_phase", "phase": "export"}
    try:
        ports_result = json.loads(find_open_ports(patched_dot))
    except (json.JSONDecodeError, TypeError):
        ports_result = {"open_ports": [], "circuit_ports": {}}

    patched_dsl = json.loads(json.dumps(circuit_dsl))
    _edges_dot_to_dsl(patched_dot, patched_dsl)

    selection = ComponentSelection(**selection_dict)
    _enrich_circuit_dsl(
        patched_dsl, selection, positions,
        footprints, ports_result.get("circuit_ports", {}),
    )

    gf_netlist_yaml = export_gf_netlist(json.dumps(patched_dsl))

    yield {"type": "pipeline_done", "result": {
        "design_intent": design_intent_dict,
        "selection": selection_dict,
        "circuit_dsl": patched_dsl,
        "dot_string": patched_dot,
        "gf_netlist_yaml": gf_netlist_yaml,
        "open_ports": ports_result,
        "footprints": footprints,
        "positions": positions,
    }}


# ---------------------------------------------------------------------------
# Tier 2: Component re-selection (targeted)
# ---------------------------------------------------------------------------

def run_pipeline_reselect(
    design_intent_dict: dict,
    prev_selection_dict: dict,
    component_swaps: list[ComponentSwap],
    model: str = "o3-mini",
) -> Generator[PipelineEvent, None, None]:
    """Tier 2: re-select only the targeted components, keeping all others
    from the previous selection, then rebuild DSL + edge routing + layout + export.
    """
    client = OpenAI()
    design_intent = DesignIntent(**design_intent_dict)

    yield {"type": "pipeline_phase", "phase": "component_reselection"}
    swap_ids = {s.component_id for s in component_swaps}
    yield {"type": "phase", "phase": "component_reselection",
           "detail": f"Re-selecting {len(swap_ids)} component(s): {', '.join(swap_ids)}"}

    prev_mappings = prev_selection_dict.get("mappings", [])
    kept = [ComponentMapping(**m) for m in prev_mappings if m["component_id"] not in swap_ids]

    new_mappings: list[ComponentMapping] = []
    for swap in component_swaps:
        comp = next(
            (c for c in design_intent.components if c.id == swap.component_id), None,
        )
        if comp is None:
            yield {"type": "phase", "phase": "component_reselection",
                   "detail": f"Warning: {swap.component_id} not in design intent, skipping."}
            continue

        results = search_components(swap.new_module_hint)
        yield {"type": "tool_call", "name": "search_pdk",
               "args": {"query": swap.new_module_hint}}
        yield {"type": "tool_result", "name": "search_pdk", "result": results}

        pdk_context = (
            f"## {comp.id}: {comp.description}\n"
            f"User requested swap to: {swap.new_module_hint}\n"
            f"PDK search results:\n{results}\n"
        )

        messages = [
            {"role": "system", "content": SELECTOR_PROMPT},
            {"role": "user", "content": (
                f"## DesignIntent (single component)\n\n"
                f"```json\n{json.dumps({'components': [comp.model_dump()]}, indent=2)}\n```\n\n"
                f"## PDK Search Results\n{pdk_context}"
            )},
        ]

        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=ComponentSelectionLLM,
            )
            llm_result: ComponentSelectionLLM = response.choices[0].message.parsed
            new_mappings.extend(llm_result.mappings)
        except Exception as exc:
            yield {"type": "phase", "phase": "component_reselection",
                   "detail": f"Selection failed for {swap.component_id}: {exc}"}

    all_mappings = kept + new_mappings
    selection = ComponentSelection(
        design_intent=design_intent,
        mappings=all_mappings,
        unmapped=[],
        ambiguities=[],
    )

    yield {"type": "selection_done",
           "selection": selection.model_dump(),
           "_result": selection}

    # -- compliance check on new mappings only --
    if new_mappings:
        yield {"type": "pipeline_phase", "phase": "compliance_check"}
        yield {"type": "phase", "phase": "compliance_check",
               "detail": "Checking compliance of re-selected components..."}
        partial_sel = ComponentSelection(
            design_intent=design_intent,
            mappings=new_mappings,
            unmapped=[],
            ambiguities=[],
        )
        compliance_feedback = _llm_compliance_check(client, partial_sel, design_intent, model)
        if compliance_feedback:
            yield {"type": "feedback", "source": "compliance_check",
                   "issues": [f.model_dump() for f in compliance_feedback]}
        else:
            yield {"type": "phase", "phase": "compliance_check",
                   "detail": "All re-selected mappings compliant."}

    # -- Phases 5-7: full rebuild from the merged selection --
    yield {"type": "pipeline_phase", "phase": "schematic_building"}

    circuit_dsl = _build_circuit_dsl(design_intent, selection)
    dot_no_edges = circuit_dsl_to_dot(json.dumps(circuit_dsl))

    if dot_no_edges.startswith("{"):
        parsed = json.loads(dot_no_edges)
        if "error" in parsed:
            yield {"type": "error", "message": f"DOT generation failed: {parsed['error']}"}
            return

    yield {"type": "dot_draft", "dot": dot_no_edges}

    preschematic = design_intent.to_dot()

    dot_with_edges: Optional[str] = None
    for event in _run_edge_routing(client, dot_no_edges, preschematic, model):
        yield event
        if "_result" in event:
            dot_with_edges = event["_result"]

    if dot_with_edges is None:
        yield {"type": "error", "message": "Edge routing failed"}
        return

    yield {"type": "pipeline_phase", "phase": "layout"}

    footprints: dict[str, list[float]] = {}
    for mapping in selection.mappings:
        try:
            fp = json.loads(get_component_footprint(mapping.pdk_module))
            if "error" not in fp:
                footprints[mapping.component_id] = [fp["dx_um"], fp["dy_um"]]
        except Exception:
            pass

    layout_result_raw = compute_layout(dot_with_edges, json.dumps(footprints))
    try:
        layout_result = json.loads(layout_result_raw)
    except (json.JSONDecodeError, TypeError):
        layout_result = {"positions": {}, "error": "Layout computation returned invalid JSON"}

    positions = layout_result.get("positions", {})
    yield {"type": "layout_done", "positions": positions}

    sch_feedback = _validate_schematic(dot_with_edges, design_intent)
    if sch_feedback:
        yield {"type": "feedback", "source": "schematic_builder",
               "issues": [f.model_dump() for f in sch_feedback]}

    yield {"type": "pipeline_phase", "phase": "export"}

    try:
        ports_result = json.loads(find_open_ports(dot_with_edges))
    except (json.JSONDecodeError, TypeError):
        ports_result = {"open_ports": [], "circuit_ports": {}}

    _edges_dot_to_dsl(dot_with_edges, circuit_dsl)
    _enrich_circuit_dsl(
        circuit_dsl, selection, positions,
        footprints, ports_result.get("circuit_ports", {}),
    )

    gf_netlist_yaml = export_gf_netlist(json.dumps(circuit_dsl))

    yield {"type": "pipeline_done", "result": {
        "design_intent": design_intent.model_dump(),
        "selection": selection.model_dump(),
        "circuit_dsl": circuit_dsl,
        "dot_string": dot_with_edges,
        "gf_netlist_yaml": gf_netlist_yaml,
        "open_ports": ports_result,
        "footprints": footprints,
        "positions": positions,
    }}


# ---------------------------------------------------------------------------
# Main pipeline entry points
# ---------------------------------------------------------------------------

def run_pipeline_finalize(
    explore_state: dict,
    user_prompt: str,
    model: str = "o3-mini",
    clarifications: Optional[dict[str, str]] = None,
    max_tool_rounds: int = 10,
    max_critic_rounds: int = 2,
    schematic_feedback: Optional[str] = None,
) -> Generator[PipelineEvent, None, None]:
    """Run from Phase 2 onward: finalize interpreter, then selector, then builder.

    Called by Streamlit after the disambiguation stage.

    Args:
        schematic_feedback: If provided, user feedback from the review gate.
            Injected into the interpreter as upstream feedback so the pipeline
            addresses the requested changes.

    Yields events for each phase. The final event is either
    ``{"type": "pipeline_done", ...}`` or ``{"type": "error", ...}``.
    """
    client = OpenAI()

    upstream_fb = None
    if schematic_feedback:
        upstream_fb = (
            "USER SCHEMATIC FEEDBACK\n"
            "=======================\n"
            "The user reviewed the generated schematic and requested changes:\n\n"
            f"{schematic_feedback}\n\n"
            "You MUST address this feedback. Re-investigate with your tools and "
            "revise the DesignIntent accordingly."
        )

    # ── Phases 2-3: Finalize interpreter ──────────────────────────────
    design_intent: Optional[DesignIntent] = None
    for event in finalize_stream(
        messages=explore_state["messages"],
        tool_log=explore_state["tool_log"],
        user_prompt=user_prompt,
        extracted_dict=explore_state["extracted"],
        model=model,
        max_tool_rounds=max_tool_rounds,
        max_critic_rounds=max_critic_rounds,
        clarifications=clarifications,
        upstream_feedback=upstream_fb,
    ):
        yield event
        if event["type"] == "done":
            design_intent = event["result"]

    if design_intent is None:
        yield {"type": "error", "message": "Interpreter failed to produce DesignIntent"}
        return

    # ── Phase 4: Component Selection ──────────────────────────────────
    yield {"type": "pipeline_phase", "phase": "component_selection"}

    selection: Optional[ComponentSelection] = None
    for event in _run_component_selection(client, design_intent, model):
        yield event
        if "_result" in event:
            selection = event["_result"]

    if selection is None:
        yield {"type": "error", "message": "Component selection failed"}
        return

    # ── Phase 4.75: Validate Selection ────────────────────────────────
    sel_feedback = _validate_selection(selection, design_intent)

    fundamental = [f for f in sel_feedback if f.severity == "fundamental"]
    if fundamental:
        yield {"type": "feedback", "source": "component_selection",
               "target": "interpreter",
               "issues": [f.model_dump() for f in fundamental]}
        yield {"type": "error",
               "message": (
                   f"Component selection has {len(fundamental)} fundamental "
                   f"issue(s) that require interpreter revision."
               )}
        return

    if sel_feedback:
        yield {"type": "feedback", "source": "component_selection",
               "issues": [f.model_dump() for f in sel_feedback]}

    # ── Phase 4.8: LLM Compliance Check ──────────────────────────────
    yield {"type": "pipeline_phase", "phase": "compliance_check"}
    yield {"type": "phase", "phase": "compliance_check",
           "detail": "Verifying selected modules match design intent..."}

    compliance_feedback = _llm_compliance_check(client, selection, design_intent, model)
    if compliance_feedback:
        yield {"type": "feedback", "source": "compliance_check",
               "issues": [f.model_dump() for f in compliance_feedback]}
        non_compliant_ids = {
            cid for f in compliance_feedback for cid in f.affected_components
        }
        yield {"type": "phase", "phase": "compliance_check",
               "detail": (
                   f"Non-compliant mappings: {', '.join(non_compliant_ids)}. "
                   f"Downstream may produce incorrect results."
               )}
    else:
        yield {"type": "phase", "phase": "compliance_check",
               "detail": "All mappings compliant with design intent."}

    # ── Phase 5: Build circuit DSL + DOT (no edges) ──────────────────
    yield {"type": "pipeline_phase", "phase": "schematic_building"}

    circuit_dsl = _build_circuit_dsl(design_intent, selection)
    dot_no_edges = circuit_dsl_to_dot(json.dumps(circuit_dsl))

    if dot_no_edges.startswith("{"):
        parsed = json.loads(dot_no_edges)
        if "error" in parsed:
            yield {"type": "error", "message": f"DOT generation failed: {parsed['error']}"}
            return

    yield {"type": "dot_draft", "dot": dot_no_edges}

    preschematic = design_intent.to_dot()

    # ── Phase 5b: LLM Edge Routing ───────────────────────────────────
    dot_with_edges: Optional[str] = None
    for event in _run_edge_routing(client, dot_no_edges, preschematic, model):
        yield event
        if "_result" in event:
            dot_with_edges = event["_result"]

    if dot_with_edges is None:
        yield {"type": "error", "message": "Edge routing failed"}
        return

    # ── Phase 6: Footprints + Layout ─────────────────────────────────
    yield {"type": "pipeline_phase", "phase": "layout"}

    footprints: dict[str, list[float]] = {}
    for mapping in selection.mappings:
        try:
            fp = json.loads(get_component_footprint(mapping.pdk_module))
            if "error" not in fp:
                footprints[mapping.component_id] = [fp["dx_um"], fp["dy_um"]]
        except Exception:
            pass

    layout_result_raw = compute_layout(dot_with_edges, json.dumps(footprints))
    try:
        layout_result = json.loads(layout_result_raw)
    except (json.JSONDecodeError, TypeError):
        layout_result = {"positions": {}, "error": "Layout computation returned invalid JSON"}

    positions = layout_result.get("positions", {})
    yield {"type": "layout_done", "positions": positions}

    # ── Phase 6.5: Validate Schematic ────────────────────────────────
    sch_feedback = _validate_schematic(dot_with_edges, design_intent)
    if sch_feedback:
        yield {"type": "feedback", "source": "schematic_builder",
               "issues": [f.model_dump() for f in sch_feedback]}

    # ── Phase 7: Open Ports + Enrich DSL + GF Export ─────────────────
    yield {"type": "pipeline_phase", "phase": "export"}

    try:
        ports_result = json.loads(find_open_ports(dot_with_edges))
    except (json.JSONDecodeError, TypeError):
        ports_result = {"open_ports": [], "circuit_ports": {}}

    _edges_dot_to_dsl(dot_with_edges, circuit_dsl)

    _enrich_circuit_dsl(
        circuit_dsl, selection, positions,
        footprints, ports_result.get("circuit_ports", {}),
    )

    gf_netlist_yaml = export_gf_netlist(json.dumps(circuit_dsl))

    yield {"type": "pipeline_done", "result": {
        "design_intent": design_intent.model_dump(),
        "selection": selection.model_dump(),
        "circuit_dsl": circuit_dsl,
        "dot_string": dot_with_edges,
        "gf_netlist_yaml": gf_netlist_yaml,
        "open_ports": ports_result,
        "footprints": footprints,
        "positions": positions,
    }}


# ---------------------------------------------------------------------------
# Tiered feedback dispatcher
# ---------------------------------------------------------------------------

def run_pipeline_with_feedback(
    feedback_text: str,
    circuit_dsl: dict,
    selection_dict: dict,
    design_intent_dict: dict,
    dot_string: str,
    footprints: dict,
    explore_state: dict,
    user_prompt: str,
    model: str = "o3-mini",
    clarifications: Optional[dict[str, str]] = None,
    max_tool_rounds: int = 10,
    max_critic_rounds: int = 2,
) -> Generator[PipelineEvent, None, None]:
    """Classify user schematic feedback and route to the cheapest pipeline tier.

    Tier 1 (edge_edit)      -> run_pipeline_patch_edges
    Tier 2 (component_swap) -> run_pipeline_reselect
    Tier 3 (architectural)  -> run_pipeline_finalize (full re-run)
    """
    client = OpenAI()

    yield {"type": "pipeline_phase", "phase": "feedback_classification"}
    yield {"type": "phase", "phase": "feedback_classification",
           "detail": "Classifying feedback to determine minimal re-run scope..."}

    try:
        classification = _classify_feedback(
            client, feedback_text, circuit_dsl, selection_dict, dot_string, model,
        )
    except Exception as exc:
        yield {"type": "phase", "phase": "feedback_classification",
               "detail": f"Classification failed ({exc}), falling back to full re-run."}
        classification = FeedbackClassification(
            tier="architectural", edge_patches=[], component_swaps=[],
            reasoning=f"Classification error: {exc}",
        )

    tier = classification.tier
    yield {"type": "phase", "phase": "feedback_classification",
           "detail": f"Classified as **{tier}**: {classification.reasoning}"}

    if tier == "edge_edit" and classification.edge_patches:
        yield from run_pipeline_patch_edges(
            circuit_dsl=circuit_dsl,
            selection_dict=selection_dict,
            design_intent_dict=design_intent_dict,
            dot_string=dot_string,
            footprints=footprints or {},
            edge_patches=classification.edge_patches,
            model=model,
        )

    elif tier == "component_swap" and classification.component_swaps:
        yield from run_pipeline_reselect(
            design_intent_dict=design_intent_dict,
            prev_selection_dict=selection_dict,
            component_swaps=classification.component_swaps,
            model=model,
        )

    else:
        yield {"type": "phase", "phase": "feedback_classification",
               "detail": "Routing to full pipeline re-run (Tier 3: architectural)."}
        yield from run_pipeline_finalize(
            explore_state=explore_state,
            user_prompt=user_prompt,
            model=model,
            clarifications=clarifications,
            max_tool_rounds=max_tool_rounds,
            max_critic_rounds=max_critic_rounds,
            schematic_feedback=feedback_text,
        )


# ---------------------------------------------------------------------------
# Phase 8: Layout + Simulation (post-review)
# ---------------------------------------------------------------------------

def run_layout_simulation(
    gf_netlist_yaml: str,
    wl_start: float = 1.5,
    wl_stop: float = 1.6,
    wl_points: int = 200,
) -> Generator[PipelineEvent, None, None]:
    """Phase 8 — GDS layout rendering and SAX S-parameter simulation.

    Called by Streamlit *after* the user approves the schematic in the review
    gate.  Yields streaming events, ending with ``layout_sim_done``.
    """
    from mcp_servers.layout_sim_server import (
        render_gds_layout,
        run_sax_simulation,
        write_gds_file,
    )

    yield {"type": "pipeline_phase", "phase": "layout_gds"}
    yield {"type": "phase", "phase": "layout_gds",
           "detail": "Instantiating GDSFactory component and rendering layout..."}

    try:
        gds_result = render_gds_layout(gf_netlist_yaml)
    except Exception as exc:
        yield {"type": "error", "message": f"GDS layout failed: {exc}"}
        return

    routing_ok = gds_result.get("routing_ok", False)
    missing = gds_result.get("missing_models", [])
    if not routing_ok:
        yield {"type": "phase", "phase": "layout_gds",
               "detail": "Optical routing failed — layout rendered without links."}
    if missing:
        yield {"type": "phase", "phase": "layout_gds",
               "detail": f"Missing SAX models: {', '.join(missing[:5])}"}

    yield {"type": "gds_rendered", "gds_fig_b64": gds_result["gds_fig_b64"],
           "routing_ok": routing_ok, "missing_models": missing}

    yield {"type": "pipeline_phase", "phase": "simulation"}
    yield {"type": "phase", "phase": "simulation",
           "detail": f"Running SAX simulation ({wl_points} points, {wl_start}-{wl_stop} µm)..."}

    try:
        sim_result = run_sax_simulation(
            gf_netlist_yaml, wl_start, wl_stop, wl_points,
        )
    except Exception as exc:
        yield {"type": "error", "message": f"SAX simulation failed: {exc}"}
        return

    if sim_result.get("error"):
        yield {"type": "phase", "phase": "simulation",
               "detail": f"Simulation error: {sim_result['error']}"}
    else:
        n_params = len(sim_result.get("s_params", {}))
        yield {"type": "phase", "phase": "simulation",
               "detail": f"Simulation complete — {n_params} S-parameter(s) computed."}

    yield {"type": "pipeline_phase", "phase": "gds_export"}
    yield {"type": "phase", "phase": "gds_export",
           "detail": "Writing GDS file..."}

    gds_file = write_gds_file(gf_netlist_yaml, filename="circuit_output")

    yield {"type": "layout_sim_done", "result": {
        "gds_fig_b64": gds_result.get("gds_fig_b64", ""),
        "sax_fig_b64": sim_result.get("sax_fig_b64", ""),
        "s_params": sim_result.get("s_params", {}),
        "wavelengths": sim_result.get("wavelengths", []),
        "sim_error": sim_result.get("error"),
        "gds_file_path": gds_file.get("gds_path", ""),
        "gds_write_ok": gds_file.get("success", False),
        "routing_ok": routing_ok,
        "missing_models": missing,
    }}


def run_pipeline(
    user_prompt: str,
    model: str = "o3-mini",
    verbose: bool = True,
    max_tool_rounds: int = 15,
    max_grounding_rounds: int = 5,
    max_critic_rounds: int = 2,
) -> Optional[dict]:
    """Blocking wrapper — runs full pipeline without disambiguation.

    Returns the pipeline_done result dict, or None on failure.
    """
    explore_state: Optional[dict] = None

    for event in explore_and_ask(
        user_prompt, model, max_tool_rounds, max_grounding_rounds,
    ):
        if verbose:
            etype = event["type"]
            detail = event.get("detail", event.get("phase", ""))
            print(f"  [{etype}] {detail}")
        if event["type"] == "clarification":
            explore_state = event

    if explore_state is None:
        if verbose:
            print("  [error] explore_and_ask ended without clarification state")
        return None

    result: Optional[dict] = None
    for event in run_pipeline_finalize(
        explore_state=explore_state,
        user_prompt=user_prompt,
        model=model,
        clarifications=None,
        max_tool_rounds=max_tool_rounds,
        max_critic_rounds=max_critic_rounds,
    ):
        if verbose:
            etype = event["type"]
            detail = event.get("detail", event.get("phase", event.get("message", "")))
            print(f"  [{etype}] {detail}")
        if event["type"] == "pipeline_done":
            result = event["result"]
        elif event["type"] == "error":
            if verbose:
                print(f"  ERROR: {event['message']}")

    return result
