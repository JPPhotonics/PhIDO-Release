"""
Pipeline Orchestrator — deterministic controller for the full PhIDO pipeline.

Chains: Interpreter -> Component Selection -> Schematic Building
with validation gates and backward feedback loops.

Not an MCP server — called directly by the Streamlit UI or scripts.

Stages:
  Phases 0-1.75: Interpreter exploration (explore_and_ask)
  Phase 2      : Structured DesignIntent extraction (extract_design_intent)
  Phase 2.5    : Topology Gate (Clingo ASP architecture validation, Checkpoint A)
  Phase 3      : Critic Review (run_critic_review — skipped if Clingo found errors)
  Phase 4      : Component Selection (LLM matches ComponentIntents to PDK modules)
  Phase 4.75   : Selection Validation Gate (programmatic checks)
  Phase 4.8    : LLM Compliance Check (verify functional match of selections)
  Phase 5      : Circuit DSL Construction + DOT generation (deterministic)
  Phase 5.25   : AR Parameter Gate (Bedrock Automated Reasoning, Checkpoint B)
  Phase 5b     : Edge Routing (LLM adds port-level edges to DOT, with planarity retries)
  Phase 6      : Layout computation (Graphviz) + footprints (GDSFactory)
  Phase 6.5    : Schematic Validation Gate (programmatic checks)
  Phase 7      : Open ports + GDSFactory netlist export
"""

import json
import re
from typing import Any, Generator, Optional

_CODE_FENCE_RE = re.compile(r"```(?:dot|graphviz)?\s*\n?", re.IGNORECASE)

from mcp_servers.llm_client import create_client, LLMClient

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
    SettingEntry,
)
from mcp_servers.interpreter_agent import (
    explore_and_ask,
    finalize_stream,
    extract_design_intent,
    build_circuit_iterative,
    run_critic_review,
    AGENT_SYSTEM_PROMPT,
    ITERATIVE_BUILD_PROMPT,
    TOOLS as INTERPRETER_TOOLS,
    _ALL_BUILDER_TOOLS,
    _BUILDER_TOOL_NAMES,
    _dispatch_builder_tool,
    _execute_tool_raw,
    _PDK_CATALOG,
    CriticVerdict,
    CriticIssue,
)
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
from mcp_servers.ar_validator import validate_parameters
from mcp_servers.clingo_validator import validate_topology
from mcp_servers.circuit_graph import CircuitGraph

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
    client: LLMClient,
    feedback_text: str,
    circuit_dsl: dict,
    selection_dict: dict,
    dot_string: str,
) -> FeedbackClassification:
    """Use an LLM to classify user schematic feedback into a tier."""
    context = _build_schematic_context(circuit_dsl, selection_dict, dot_string)
    prompt = FEEDBACK_CLASSIFIER_PROMPT.replace("{context}", context)

    return client.complete_structured(
        messages=[{"role": "user", "content": feedback_text}],
        response_model=FeedbackClassification,
        system=prompt,
    )


# ---------------------------------------------------------------------------
# Phase 4: Component Selection
# ---------------------------------------------------------------------------

def _build_selection_from_intent(
    design_intent: DesignIntent,
) -> Optional[ComponentSelection]:
    """Build a ComponentSelection directly from pre-grounded ComponentIntents.

    When the iterative builder has already resolved PDK modules (via
    ``ComponentIntent.pdk_module``), there is no need to re-derive them
    through the LLM selection gate.  Returns ``None`` if any component
    lacks a ``pdk_module``, signalling that the LLM path is required.
    """
    if not all(c.pdk_module for c in design_intent.components):
        return None

    mappings: list[ComponentMapping] = []
    for comp in design_intent.components:
        params = _get_ground_truth_params(comp.pdk_module)  # type: ignore[arg-type]
        overrides = [
            SettingEntry(key=s.key, value=s.value) for s in comp.specs
        ]
        mappings.append(ComponentMapping(
            component_id=comp.id,
            pdk_module=comp.pdk_module,  # type: ignore[arg-type]
            match_quality="exact",
            port_config=comp.port_config or "",
            resolved_settings=[
                SettingEntry(key=k, value=str(v)) for k, v in params.items()
            ],
            user_overrides=overrides,
            notes=["Pre-grounded by iterative builder"],
        ))

    return ComponentSelection(
        design_intent=design_intent,
        mappings=mappings,
        unmapped=[],
        ambiguities=[],
    )


def _run_component_selection(
    client: LLMClient,
    design_intent: DesignIntent,
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

    llm_result: ComponentSelectionLLM = client.complete_structured(
        messages=[{"role": "user", "content": (
            f"## DesignIntent\n\n"
            f"```json\n{json.dumps(design_intent.summary(), indent=2)}\n```\n\n"
            f"## PDK Search Results\n{pdk_context}"
        )}],
        response_model=ComponentSelectionLLM,
        system=SELECTOR_PROMPT,
    )

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
    client: LLMClient,
    selection: ComponentSelection,
    design_intent: DesignIntent,
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

    try:
        verdict: ComplianceVerdict = client.complete_structured(
            messages=[{"role": "user", "content": "\n".join(review_parts)}],
            response_model=ComplianceVerdict,
            system=COMPLIANCE_PROMPT,
        )
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


def _render_dot_to_png(dot_string: str) -> Optional[bytes]:
    """Render a DOT string to PNG bytes via the graphviz package."""
    try:
        import graphviz
        src = graphviz.Source(dot_string)
        return src.pipe(format="png")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 5c: Visual Critic (optional — vision-capable models only)
# ---------------------------------------------------------------------------

_VISUAL_CRITIC_SYSTEM_PROMPT = """\
You are a rigorous photonic circuit schematic reviewer. You will be shown:

1. A rendered schematic diagram (image) of a photonic integrated circuit.
2. The original user prompt describing what they want.
3. A summary of the DesignIntent that was extracted from the user prompt.

**Schematic legend:**
- Each node label follows the format: ``ComponentID: Role (pdk_module)``
- Each edge follows the format: ``srcNode:port -- tgtNode:port``
- Unconnected ports may appear as dangling stubs on a node.

**Your job:**
Compare the schematic image against the user's original design request and the
DesignIntent summary. Check for the following:

1. **Component count**: Does the schematic contain the expected number of each
   component type described by the user?
2. **Connectivity**: Are all connections described by the user present? Are there
   any missing or extra edges?
3. **Orphaned nodes**: Are there any nodes with zero connections that should be
   connected?
4. **Topology**: Does the overall graph topology match the described architecture
   (e.g. cascaded, ring, MZI arms, balanced tree)?
5. **Visual layout**: Are there obvious edge crossings, overlapping nodes, or
   layout issues that suggest a structural problem?
6. **Port consistency**: Do the port labels on edges look reasonable for the
   component types involved?

For each issue found, classify severity:
- **major**: Missing component, wrong topology, orphaned node that should be
  connected, missing connections.
- **minor**: Slightly suboptimal layout, minor port naming concern, cosmetic issue.

Be thorough but fair. If the schematic correctly implements what the user asked
for, pass it. Do not nitpick cosmetic details unless they indicate a structural
problem.

After your analysis, state VERDICT: PASS or VERDICT: FAIL, then list any issues.
"""


def _run_visual_critic(
    client: "LLMClient",
    user_prompt: str,
    design_intent: "DesignIntent",
    dot_with_edges: str,
) -> Generator[PipelineEvent, None, None]:
    """Phase 5c: optional visual review of the rendered schematic.

    Automatically skipped when the model lacks vision support or when
    DOT-to-PNG rendering fails. Non-blocking: issues are yielded as
    feedback events but never abort the pipeline.
    """
    if not client.supports_vision:
        yield {"type": "phase", "phase": "visual_critic",
               "detail": "Skipped (model does not support vision input)."}
        return

    png_bytes = _render_dot_to_png(dot_with_edges)
    if png_bytes is None:
        yield {"type": "phase", "phase": "visual_critic",
               "detail": "Skipped (DOT rendering to PNG failed)."}
        return

    yield {"type": "phase", "phase": "visual_critic",
           "detail": "Vision model reviewing rendered schematic..."}

    di_summary = (
        f"Title: {design_intent.title}\n"
        f"Summary: {design_intent.brief_summary}\n"
        f"Components ({len(design_intent.components)}):\n"
        + "\n".join(
            f"  - {c.id}: {c.description} (type={c.component_type}, "
            f"ports={c.port_config or '?'}, pdk={c.pdk_module or 'ungrounded'})"
            for c in design_intent.components
        )
        + f"\nConnections ({len(design_intent.connections)}):\n"
        + "\n".join(
            f"  - {c.from_component} → {c.to_component}"
            + (f" ({c.from_port}→{c.to_port})" if c.from_port else "")
            for c in design_intent.connections
        )
    )

    vision_text = (
        f"## Original User Prompt\n{user_prompt}\n\n"
        f"## DesignIntent Summary\n{di_summary}\n\n"
        f"## Task\n"
        f"Review the attached schematic image. Does it faithfully implement "
        f"what the user asked for? Identify any structural issues."
    )

    try:
        resp = client.complete_vision(
            text=vision_text,
            image_bytes=png_bytes,
            system=_VISUAL_CRITIC_SYSTEM_PROMPT,
        )
    except Exception as exc:
        yield {"type": "phase", "phase": "visual_critic",
               "detail": f"Vision critic call failed: {exc}"}
        return

    analysis = resp.content or ""
    yield {"type": "agent_text", "content": f"**Visual critic analysis:**\n{analysis}"}

    try:
        verdict = client.complete_structured(
            messages=[
                {"role": "user", "content": (
                    f"Based on the following visual schematic review, produce a "
                    f"structured CriticVerdict.\n\n{analysis}"
                )},
            ],
            response_model=CriticVerdict,
        )
    except Exception:
        verdict = CriticVerdict(
            passed=True, issues=[],
            summary="Could not extract structured verdict; assuming pass.",
        )

    yield {"type": "visual_critic_verdict", "verdict": verdict.model_dump()}

    if not verdict.passed:
        yield {"type": "feedback", "source": "visual_critic",
               "issues": [i.model_dump() for i in verdict.issues]}


def _run_edge_routing(
    client: LLMClient,
    dot_with_ports: str,
    preschematic_dot: str,
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

    prompt = (
        f"{EDGE_ROUTING_PROMPT}\n\n"
        f"INPUT graph1:\n{dot_with_ports}\n\n"
        f"INPUT graph2:\n{preschematic_dot}\n"
    )
    resp = client.complete(
        messages=[{"role": "user", "content": prompt}],
    )
    dot_with_edges = _sanitise_dot(resp.content or "")

    for attempt in range(max_retries):
        try:
            planarity = json.loads(check_planarity(dot_with_edges))
        except (json.JSONDecodeError, TypeError):
            break

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
        resp = client.complete(
            messages=[{"role": "user", "content": retry_prompt}],
        )
        dot_with_edges = _sanitise_dot(resp.content or "")

    yield {"type": "phase", "phase": "edge_routing",
           "detail": "Verifying edge constraints..."}

    verify_prompt = f"{EDGE_VERIFY_PROMPT}\n\n{dot_with_edges}"
    resp = client.complete(
        messages=[{"role": "user", "content": verify_prompt}],
    )
    dot_verified = _sanitise_dot(resp.content or "")

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

def _coerce_setting_value(value_str: str, reference_value=None):
    """Convert a string setting value to the native type matching the reference.

    GDSFactory components expect float/int parameters, not strings.
    """
    if reference_value is not None:
        target_type = type(reference_value)
        try:
            return target_type(value_str)
        except (ValueError, TypeError):
            pass
    for converter in (int, float):
        try:
            return converter(value_str)
        except (ValueError, TypeError):
            continue
    return value_str


def _get_ground_truth_params(pdk_module: str) -> dict:
    """Fetch real GDSFactory default settings for a PDK module.

    Returns a dict of param_name -> value (preserving native types so
    GDSFactory receives floats/ints rather than quoted strings in YAML).
    Falls back to an empty dict if GDSFactory instantiation fails.
    """
    try:
        result = json.loads(get_module_params(pdk_module))
        if "error" in result:
            return {}
        return result.get("settings", {})
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
                    params[override.key] = _coerce_setting_value(
                        override.value, params.get(override.key)
                    )
            display = (comp.role.title() if comp.role else comp.description)
            nodes[comp.id] = {
                "component": mapping.pdk_module,
                "properties": {"ports": mapping.port_config},
                "params": params,
                "label": f"{comp.id}: {display}\\n({mapping.pdk_module})",
            }
    # Populate edges from connections that have port-level info
    # (set by the iterative builder). Format: "C1,o3: C3,o2"
    edges: dict[str, dict] = {}
    for i, conn in enumerate(design_intent.connections):
        if conn.from_port and conn.to_port:
            if conn.from_component in nodes and conn.to_component in nodes:
                edges[f"E{i + 1}"] = {
                    "link": f"{conn.from_component},{conn.from_port}: "
                            f"{conn.to_component},{conn.to_port}"
                }

    return {
        "doc": {
            "name": design_intent.title,
            "description": design_intent.brief_summary,
        },
        "nodes": nodes,
        "edges": edges,
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
    model: str = "gpt-5.4",
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
    model: str = "gpt-5.4",
) -> Generator[PipelineEvent, None, None]:
    """Tier 2: re-select only the targeted components, keeping all others
    from the previous selection, then rebuild DSL + edge routing + layout + export.
    """
    client = create_client(model)
    design_intent = DesignIntent(**design_intent_dict)

    # ── Topology Gate (Checkpoint A — Clingo) ────────────────────────
    yield {"type": "pipeline_phase", "phase": "topology_gate"}
    yield {"type": "phase", "phase": "topology_gate",
           "detail": "Validating topology against architecture rules..."}

    topo_feedback = validate_topology(design_intent)
    if topo_feedback:
        topo_fundamental = [f for f in topo_feedback if f.severity == "fundamental"]
        yield {"type": "feedback", "source": "topology_gate",
               "issues": [f.model_dump() for f in topo_feedback]}
        if topo_fundamental:
            yield {"type": "error",
                   "message": (
                       f"Topology validation found {len(topo_fundamental)} "
                       f"fundamental issue(s) requiring interpreter revision."
                   )}
            return
        yield {"type": "phase", "phase": "topology_gate",
               "detail": "Topology gate: issues noted (non-blocking)."}
    else:
        yield {"type": "phase", "phase": "topology_gate",
               "detail": "Topology gate passed (or skipped)."}

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
            llm_result: ComponentSelectionLLM = client.complete_structured(
                messages=messages,
                response_model=ComponentSelectionLLM,
            )
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
        compliance_feedback = _llm_compliance_check(client, partial_sel, design_intent)
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
    if circuit_dsl.get("edges"):
        yield {"type": "phase", "phase": "edge_routing",
               "detail": "Using port-level edges from iterative builder (skipping LLM routing)."}
        edge_lines: list[str] = []
        for _eid, einfo in circuit_dsl["edges"].items():
            link = einfo.get("link", "")
            parts = link.split(": ")
            if len(parts) == 2:
                src_node, src_port = parts[0].split(",", 1)
                tgt_node, tgt_port = parts[1].split(",", 1)
                edge_lines.append(
                    f"  {src_node}:{src_port} -- {tgt_node}:{tgt_port};"
                )
        dot_lines = dot_no_edges.rstrip().rstrip("}").rstrip()
        dot_with_edges = dot_lines + "\n" + "\n".join(edge_lines) + "\n}"
        yield {"type": "edge_routing_done", "_result": dot_with_edges}
    else:
        for event in _run_edge_routing(client, dot_no_edges, preschematic):
            yield event
            if "_result" in event:
                dot_with_edges = event["_result"]

    if dot_with_edges is None:
        yield {"type": "error", "message": "Edge routing failed"}
        return

    # Phase 5c: Visual critic (optional)
    yield {"type": "pipeline_phase", "phase": "visual_critic"}
    _reselect_prompt = f"{design_intent.title}: {design_intent.brief_summary}"
    for vc_event in _run_visual_critic(client, _reselect_prompt, design_intent, dot_with_edges):
        yield vc_event

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

_MAX_VALIDATION_RETRIES = 2


def _format_validation_feedback(
    errors: list[PipelineFeedback],
    gate_name: str,
    attempt: int,
    max_attempts: int,
) -> str:
    """Format validation errors into upstream feedback for the interpreter."""
    items = "\n".join(
        f"  - [{f.severity}] {f.description}"
        + (f" (suggested: {f.suggested_action})" if f.suggested_action else "")
        for f in errors
    )
    return (
        f"VALIDATION FEEDBACK (attempt {attempt}/{max_attempts})\n"
        f"{'=' * 50}\n"
        f"The {gate_name} found the following issues with your DesignIntent:\n\n"
        f"{items}\n\n"
        f"You MUST revise the DesignIntent to fix these issues. Pay close\n"
        f"attention to the component_type and role fields — Clingo rules\n"
        f"count components by their functional role (splitter, combiner, etc.),\n"
        f"not their device type (coupler, mmi, etc.).\n"
        f"Re-investigate with your tools if needed."
    )


_AUTO_ITERATIVE_THRESHOLD = 12


def run_pipeline_finalize(
    explore_state: dict,
    user_prompt: str,
    model: str = "gpt-5.4",
    clarifications: Optional[dict[str, str]] = None,
    max_tool_rounds: int = 10,
    max_critic_rounds: int = 2,
    schematic_feedback: Optional[str] = None,
    extraction_mode: str = "single_shot",
    orchestration: str = "rigid",
    unified_resume_state: Optional[dict] = None,
) -> Generator[PipelineEvent, None, None]:
    """Run from Phase 2 onward: extract → Clingo → critic → selector → builder.

    Called by Streamlit after the disambiguation stage.

    Args:
        extraction_mode: ``"single_shot"`` (default) uses the existing
            ``extract_design_intent`` path.  ``"iterative"`` uses the new
            ``build_circuit_iterative`` tool-calling loop.  ``"auto"``
            picks iterative when the estimated component count exceeds
            ``_AUTO_ITERATIVE_THRESHOLD``.
        schematic_feedback: If provided, user feedback from the review gate.
        orchestration: ``"rigid"`` (default) runs the phased pipeline.
            ``"unified"`` runs a single continuous LLM session.
        unified_resume_state: If provided, resume a paused unified session
            (after the user answered an ``ask_user`` question).

    Yields events for each phase. The final event is either
    ``{"type": "pipeline_done", ...}`` or ``{"type": "error", ...}``
    or ``{"type": "validation_failed", ...}`` (if retries exhaust).
    """
    if orchestration == "unified":
        yield from run_pipeline_unified(
            user_prompt, model,
            resume_state=unified_resume_state,
            schematic_feedback=schematic_feedback,
        )
        return

    client = create_client(model)

    # Resolve "auto" mode
    effective_mode = extraction_mode
    if effective_mode == "auto":
        extracted = explore_state.get("extracted", {})
        n_components = len(extracted.get("components", []))
        n_specs = len(extracted.get("specs", []))
        estimated = n_components + n_specs
        if estimated >= _AUTO_ITERATIVE_THRESHOLD:
            effective_mode = "iterative"
            yield {"type": "phase", "phase": "mode_selection",
                   "detail": (f"Auto mode: estimated {estimated} concepts "
                              f"(>= {_AUTO_ITERATIVE_THRESHOLD}), using iterative builder.")}
        else:
            effective_mode = "single_shot"
            yield {"type": "phase", "phase": "mode_selection",
                   "detail": (f"Auto mode: estimated {estimated} concepts "
                              f"(< {_AUTO_ITERATIVE_THRESHOLD}), using single-shot extraction.")}

    # ── Iterative builder path ────────────────────────────────────────
    if effective_mode == "iterative":
        yield {"type": "pipeline_phase", "phase": "iterative_build"}
        design_intent = None
        for event in build_circuit_iterative(
            messages=explore_state["messages"],
            tool_log=explore_state["tool_log"],
            extracted_dict=explore_state["extracted"],
            model=model,
            max_rounds=1000,
            requirement_manifest_dict=explore_state.get("requirement_manifest"),
        ):
            yield event
            if event["type"] == "done":
                design_intent = event["result"]

        if design_intent is None:
            yield {"type": "error", "message": "Iterative builder failed to produce DesignIntent"}
            return

        # Skip directly to Clingo + downstream (no critic for iterative builds
        # since the graph object already enforces structural validity)
        yield {"type": "pipeline_phase", "phase": "topology_gate"}
        yield {"type": "phase", "phase": "topology_gate",
               "detail": "Validating topology against architecture rules..."}

        topo_feedback = validate_topology(design_intent)
        if topo_feedback:
            topo_fundamental = [f for f in topo_feedback if f.severity == "fundamental"]
            yield {"type": "feedback", "source": "topology_gate",
                   "issues": [f.model_dump() for f in topo_feedback]}
            if topo_fundamental:
                yield {"type": "phase", "phase": "topology_gate",
                       "detail": f"Topology gate: {len(topo_fundamental)} fundamental issue(s) noted."}
            else:
                yield {"type": "phase", "phase": "topology_gate",
                       "detail": "Topology gate: issues noted (non-blocking)."}
        else:
            yield {"type": "phase", "phase": "topology_gate",
                   "detail": "Topology gate passed (or skipped)."}

        # Component selection — reuse PDK grounding from iterative builder if available
        yield {"type": "pipeline_phase", "phase": "component_selection"}
        selection = _build_selection_from_intent(design_intent)
        if selection is not None:
            yield {"type": "phase", "phase": "component_selection",
                   "detail": "Using pre-grounded PDK mappings from iterative builder."}
        else:
            for event in _run_component_selection(client, design_intent):
                yield event
                if "_result" in event:
                    selection = event["_result"]

        if selection is None:
            yield {"type": "error",
                   "message": "Component selection failed to produce any mappings."}
            return

        sel_feedback = _validate_selection(selection, design_intent)
        if sel_feedback:
            yield {"type": "feedback", "source": "component_selection",
                   "issues": [f.model_dump() for f in sel_feedback]}

    else:
        # ── Single-shot path (existing behavior) ─────────────────────
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

        max_attempts = _MAX_VALIDATION_RETRIES + 1
        design_intent = None
        selection = None

        for attempt in range(1, max_attempts + 1):
            # ── Phase 2: Extract DesignIntent ─────────────────────────
            if attempt > 1:
                yield {"type": "phase", "phase": "validation_retry",
                       "detail": f"Retry {attempt - 1}/{_MAX_VALIDATION_RETRIES}: "
                                 f"re-running interpreter with validation feedback..."}

            design_intent = None
            for event in extract_design_intent(
                messages=explore_state["messages"],
                tool_log=explore_state["tool_log"],
                extracted_dict=explore_state["extracted"],
                model=model,
                max_tool_rounds=max_tool_rounds,
                clarifications=clarifications if attempt == 1 else None,
                upstream_feedback=upstream_fb,
                requirement_manifest_dict=explore_state.get("requirement_manifest"),
            ):
                yield event
                if event["type"] == "done":
                    design_intent = event["result"]
                    explore_state = {
                        **explore_state,
                        "messages": event.get("messages", explore_state["messages"]),
                    }

            if design_intent is None:
                yield {"type": "error", "message": "Interpreter failed to produce DesignIntent"}
                return

            # ── Phase 2.5: Topology Gate (Checkpoint A — Clingo) ──────
            yield {"type": "pipeline_phase", "phase": "topology_gate"}
            yield {"type": "phase", "phase": "topology_gate",
                   "detail": "Validating topology against architecture rules..."}

            topo_feedback = validate_topology(design_intent)
            if topo_feedback:
                topo_fundamental = [f for f in topo_feedback if f.severity == "fundamental"]
                yield {"type": "feedback", "source": "topology_gate",
                       "issues": [f.model_dump() for f in topo_feedback]}
                if topo_fundamental:
                    if attempt < max_attempts:
                        upstream_fb = _format_validation_feedback(
                            topo_fundamental, "topology gate (Clingo)",
                            attempt, max_attempts,
                        )
                        continue
                    yield {"type": "validation_failed",
                           "gate": "topology_gate",
                           "issues": [f.model_dump() for f in topo_fundamental],
                           "design_intent": design_intent.model_dump(),
                           "attempts": attempt,
                           "message": (
                               f"Topology validation still has {len(topo_fundamental)} "
                               f"fundamental issue(s) after {_MAX_VALIDATION_RETRIES} "
                               f"retry(ies)."
                           )}
                    return
                yield {"type": "phase", "phase": "topology_gate",
                       "detail": "Topology gate: issues noted (non-blocking)."}
            else:
                yield {"type": "phase", "phase": "topology_gate",
                       "detail": "Topology gate passed (or skipped)."}

            # ── Phase 3: Critic Review ────────────────────────────────
            for event in run_critic_review(
                user_prompt=user_prompt,
                extracted_dict=explore_state["extracted"],
                design_intent=design_intent,
                messages=explore_state["messages"],
                model=model,
                max_critic_rounds=max_critic_rounds,
                max_tool_rounds=max_tool_rounds,
                clarifications=clarifications if attempt == 1 else None,
            ):
                yield event
                if event["type"] == "done":
                    design_intent = event["result"]
                    explore_state = {
                        **explore_state,
                        "messages": event.get("messages", explore_state["messages"]),
                    }

            # ── Phase 4: Component Selection ──────────────────────────
            yield {"type": "pipeline_phase", "phase": "component_selection"}

            selection = None
            for event in _run_component_selection(client, design_intent):
                yield event
                if "_result" in event:
                    selection = event["_result"]

            if selection is None:
                no_selection_error = PipelineFeedback(
                    severity="fundamental",
                    description="Component selection failed to produce any mappings.",
                    suggested_action="Ensure every ComponentIntent has a clear "
                                     "component_type and role that maps to a PDK module.",
                )
                if attempt < max_attempts:
                    upstream_fb = _format_validation_feedback(
                        [no_selection_error],
                        "component selection", attempt, max_attempts,
                    )
                    continue
                yield {"type": "validation_failed",
                       "gate": "component_selection",
                       "issues": [no_selection_error.model_dump()],
                       "design_intent": design_intent.model_dump(),
                       "attempts": attempt,
                       "message": "Component selection produced no mappings after retries."}
                return

            # ── Phase 4.75: Validate Selection ────────────────────────
            sel_feedback = _validate_selection(selection, design_intent)

            sel_fundamental = [f for f in sel_feedback if f.severity == "fundamental"]
            if sel_fundamental:
                print(f"── Selection gate: {len(sel_fundamental)} fundamental "
                      f"issue(s), attempt {attempt}/{max_attempts} ──")
                yield {"type": "feedback", "source": "component_selection",
                       "target": "interpreter",
                       "issues": [f.model_dump() for f in sel_fundamental]}
                if attempt < max_attempts:
                    upstream_fb = _format_validation_feedback(
                        sel_fundamental, "component selection validation",
                        attempt, max_attempts,
                    )
                    continue
                print("── Yielding validation_failed for component_selection ──")
                yield {"type": "validation_failed",
                       "gate": "component_selection",
                       "issues": [f.model_dump() for f in sel_fundamental],
                       "design_intent": design_intent.model_dump(),
                       "attempts": attempt,
                       "message": (
                           f"Component selection has {len(sel_fundamental)} fundamental "
                           f"issue(s) after {_MAX_VALIDATION_RETRIES} retry(ies)."
                       )}
                return

            if sel_feedback:
                yield {"type": "feedback", "source": "component_selection",
                       "issues": [f.model_dump() for f in sel_feedback]}

            break  # all gates passed, exit retry loop

    # ── Phase 4.8: LLM Compliance Check ──────────────────────────────
    yield {"type": "pipeline_phase", "phase": "compliance_check"}
    yield {"type": "phase", "phase": "compliance_check",
           "detail": "Verifying selected modules match design intent..."}

    compliance_feedback = _llm_compliance_check(client, selection, design_intent)
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

    # ── Phase 5.25: AR Parameter Gate (Checkpoint B) ──────────────────
    yield {"type": "pipeline_phase", "phase": "ar_parameter_gate"}
    yield {"type": "phase", "phase": "ar_parameter_gate",
           "detail": "Validating component parameters against AR policy..."}

    ar_param_feedback = validate_parameters(circuit_dsl)
    if ar_param_feedback:
        yield {"type": "feedback", "source": "ar_parameter_gate",
               "issues": [f.model_dump() for f in ar_param_feedback]}
        yield {"type": "phase", "phase": "ar_parameter_gate",
               "detail": "AR parameter gate: issues detected (non-blocking)."}
    else:
        yield {"type": "phase", "phase": "ar_parameter_gate",
               "detail": "AR parameter gate passed (or skipped)."}

    # ── Phase 5b: Edge Routing ─────────────────────────────────────
    # If the DSL already has edges (iterative builder), inject them into
    # the DOT directly.  Otherwise fall back to LLM edge routing.
    dot_with_edges: Optional[str] = None
    if circuit_dsl.get("edges"):
        yield {"type": "phase", "phase": "edge_routing",
               "detail": "Using port-level edges from iterative builder (skipping LLM routing)."}
        edge_lines: list[str] = []
        for _eid, einfo in circuit_dsl["edges"].items():
            link = einfo.get("link", "")
            parts = link.split(": ")
            if len(parts) == 2:
                src_node, src_port = parts[0].split(",", 1)
                tgt_node, tgt_port = parts[1].split(",", 1)
                edge_lines.append(
                    f"  {src_node}:{src_port} -- {tgt_node}:{tgt_port};"
                )
        # Insert edges before the closing brace of the DOT
        dot_lines = dot_no_edges.rstrip().rstrip("}").rstrip()
        dot_with_edges = dot_lines + "\n" + "\n".join(edge_lines) + "\n}"
        yield {"type": "edge_routing_done", "_result": dot_with_edges}
    else:
        for event in _run_edge_routing(client, dot_no_edges, preschematic):
            yield event
            if "_result" in event:
                dot_with_edges = event["_result"]

    if dot_with_edges is None:
        yield {"type": "error", "message": "Edge routing failed"}
        return

    # ── Phase 5c: Visual Critic (optional) ───────────────────────────
    yield {"type": "pipeline_phase", "phase": "visual_critic"}
    for vc_event in _run_visual_critic(client, user_prompt, design_intent, dot_with_edges):
        yield vc_event

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
# Unified orchestration strategy (single continuous LLM session)
# ---------------------------------------------------------------------------

_ASK_USER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user a clarification question when there is genuine "
            "ambiguity in their design request that you cannot resolve through "
            "tool calls alone.  Use this ONLY when a design decision critically "
            "depends on user preference (e.g. which tuning mechanism, target "
            "wavelength band, specific port count).  Do NOT ask about things "
            "you can determine from the PDK or KG."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to ask the user",
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Why you need this information — what you found in "
                        "tool results that raised the ambiguity"
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Concrete answer choices if applicable "
                        "(empty array for open-ended questions)"
                    ),
                },
                "default": {
                    "type": "string",
                    "description": "What you would assume if the user doesn't answer",
                },
            },
            "required": ["question", "context", "default"],
        },
    },
}

_UNIFIED_SYSTEM_PROMPT = (
    AGENT_SYSTEM_PROMPT
    + "\n\n--- BUILDER INSTRUCTIONS ---\n\n"
    + ITERATIVE_BUILD_PROMPT
    + "\n\n--- OVERALL WORKFLOW ---\n\n"
    "You are operating in UNIFIED mode. In a single session you will:\n"
    "1. Explore the user's request using PDK and KG tools.\n"
    "2. If you encounter genuine ambiguity that cannot be resolved from your "
    "tool results, use the ask_user tool to get a clarification from the user. "
    "Only ask when the design truly depends on user preference — do not ask "
    "about things you can determine from the PDK or KG.\n"
    "3. Once you understand the design, build the circuit using builder tools "
    "(add_component, connect, replicate_stage, get_open_ports, get_state).\n"
    "4. When the circuit is complete, call finalize.\n\n"
    "Do NOT wait for further instructions between exploration and building — "
    "transition seamlessly once all ambiguities are resolved."
)


def run_pipeline_unified(
    user_prompt: str,
    model: str = "gpt-5.4",
    max_rounds: int = 200,
    resume_state: Optional[dict] = None,
    schematic_feedback: Optional[str] = None,
) -> Generator[PipelineEvent, None, None]:
    """Unified orchestration: single continuous LLM session for exploration + building.

    The LLM explores the design space and builds the circuit in one session,
    then deterministic gates run sequentially on the result.

    When the LLM calls the ``ask_user`` tool, the generator yields a
    ``user_question`` event containing the full session state and returns.
    The caller (Streamlit) collects the answer and resumes by passing
    ``resume_state`` on the next invocation.

    If ``schematic_feedback`` is provided (Tier 3 architectural re-run),
    the feedback is appended to the user prompt so the LLM can revise
    the design in a fresh unified session.

    Yields the same event types as ``run_pipeline_finalize``, plus
    ``{"type": "user_question", ...}`` when disambiguation is needed.
    """
    import logging
    _log = logging.getLogger(__name__)

    unified_tools = [_ASK_USER_TOOL] + (_ALL_BUILDER_TOOLS or [])

    client = create_client(model)

    if resume_state:
        graph = resume_state["graph"]
        messages = resume_state["messages"]
        tool_log = resume_state["tool_log"]
        start_round = resume_state["round_num"]
        finalized = resume_state.get("finalized", False)

        pending_calls = resume_state["pending_ask_user_calls"]
        user_answers = resume_state.get("user_answers", {})
        for tc in pending_calls:
            question_text = ""
            try:
                question_text = json.loads(tc.arguments).get("question", "")
            except Exception:
                pass
            answer = user_answers.get(question_text, "(no answer provided)")
            messages.append(
                client.tool_result_message(
                    tc, json.dumps({"user_answer": answer})
                )
            )

        yield {"type": "phase", "phase": "unified_session",
               "detail": f"Resuming with {len(user_answers)} answer(s)..."}
    else:
        graph = CircuitGraph(pdk_catalog=_PDK_CATALOG)
        effective_prompt = user_prompt
        if schematic_feedback:
            effective_prompt = (
                f"{user_prompt}\n\n"
                f"SCHEMATIC REVISION REQUEST\n"
                f"==========================\n"
                f"A previous design was generated for this prompt but the user "
                f"reviewed it and requested architectural changes:\n\n"
                f"{schematic_feedback}\n\n"
                f"You MUST address this feedback when building the revised circuit."
            )
        messages = [
            {"role": "system", "content": _UNIFIED_SYSTEM_PROMPT},
            {"role": "user", "content": effective_prompt},
        ]
        tool_log = []
        start_round = 0
        finalized = False

        phase_detail = "Re-running unified session with user feedback..." if schematic_feedback \
            else f"Starting unified session (max {max_rounds} rounds)..."
        yield {"type": "pipeline_phase", "phase": "unified_session"}
        yield {"type": "phase", "phase": "unified_session",
               "detail": phase_detail}

    for round_num in range(start_round, max_rounds):
        try:
            resp = client.complete(
                messages,
                tools=unified_tools,
            )
        except Exception as exc:
            _log.error("API error on round %d: %s", round_num + 1, exc)
            yield {"type": "error", "message": f"LLM API error: {exc}"}
            return

        messages.append(client.assistant_message(resp))

        if resp.tool_calls:
            tool_names = [tc.name for tc in resp.tool_calls]
            yield {"type": "phase", "phase": "unified_session",
                   "detail": f"Round {round_num + 1}: "
                             f"{len(resp.tool_calls)} tool call(s) — "
                             f"{', '.join(tool_names)}"}

            # Separate ask_user calls from the rest so we can process
            # normal tools first, then pause for user input if needed.
            ask_user_calls: list = []

            for tc in resp.tool_calls:
                if tc.name == "ask_user":
                    ask_user_calls.append(tc)
                    continue

                try:
                    parsed_args = json.loads(tc.arguments)
                except json.JSONDecodeError:
                    parsed_args = {}
                    messages.append(
                        client.tool_result_message(
                            tc, json.dumps({"error": "Invalid JSON arguments"})
                        )
                    )
                    continue

                tool_log.append((tc.name, parsed_args))
                yield {"type": "tool_call", "name": tc.name, "args": parsed_args}

                if tc.name in _BUILDER_TOOL_NAMES:
                    try:
                        result_str = _dispatch_builder_tool(tc.name, tc.arguments, graph)
                    except Exception as exc:
                        result_str = json.dumps({"error": str(exc)})
                else:
                    _, result_str = _execute_tool_raw(tc.name, tc.arguments)

                yield {"type": "tool_result", "name": tc.name, "result": result_str}
                messages.append(client.tool_result_message(tc, result_str))

                if tc.name in _BUILDER_TOOL_NAMES:
                    dot = graph.to_dot(highlight=True)
                    state = graph.get_state()
                    yield {
                        "type": "circuit_updated",
                        "dot": dot,
                        "component_count": state["total_components"],
                        "connection_count": state["total_connections"],
                    }

                if tc.name == "finalize":
                    try:
                        result_data = json.loads(result_str)
                    except (json.JSONDecodeError, TypeError):
                        result_data = {}
                    if result_data.get("status") != "error":
                        finalized = True

            # If the LLM asked clarification questions, pause the generator
            # and hand control back to the UI for user input.
            if ask_user_calls and not finalized:
                questions = []
                for tc in ask_user_calls:
                    try:
                        q_args = json.loads(tc.arguments)
                    except json.JSONDecodeError:
                        q_args = {"question": "(parse error)", "context": "",
                                  "default": ""}
                    questions.append(q_args)
                    yield {"type": "tool_call", "name": "ask_user", "args": q_args}

                yield {
                    "type": "user_question",
                    "questions": questions,
                    "resume_state": {
                        "graph": graph,
                        "messages": messages,
                        "tool_log": tool_log,
                        "round_num": round_num,
                        "finalized": finalized,
                        "pending_ask_user_calls": ask_user_calls,
                        "model": model,
                    },
                }
                return  # Pause — caller will resume with answers

            if finalized:
                break
        else:
            content = resp.content or ""
            if content.strip():
                yield {"type": "agent_text", "content": content}
            if finalized:
                break

    if not finalized:
        graph.finalize("Untitled Circuit", "Auto-finalized — max rounds reached.", force=True)
        yield {"type": "phase", "phase": "unified_session",
               "detail": "Max rounds reached — auto-finalizing."}

    design_intent = graph.to_design_intent()

    yield {"type": "done", "result": design_intent}
    yield {"type": "phase", "phase": "unified_session",
           "detail": f"Session complete: {len(design_intent.components)} components, "
                     f"{len(design_intent.connections)} connections."}

    # ── Deterministic gates ───────────────────────────────────────────

    # Topology gate
    yield {"type": "pipeline_phase", "phase": "topology_gate"}
    yield {"type": "phase", "phase": "topology_gate",
           "detail": "Validating topology against architecture rules..."}

    topo_feedback = validate_topology(design_intent)
    if topo_feedback:
        topo_fundamental = [f for f in topo_feedback if f.severity == "fundamental"]
        yield {"type": "feedback", "source": "topology_gate",
               "issues": [f.model_dump() for f in topo_feedback]}
        if topo_fundamental:
            yield {"type": "phase", "phase": "topology_gate",
                   "detail": f"Topology gate: {len(topo_fundamental)} fundamental issue(s)."}
        else:
            yield {"type": "phase", "phase": "topology_gate",
                   "detail": "Topology gate: issues noted (non-blocking)."}
    else:
        yield {"type": "phase", "phase": "topology_gate",
               "detail": "Topology gate passed (or skipped)."}

    # Component selection — reuse PDK grounding from unified builder when available
    yield {"type": "pipeline_phase", "phase": "component_selection"}
    selection = _build_selection_from_intent(design_intent)
    if selection is not None:
        yield {"type": "phase", "phase": "component_selection",
               "detail": "Using pre-grounded PDK mappings from unified builder."}
    else:
        for event in _run_component_selection(client, design_intent):
            yield event
            if "_result" in event:
                selection = event["_result"]

    if selection is None:
        yield {"type": "error", "message": "Component selection failed."}
        return

    sel_feedback = _validate_selection(selection, design_intent)
    if sel_feedback:
        yield {"type": "feedback", "source": "component_selection",
               "issues": [f.model_dump() for f in sel_feedback]}

    # Compliance check
    compliance_feedback = _llm_compliance_check(client, selection, design_intent)
    if compliance_feedback:
        yield {"type": "feedback", "source": "compliance_check",
               "issues": [f.model_dump() for f in compliance_feedback]}

    # Build circuit DSL
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

    # AR Parameter gate
    yield {"type": "pipeline_phase", "phase": "ar_parameter_gate"}
    ar_param_feedback = validate_parameters(circuit_dsl)
    if ar_param_feedback:
        yield {"type": "feedback", "source": "ar_parameter_gate",
               "issues": [f.model_dump() for f in ar_param_feedback]}

    # Edge routing — use builder edges if available
    dot_with_edges: Optional[str] = None
    if circuit_dsl.get("edges"):
        yield {"type": "phase", "phase": "edge_routing",
               "detail": "Using port-level edges from builder (skipping LLM routing)."}
        edge_lines: list[str] = []
        for _eid, einfo in circuit_dsl["edges"].items():
            link = einfo.get("link", "")
            parts = link.split(": ")
            if len(parts) == 2:
                src_node, src_port = parts[0].split(",", 1)
                tgt_node, tgt_port = parts[1].split(",", 1)
                edge_lines.append(f"  {src_node}:{src_port} -- {tgt_node}:{tgt_port};")
        dot_lines = dot_no_edges.rstrip().rstrip("}").rstrip()
        dot_with_edges = dot_lines + "\n" + "\n".join(edge_lines) + "\n}"
    else:
        for event in _run_edge_routing(client, dot_no_edges, preschematic):
            yield event
            if "_result" in event:
                dot_with_edges = event["_result"]

    if dot_with_edges is None:
        yield {"type": "error", "message": "Edge routing failed"}
        return

    # Phase 5c: Visual critic (optional)
    yield {"type": "pipeline_phase", "phase": "visual_critic"}
    for vc_event in _run_visual_critic(client, user_prompt, design_intent, dot_with_edges):
        yield vc_event

    # Layout
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

    # Schematic validation
    sch_feedback = _validate_schematic(dot_with_edges, design_intent)
    if sch_feedback:
        yield {"type": "feedback", "source": "schematic_builder",
               "issues": [f.model_dump() for f in sch_feedback]}

    # Export
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
    model: str = "gpt-5.4",
    clarifications: Optional[dict[str, str]] = None,
    max_tool_rounds: int = 10,
    max_critic_rounds: int = 2,
    extraction_mode: str = "single_shot",
    orchestration: str = "rigid",
) -> Generator[PipelineEvent, None, None]:
    """Classify user schematic feedback and route to the cheapest pipeline tier.

    Tier 1 (edge_edit)      -> run_pipeline_patch_edges
    Tier 2 (component_swap) -> run_pipeline_reselect
    Tier 3 (architectural)  -> run_pipeline_finalize (full re-run, preserves orchestration mode)
    """
    client = create_client(model)

    yield {"type": "pipeline_phase", "phase": "feedback_classification"}
    yield {"type": "phase", "phase": "feedback_classification",
           "detail": "Classifying feedback to determine minimal re-run scope..."}

    try:
        classification = _classify_feedback(
            client, feedback_text, circuit_dsl, selection_dict, dot_string,
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
        mode_label = "unified session" if orchestration == "unified" else "full pipeline"
        yield {"type": "phase", "phase": "feedback_classification",
               "detail": f"Routing to {mode_label} re-run (Tier 3: architectural)."}
        yield from run_pipeline_finalize(
            explore_state=explore_state,
            user_prompt=user_prompt,
            model=model,
            clarifications=clarifications,
            max_tool_rounds=max_tool_rounds,
            max_critic_rounds=max_critic_rounds,
            schematic_feedback=feedback_text,
            extraction_mode=extraction_mode,
            orchestration=orchestration,
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
    routing_warnings = gds_result.get("routing_warnings", [])
    if not routing_ok:
        yield {"type": "phase", "phase": "layout_gds",
               "detail": "Optical routing failed — layout rendered without links."}
    if missing:
        yield {"type": "phase", "phase": "layout_gds",
               "detail": f"Missing SAX models: {', '.join(missing[:5])}"}

    yield {"type": "gds_rendered", "gds_fig_b64": gds_result["gds_fig_b64"],
           "routing_ok": routing_ok, "missing_models": missing,
           "routing_error": gds_result.get("routing_error", ""),
           "routing_warnings": routing_warnings}

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
    model: str = "gpt-5.4",
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
