"""
Streamlit UI for the PhIDO Interpreter Agent.

Streams the agent's tool-calling loop in real time, displays the structured
DesignIntent output, and renders a preschematic DOT graph.

Supports interactive disambiguation: after the agent explores, it may ask
clarification questions before finalizing the design.

Run with:
    streamlit run mcp_servers/streamlit_app.py
"""

import base64
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from mcp_servers.interpreter_agent import explore_and_ask, finalize_stream
from mcp_servers.pipeline_orchestrator import (
    run_pipeline_finalize,
    run_pipeline_with_feedback,
    run_layout_simulation,
)
from mcp_servers.pdk_catalog_server import get_port_names
from mcp_servers.models import DesignIntent

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PhIDO Interpreter",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("PhIDO Interpreter")
    st.caption("Photonic Intelligent Design & Optimization")
    st.markdown("---")

    model = st.selectbox(
        "Model",
        options=[
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "gemini-3.1-pro-preview",
            "gemini-3-flash-preview",
            "gpt-5.4",
            "gpt-4o",
            "gpt-4o-mini",
            "o3-mini",
            "o1",
            "claude-sonnet-4-20250514",
            "claude-3.5-sonnet-20241022",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        index=0,
        help=(
            "LLM for the agent loop. Frontier: Claude Opus/Sonnet 4.6, Gemini 3.x. "
            "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY / GEMINI_API_KEY."
        ),
    )
    max_rounds = st.slider(
        "Max tool-calling rounds",
        min_value=1, max_value=30, value=15,
        help="Safety cap on the number of agentic reasoning rounds.",
    )
    max_grounding = st.slider(
        "Max grounding rounds",
        min_value=1, max_value=10, value=5,
        help="Cap on extra rounds if the KG grounding gate triggers.",
    )
    max_critic = st.slider(
        "Max critic iterations",
        min_value=0, max_value=5, value=2,
        help="How many times the critic can reject and loop back. 0 = no critic.",
    )

    st.markdown("---")

    reasoning_strategy = st.radio(
        "Reasoning strategy",
        options=["balanced", "kg_first", "pdk_first"],
        index=0,
        format_func=lambda x: {
            "balanced": "Balanced (default)",
            "kg_first": "KG-first",
            "pdk_first": "PDK-first",
        }[x],
        help=(
            "**Balanced**: Agent freely decides when to use KG vs PDK tools.\n\n"
            "**KG-first**: Agent starts by querying the Knowledge Graph for domain "
            "understanding, then maps findings to PDK components.\n\n"
            "**PDK-first**: Agent starts by searching the PDK catalog directly, "
            "only consulting the KG when PDK results are ambiguous."
        ),
    )

    st.markdown("---")

    extraction_mode = st.radio(
        "Extraction mode",
        options=["single_shot", "iterative", "auto"],
        index=0,
        format_func=lambda x: {
            "single_shot": "Single-shot (default)",
            "iterative": "Iterative builder",
            "auto": "Auto (complexity-based)",
        }[x],
        help=(
            "**Single-shot**: Generate the full DesignIntent in one structured output call.\n\n"
            "**Iterative builder**: Build the circuit step-by-step using tool calls on a "
            "graph object. Better for complex circuits with many components.\n\n"
            "**Auto**: Automatically pick iterative for complex designs."
        ),
    )

    orchestration_strategy = st.radio(
        "Orchestration strategy",
        options=["rigid", "unified"],
        index=0,
        format_func=lambda x: {
            "rigid": "Rigid (phased pipeline)",
            "unified": "Unified (single session)",
        }[x],
        help=(
            "**Rigid**: Phased pipeline with separate LLM calls per stage. "
            "Each stage has its own context window. Best for weaker models.\n\n"
            "**Unified**: Single continuous agent session that explores and builds "
            "in one context. Reduces overhead and context fragmentation. "
            "Best for capable models (e.g., GPT-5, Claude Opus/Sonnet 4.6, Gemini 3)."
        ),
    )

    st.markdown("---")
    tools_md = (
        "**Tools available:**\n"
        "- `search_pdk` — PDK component search\n"
        "- `validate_ports` — port config check\n"
        "- `get_component_info` — full PDK details\n"
        "- `get_module_params` — GDSFactory parameters\n"
        "- `search_knowledge_graph` — KG semantic search\n"
        "- `resolve_function` — function → components\n"
        "- `get_concept_neighborhood` — KG graph traversal\n"
        "- `get_component_properties` — KG properties\n"
        "- `get_pdk_implementations` — KG concept → PDK cells\n"
        "- `get_pdk_cell_details` — full KG details for a PDK cell\n"
        "- `search_pdk_by_function` — function → PDK cells"
    )
    st.markdown(tools_md)

# ---------------------------------------------------------------------------
# Main area — prompt input
# ---------------------------------------------------------------------------
st.header("Circuit Description")

prompt = st.text_area(
    "Describe your photonic circuit in natural language:",
    height=100,
    placeholder="e.g. A 1x2 beamsplitter connected to two MZI modulators with heaters",
)

run_btn = st.button("Interpret", type="primary", disabled=not prompt.strip())

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "stage": "initial",          # initial | exploring | disambiguating | finalizing
                                 # | unified_disambiguating | error_review
                                 # | reviewing | patching | simulating | done
    "design_intent": None,
    "dot_string": None,
    "preschematic_dot": None,
    "run_log": [],
    "explore_state": None,
    "clarification_req": None,
    "user_prompt": "",
    "component_selection": None,
    "circuit_dsl": None,
    "gf_netlist_yaml": None,
    "final_dot": None,
    "pipeline_finalize": False,
    # Layout + Simulation (Phase 8)
    "gds_fig_b64": None,
    "sax_fig_b64": None,
    "s_params_json": None,
    "gds_file_path": None,
    "drc_clean": None,
    "drc_violations": None,
    "drc_report_path": None,
    "schematic_feedback": None,
    "live_circuit_dot": None,
    # Intermediate state for tiered feedback
    "footprints": None,
    "positions": None,
    "_edge_patches": None,
    # Validation failure state for error_review stage
    "validation_failed_event": None,
    # Unified orchestration mid-session disambiguation state
    "unified_questions": None,
    "unified_resume_state": None,
    "unified_user_answers": None,
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Phase labels for display — grouped by pipeline stage
PHASE_LABELS = {
    # Interpreter
    "extraction":           "Concept Extraction",
    "exploration":          "Agentic Exploration",
    "grounding":            "KG Grounding Gate",
    "disambiguation":       "Disambiguation",
    "clarification_update": "Clarification Update",
    "strategy":             "Reasoning Strategy",
    "upstream_feedback":    "Upstream Feedback",
    "structuring":          "Structured Output",
    "critic":               "Critic Review",
    "iterative_build":      "Iterative Circuit Builder",
    "mode_selection":       "Extraction Mode",
    # Topology Gate
    "topology_gate":        "Topology Validation",
    "validation_retry":     "Validation Retry",
    # Component Selection
    "component_selection":  "Component Selection",
    "compliance_check":     "Compliance Check",
    # Schematic Building
    "schematic_building":   "Schematic Building",
    "edge_routing":         "Edge Routing",
    "layout":               "Layout Computation",
    "export":               "Export",
    # Tiered Feedback
    "feedback_classification": "Feedback Classification",
    "edge_patching":           "Edge Patching",
    "component_reselection":   "Component Re-selection",
    # Layout + Simulation
    "layout_gds":           "GDS Layout Generation",
    "simulation":           "Circuit Simulation (SAX)",
    "gds_export":           "GDS File Export",
    # Visual Critic
    "visual_critic":        "Visual Schematic Review",
    # Unified orchestration
    "unified_session":      "Unified Session",
    "drc":                  "Design Rule Check (DRC)",
    # Generic fallback
    "pipeline_phase":       "Pipeline",
}


# ---------------------------------------------------------------------------
# Helper: render a stream of agent events inside a status widget
# ---------------------------------------------------------------------------
def _render_events(event_stream, status_widget, graph_placeholder=None):
    """Consume an event generator and render each event in the Streamlit status widget.

    If ``graph_placeholder`` is an ``st.empty()`` container, ``circuit_updated``
    events will live-update a Graphviz chart there.
    """
    for event in event_stream:
        etype = event["type"]
        st.session_state.run_log.append(event)

        if etype == "circuit_updated":
            dot = event.get("dot", "")
            st.session_state["live_circuit_dot"] = dot
            n_comp = event.get("component_count", 0)
            n_conn = event.get("connection_count", 0)
            st.write(f"Circuit: {n_comp} component(s), {n_conn} connection(s)")
            if graph_placeholder and dot:
                try:
                    graph_placeholder.graphviz_chart(dot, use_container_width=True)
                except Exception:
                    pass
            continue

        if etype == "phase":
            label = PHASE_LABELS.get(event["phase"], event["phase"])
            st.write(f"**{label}:** {event['detail']}")

        elif etype == "concepts":
            components = event["components"]
            parameters = event["parameters"]
            specs = event["specs"]
            if components or parameters or specs:
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Components / Architectures**")
                    st.markdown(
                        " ".join(f"`{c}`" for c in components) if components
                        else "_none detected_"
                    )
                with c2:
                    st.markdown("**Parameters / Properties**")
                    st.markdown(
                        " ".join(f"`{p}`" for p in parameters) if parameters
                        else "_none detected_"
                    )
                with c3:
                    st.markdown("**Specs / Measurements**")
                    st.markdown(
                        " ".join(f"`{s}`" for s in specs) if specs
                        else "_none detected_"
                    )

        elif etype == "tool_call":
            args_str = json.dumps(event["args"], ensure_ascii=False)
            if len(args_str) > 120:
                args_str = args_str[:120] + "..."
            st.write(f"🔧 `{event['name']}` → {args_str}")

        elif etype == "tool_result":
            preview = event["result"][:300]
            if len(event["result"]) > 300:
                preview += "..."
            with st.expander(f"Result from `{event['name']}`", expanded=False):
                st.code(preview, language="json")

        elif etype == "agent_text":
            text = event["content"].strip()
            if text:
                st.markdown(text)

        elif etype == "grounding":
            concepts = event["ungrounded"]
            st.warning(f"⚠️ **Grounding gate** — {event['detail']}")
            n_cols = 4
            cols = st.columns(n_cols)
            for i, concept in enumerate(concepts):
                cols[i % n_cols].markdown(f"`{concept}`")

        elif etype == "stats":
            st.write(
                f"**Tool call summary:** {event['total']} total "
                f"({event['pdk']} PDK, {event['kg']} KG)"
            )

        elif etype == "clarification":
            st.session_state.explore_state = event
            st.session_state.clarification_req = event["request"]

        elif etype == "critic":
            verdict = event["verdict"]
            attempt_num = event.get("attempt", "?")
            if verdict["passed"]:
                st.success(f"**Critic PASSED** (attempt {attempt_num}): {verdict['summary']}")
            else:
                st.error(f"**Critic FAILED** (attempt {attempt_num}): {verdict['summary']}")
                issues = verdict.get("issues", [])
                if issues:
                    with st.expander(f"{len(issues)} issue(s) found", expanded=True):
                        for iss in issues:
                            sev_icon = "🔴" if iss.get("severity") == "major" else "🟡"
                            affected = ", ".join(iss.get("affected_components", [])) or "global"
                            st.markdown(
                                f"{sev_icon} **[{iss.get('severity', '?').upper()}]** "
                                f"{iss['description']}  \n"
                                f"_Affects: {affected}_"
                            )

        elif etype == "done":
            di: DesignIntent = event["result"]
            st.session_state.design_intent = di
            st.session_state.dot_string = di.to_dot()
            st.session_state.preschematic_dot = st.session_state.dot_string
            if st.session_state.get("pipeline_finalize"):
                status_widget.update(
                    label="Interpreter done — continuing pipeline...",
                    state="running",
                    expanded=True,
                )
            else:
                st.session_state.stage = "done"
                status_widget.update(
                    label=f"Done — {len(di.components)} components, "
                          f"{len(di.connections)} connections",
                    state="complete",
                    expanded=False,
                )

        elif etype == "pipeline_phase":
            phase = event.get("phase", "?")
            label = PHASE_LABELS.get(phase, phase)
            st.write(f"**{label}**")

        elif etype == "selection_done":
            st.session_state.component_selection = event.get("selection")
            n = len(event.get("selection", {}).get("mappings", []))
            st.write(f"**Component selection:** {n} mapping(s)")

        elif etype == "dot_draft":
            st.write("**DOT draft** (nodes with ports) generated.")

        elif etype == "edge_routing_done":
            st.write("**Edges routed** (port-level).")

        elif etype == "visual_critic_verdict":
            verdict = event.get("verdict", {})
            if verdict.get("passed"):
                st.write("**Visual critic:** Passed")
            else:
                st.warning(
                    f"**Visual critic: Issues found** — {verdict.get('summary', '')}"
                )
                issues = verdict.get("issues", [])
                if issues:
                    with st.expander(f"{len(issues)} visual issue(s)", expanded=False):
                        for iss in issues:
                            sev = iss.get("severity", "?")
                            desc = iss.get("description", "")
                            st.markdown(f"- **[{sev}]** {desc}")

        elif etype == "layout_done":
            n = len(event.get("positions", {}))
            st.write(f"**Layout computed** for {n} node(s).")

        elif etype == "feedback":
            issues = event.get("issues", [])
            st.warning(f"**Validation feedback:** {len(issues)} issue(s)")
            for iss in issues:
                affected = ", ".join(iss.get("affected_components", [])) or "global"
                st.markdown(f"  - [{iss.get('severity', '?')}] {iss.get('description', '')} — _Affects: {affected}_")

        elif etype == "pipeline_done":
            result = event.get("result", {})
            st.session_state.component_selection = result.get("selection")
            st.session_state.circuit_dsl = result.get("circuit_dsl")
            st.session_state.gf_netlist_yaml = result.get("gf_netlist_yaml")
            st.session_state.final_dot = result.get("dot_string")
            st.session_state.footprints = result.get("footprints")
            st.session_state.positions = result.get("positions")
            if result.get("dot_string"):
                st.session_state.dot_string = result["dot_string"]
            di_raw = result.get("design_intent")
            if di_raw and st.session_state.design_intent is None:
                try:
                    st.session_state.design_intent = (
                        DesignIntent(**di_raw) if isinstance(di_raw, dict) else di_raw
                    )
                    st.session_state.preschematic_dot = (
                        st.session_state.design_intent.to_dot()
                    )
                except Exception:
                    pass
            st.session_state.stage = "reviewing"
            st.session_state.pipeline_finalize = False
            status_widget.update(
                label="Schematic ready for review",
                state="complete",
                expanded=False,
            )

        elif etype == "gds_rendered":
            routing = "with" if event.get("routing_ok") else "without"
            st.write(f"**GDS layout rendered** ({routing} optical routing).")
            for warn_msg in event.get("routing_warnings", []):
                if event.get("routing_ok"):
                    st.info(warn_msg)
                else:
                    st.warning(warn_msg)

        elif etype == "layout_sim_done":
            result = event.get("result", {})
            st.session_state.gds_fig_b64 = result.get("gds_fig_b64")
            st.session_state.sax_fig_b64 = result.get("sax_fig_b64")
            st.session_state.s_params_json = result.get("s_params")
            st.session_state.gds_file_path = result.get("gds_file_path")
            st.session_state.drc_clean = result.get("drc_clean")
            st.session_state.drc_violations = result.get("drc_violations")
            st.session_state.drc_report_path = result.get("drc_report_path")
            drc_clean = result.get("drc_clean")
            drc_error = result.get("drc_error")
            if drc_clean is True:
                st.success("**DRC clean** — no design-rule violations.")
            elif drc_clean is False:
                st.warning(
                    f"**DRC found {result.get('drc_violations')} violation(s)** — "
                    "see the report database for details."
                )
            elif drc_error:
                st.info(f"DRC not conclusive: {drc_error}")
            st.session_state.stage = "done"
            status_widget.update(
                label="Layout and simulation complete",
                state="complete",
                expanded=False,
            )

        elif etype == "user_question":
            st.session_state.unified_questions = event["questions"]
            st.session_state.unified_resume_state = event["resume_state"]
            st.session_state.stage = "unified_disambiguating"
            st.session_state.pipeline_finalize = False
            status_widget.update(
                label="Waiting for user input...",
                state="complete",
                expanded=False,
            )
            st.rerun()

        elif etype == "validation_retry":
            st.info(event.get("detail", "Retrying interpreter with validation feedback..."))

        elif etype == "validation_failed":
            st.session_state.validation_failed_event = event
            st.session_state.stage = "error_review"
            st.session_state.pipeline_finalize = False
            status_widget.update(
                label="Validation failed — review needed",
                state="error",
                expanded=False,
            )
            st.rerun()

        elif etype == "error":
            st.error(event["message"])
            if st.session_state.get("pipeline_finalize"):
                st.session_state.validation_failed_event = {
                    "gate": "pipeline",
                    "issues": [{"severity": "fundamental",
                                "description": event["message"]}],
                    "attempts": 0,
                    "message": event["message"],
                }
                st.session_state.stage = "error_review"
                st.session_state.pipeline_finalize = False
                status_widget.update(label="Failed — review needed", state="error")
                st.rerun()
            else:
                status_widget.update(label="Failed", state="error")


# ---------------------------------------------------------------------------
# Stage: INITIAL → start exploration
# ---------------------------------------------------------------------------
if run_btn and prompt.strip():
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default
    st.session_state.user_prompt = prompt.strip()
    if orchestration_strategy == "unified":
        st.session_state.stage = "finalizing"
    else:
        st.session_state.stage = "exploring"

# ---------------------------------------------------------------------------
# Stage: EXPLORING — run Phases 0 → 1 → 1.5 → 1.75
# ---------------------------------------------------------------------------
if st.session_state.stage == "exploring":
    st.markdown("---")
    with st.status("Exploring design space...", expanded=True) as status_widget:
        _render_events(
            explore_and_ask(
                user_prompt=st.session_state.user_prompt,
                model=model,
                max_tool_rounds=max_rounds,
                max_grounding_rounds=max_grounding,
                reasoning_strategy=reasoning_strategy,
            ),
            status_widget,
        )
        status_widget.update(label="Exploration complete", state="complete", expanded=False)

    if st.session_state.explore_state is not None:
        st.session_state.stage = "disambiguating"
        st.rerun()

# ---------------------------------------------------------------------------
# Stage: DISAMBIGUATING — show clarification form (or auto-proceed)
# ---------------------------------------------------------------------------
if st.session_state.stage == "disambiguating":
    st.markdown("---")

    clar_req = st.session_state.clarification_req
    questions = clar_req.get("questions", []) if clar_req else []
    ready = clar_req.get("ready_to_proceed", True) if clar_req else True

    has_questions = len(questions) > 0

    if has_questions:
        critical = [q for q in questions if q.get("priority") == "critical"]
        helpful = [q for q in questions if q.get("priority") != "critical"]

        st.subheader("Agent has questions before finalizing")
        if ready:
            st.info(
                "The agent can proceed with its defaults, but your answers will improve accuracy."
            )
        else:
            st.warning(
                "The agent needs answers to critical questions before it can produce a reliable design."
            )

        with st.form("disambiguation_form"):
            answers = {}

            if critical:
                st.markdown("#### Critical — design blocked without an answer")
                for i, q in enumerate(critical):
                    st.markdown(f"**{q['question']}**")
                    st.caption(f"Context: {q['context']}")
                    if q.get("options"):
                        options_list = ["(use agent default)"] + q["options"]
                        choice = st.selectbox(
                            "Your answer",
                            options=options_list,
                            key=f"crit_{i}",
                        )
                        custom = st.text_input(
                            "Or enter a custom answer (overrides dropdown if non-empty)",
                            key=f"crit_{i}_custom",
                        )
                        if custom.strip():
                            answers[q["question"]] = custom.strip()
                        elif choice != "(use agent default)":
                            answers[q["question"]] = choice
                    else:
                        ans = st.text_input(
                            "Your answer",
                            placeholder=f"Default: {q.get('default', 'N/A')}",
                            key=f"crit_{i}",
                        )
                        if ans.strip():
                            answers[q["question"]] = ans.strip()
                    st.markdown("---")

            if helpful:
                st.markdown("#### Helpful — improves accuracy (optional)")
                for i, q in enumerate(helpful):
                    st.markdown(f"**{q['question']}**")
                    st.caption(f"Context: {q['context']}")
                    if q.get("options"):
                        options_list = ["(use agent default)"] + q["options"]
                        choice = st.selectbox(
                            "Your answer",
                            options=options_list,
                            key=f"help_{i}",
                        )
                        custom = st.text_input(
                            "Or enter a custom answer (overrides dropdown if non-empty)",
                            key=f"help_{i}_custom",
                        )
                        if custom.strip():
                            answers[q["question"]] = custom.strip()
                        elif choice != "(use agent default)":
                            answers[q["question"]] = choice
                    else:
                        ans = st.text_input(
                            "Your answer",
                            placeholder=f"Default: {q.get('default', 'N/A')}",
                            key=f"help_{i}",
                        )
                        if ans.strip():
                            answers[q["question"]] = ans.strip()
                    st.markdown("---")

            col_submit, col_skip = st.columns(2)
            with col_submit:
                submitted = st.form_submit_button("Submit answers & finalize", type="primary")
            with col_skip:
                skipped = st.form_submit_button("Skip — proceed with defaults")

            if submitted or skipped:
                st.session_state.clarifications = answers if (submitted and answers) else None
                st.session_state.stage = "finalizing"
                st.rerun()
    else:
        st.session_state.clarifications = None
        st.session_state.stage = "finalizing"
        st.rerun()

# ---------------------------------------------------------------------------
# Stage: UNIFIED_DISAMBIGUATING — mid-session clarification from ask_user tool
# ---------------------------------------------------------------------------
if st.session_state.stage == "unified_disambiguating":
    st.markdown("---")
    st.header("Agent has a question")
    st.info(
        "The agent encountered ambiguity during design exploration and needs "
        "your input before continuing. Answer below, or skip to let the agent "
        "use its defaults."
    )

    questions = st.session_state.get("unified_questions") or []

    with st.form("unified_disambiguation_form"):
        answers: dict[str, str] = {}

        for i, q in enumerate(questions):
            st.markdown(f"**{q.get('question', '?')}**")
            context = q.get("context", "")
            if context:
                st.caption(f"Context: {context}")

            options = q.get("options") or []
            default = q.get("default", "")

            if options:
                options_list = [f"(use default: {default})"] + options
                choice = st.selectbox(
                    "Your answer",
                    options=options_list,
                    key=f"uq_{i}",
                )
                custom = st.text_input(
                    "Or enter a custom answer (overrides dropdown if non-empty)",
                    key=f"uq_{i}_custom",
                )
                if custom.strip():
                    answers[q["question"]] = custom.strip()
                elif choice != options_list[0]:
                    answers[q["question"]] = choice
            else:
                ans = st.text_input(
                    "Your answer",
                    placeholder=f"Default: {default}",
                    key=f"uq_{i}",
                )
                if ans.strip():
                    answers[q["question"]] = ans.strip()

            if i < len(questions) - 1:
                st.markdown("---")

        col_submit, col_skip = st.columns(2)
        with col_submit:
            submitted = st.form_submit_button("Submit answers & continue", type="primary")
        with col_skip:
            skipped = st.form_submit_button("Skip — use agent defaults")

        if submitted or skipped:
            if skipped or not answers:
                final_answers = {
                    q.get("question", ""): q.get("default", "")
                    for q in questions
                }
            else:
                final_answers = {
                    q.get("question", ""): answers.get(
                        q.get("question", ""), q.get("default", "")
                    )
                    for q in questions
                }
            st.session_state.unified_user_answers = final_answers
            st.session_state.stage = "finalizing"
            st.rerun()

# ---------------------------------------------------------------------------
# Stage: FINALIZING — run Phases 2 → 7 (interpreter + component selection + schematic)
# ---------------------------------------------------------------------------
if st.session_state.stage == "finalizing":
    st.markdown("---")

    explore_state = st.session_state.explore_state
    clarifications = st.session_state.get("clarifications")

    # Detect whether we are resuming a paused unified session
    resume_state = st.session_state.get("unified_resume_state")
    user_answers = st.session_state.get("unified_user_answers")
    if resume_state and user_answers is not None:
        resume_state["user_answers"] = user_answers
        label = "Unified session — resuming with your answers..."
    elif orchestration_strategy == "unified":
        resume_state = None
        label = "Unified session — exploring & building..."
    else:
        resume_state = None
        label = "Finalizing design"
        if clarifications:
            label += f" (with {len(clarifications)} clarification(s))"
        if extraction_mode == "iterative":
            label += " [iterative builder]"
        label += "..."

    feedback_for_retry = st.session_state.get("schematic_feedback")

    # Live circuit visualization placeholder (updated by circuit_updated events)
    live_graph_placeholder = st.empty()

    st.session_state.pipeline_finalize = True
    with st.status(label, expanded=True) as status_widget:
        _render_events(
            run_pipeline_finalize(
                explore_state=explore_state,
                user_prompt=st.session_state.user_prompt,
                model=model,
                clarifications=clarifications,
                max_tool_rounds=max_rounds,
                max_critic_rounds=max_critic,
                schematic_feedback=feedback_for_retry,
                extraction_mode=extraction_mode,
                orchestration=orchestration_strategy,
                unified_resume_state=resume_state,
            ),
            status_widget,
            graph_placeholder=live_graph_placeholder,
        )

    # Clear unified disambiguation state after pipeline completes or pauses
    st.session_state.unified_resume_state = None
    st.session_state.unified_user_answers = None
    st.session_state.unified_questions = None
    st.session_state.schematic_feedback = None

    if st.session_state.stage in ("reviewing", "error_review", "unified_disambiguating"):
        st.rerun()

# ---------------------------------------------------------------------------
# Stage: ERROR_REVIEW — validation failed after retries; let user help
# ---------------------------------------------------------------------------
if st.session_state.stage == "error_review":
    st.markdown("---")
    st.header("Validation Failed")

    vf = st.session_state.get("validation_failed_event", {})
    gate = vf.get("gate", "unknown gate")
    attempts = vf.get("attempts", 0)
    issues = vf.get("issues", [])

    st.error(
        f"The **{gate.replace('_', ' ')}** found fundamental issues that "
        f"could not be resolved after **{attempts}** attempt(s)."
    )

    if issues:
        st.subheader("Issues")
        for iss in issues:
            severity = iss.get("severity", "?")
            desc = iss.get("description", "")
            affected = ", ".join(iss.get("affected_components", [])) or "global"
            suggestion = iss.get("suggested_action", "")
            st.markdown(f"- **[{severity}]** {desc} — _Affects: {affected}_")
            if suggestion:
                st.markdown(f"  - Suggestion: {suggestion}")

    st.subheader("What would you like to do?")

    with st.form("error_review_form"):
        guidance = st.text_area(
            "Provide guidance to help the interpreter fix the issues "
            "(or leave blank to start over with a new prompt):",
            height=100,
            placeholder="e.g. 'The splitter should use a directional coupler, "
                        "make sure each arm has a phase shifter'",
        )

        col_retry, col_restart = st.columns(2)
        with col_retry:
            retry_btn = st.form_submit_button(
                "Retry with guidance", type="primary",
                disabled=False,
            )
        with col_restart:
            restart_btn = st.form_submit_button("Start over")

        if retry_btn and guidance.strip():
            user_guidance = (
                "USER GUIDANCE FOR VALIDATION FAILURE\n"
                "====================================\n"
                f"The {gate.replace('_', ' ')} failed with these issues:\n"
                + "\n".join(f"  - {iss.get('description', '')}" for iss in issues)
                + "\n\nThe user provided this guidance:\n"
                f"{guidance}\n\n"
                "You MUST address this feedback. Re-investigate with your tools "
                "and revise the DesignIntent accordingly."
            )
            st.session_state.schematic_feedback = user_guidance
            st.session_state.validation_failed_event = None
            st.session_state.stage = "finalizing"
            st.rerun()
        elif restart_btn:
            for key, default in _DEFAULTS.items():
                st.session_state[key] = default
            st.rerun()

# ---------------------------------------------------------------------------
# Stage: REVIEWING — user approves schematic or requests changes
# ---------------------------------------------------------------------------
if st.session_state.stage == "reviewing":
    st.markdown("---")
    st.header("Schematic Review")
    st.info("Review the generated schematic below. You can approve it to proceed "
            "to GDS layout and simulation, or request changes. "
            "Minor edits (e.g. rewiring a connection) will be applied surgically "
            "without re-running the full pipeline.")

    schematic_dot = st.session_state.get("final_dot")
    if schematic_dot:
        try:
            st.graphviz_chart(schematic_dot, use_container_width=True)
        except Exception as e:
            st.error(f"Graphviz rendering failed: {e}")
            st.code(schematic_dot, language="dot")

    sel = st.session_state.get("component_selection")
    if sel:
        mappings = sel.get("mappings", [])
        if mappings:
            st.subheader("Component Mappings")
            map_rows = [
                {
                    "Component ID": m.get("component_id", ""),
                    "PDK Module": m.get("pdk_module", ""),
                    "Match Quality": m.get("match_quality", ""),
                    "Port Config": m.get("port_config", ""),
                }
                for m in mappings
            ]
            st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)

    gf_yaml = st.session_state.get("gf_netlist_yaml")
    if gf_yaml:
        with st.expander("GDSFactory Netlist (YAML)", expanded=False):
            st.code(gf_yaml, language="yaml")

    st.markdown("---")

    if st.button("Approve and run Layout + Simulation", type="primary"):
        st.session_state.schematic_feedback = None
        st.session_state.stage = "simulating"
        st.rerun()

    # -- Structured edge-edit form --
    _sel = st.session_state.get("component_selection") or {}
    _mappings = _sel.get("mappings", [])
    _node_ids = [m.get("component_id", "") for m in _mappings]
    _port_map: dict[str, list[str]] = {}
    for m in _mappings:
        _port_map[m.get("component_id", "")] = get_port_names(m.get("pdk_module", ""))

    if _node_ids:
        with st.expander("Quick edge edit", expanded=False):
            st.caption("Add or remove a single port-to-port connection without "
                       "re-running the full pipeline.")
            edge_col1, edge_col2 = st.columns(2)
            with edge_col1:
                edge_action = st.selectbox(
                    "Action", ["add", "remove"], key="edge_action",
                )
                src_node = st.selectbox(
                    "Source node", _node_ids, key="edge_src_node",
                )
                src_ports = _port_map.get(src_node, [])
                src_port = st.selectbox(
                    "Source port", src_ports if src_ports else ["(no ports)"],
                    key="edge_src_port",
                )
            with edge_col2:
                st.markdown("")  # spacer
                st.markdown("")
                tgt_node = st.selectbox(
                    "Target node", _node_ids, key="edge_tgt_node",
                )
                tgt_ports = _port_map.get(tgt_node, [])
                tgt_port = st.selectbox(
                    "Target port", tgt_ports if tgt_ports else ["(no ports)"],
                    key="edge_tgt_port",
                )

            if st.button("Apply edge edit"):
                if src_port == "(no ports)" or tgt_port == "(no ports)":
                    st.warning("Port information not available for selected component.")
                else:
                    from mcp_servers.models import EdgePatch
                    patch = EdgePatch(
                        action=edge_action,
                        src_node=src_node,
                        src_port=src_port,
                        tgt_node=tgt_node,
                        tgt_port=tgt_port,
                    )
                    st.session_state.schematic_feedback = None
                    st.session_state._edge_patches = [patch.model_dump()]
                    st.session_state.stage = "patching"
                    st.rerun()

    # -- Free-text feedback (auto-classified into tiers) --
    st.markdown("**Or describe changes in natural language:**")
    feedback_text = st.text_area(
        "Request changes (describe what to fix):",
        height=120,
        key="review_feedback_text",
    )
    if st.button("Submit feedback"):
        if feedback_text.strip():
            st.session_state.schematic_feedback = feedback_text.strip()
            st.session_state._edge_patches = None
            st.session_state.stage = "patching"
            st.rerun()
        else:
            st.warning("Please enter feedback before submitting.")

# ---------------------------------------------------------------------------
# Stage: PATCHING — tiered feedback (edge patch / component swap / full re-run)
# ---------------------------------------------------------------------------
if st.session_state.stage == "patching":
    st.markdown("---")

    edge_patches_raw = st.session_state.get("_edge_patches")
    feedback_text = st.session_state.get("schematic_feedback")

    if edge_patches_raw:
        from mcp_servers.models import EdgePatch
        patches = [EdgePatch(**p) for p in edge_patches_raw]
        label = f"Applying {len(patches)} edge edit(s)..."
        with st.status(label, expanded=True) as status_widget:
            from mcp_servers.pipeline_orchestrator import run_pipeline_patch_edges
            _render_events(
                run_pipeline_patch_edges(
                    circuit_dsl=st.session_state.circuit_dsl,
                    selection_dict=st.session_state.component_selection,
                    design_intent_dict=(
                        st.session_state.design_intent.model_dump()
                        if st.session_state.design_intent
                        and hasattr(st.session_state.design_intent, "model_dump")
                        else (st.session_state.design_intent or {})
                    ),
                    dot_string=st.session_state.final_dot,
                    footprints=st.session_state.footprints or {},
                    edge_patches=patches,
                ),
                status_widget,
            )
        st.session_state._edge_patches = None
    elif feedback_text:
        label = "Classifying feedback and applying changes..."
        di = st.session_state.design_intent
        di_dict = (
            di.model_dump() if di and hasattr(di, "model_dump")
            else (di or {})
        )
        with st.status(label, expanded=True) as status_widget:
            _render_events(
                run_pipeline_with_feedback(
                    feedback_text=feedback_text,
                    circuit_dsl=st.session_state.circuit_dsl,
                    selection_dict=st.session_state.component_selection,
                    design_intent_dict=di_dict,
                    dot_string=st.session_state.final_dot or "",
                    footprints=st.session_state.footprints or {},
                    explore_state=st.session_state.explore_state,
                    user_prompt=st.session_state.user_prompt,
                    model=model,
                    clarifications=st.session_state.get("clarifications"),
                    max_tool_rounds=max_rounds,
                    max_critic_rounds=max_critic,
                    extraction_mode=extraction_mode,
                    orchestration=orchestration_strategy,
                ),
                status_widget,
            )
        st.session_state.schematic_feedback = None
    else:
        st.warning("No feedback to process.")
        st.session_state.stage = "reviewing"

    if st.session_state.stage == "reviewing":
        st.rerun()

# ---------------------------------------------------------------------------
# Stage: SIMULATING — run Phase 8 (GDS layout + SAX simulation)
# ---------------------------------------------------------------------------
if st.session_state.stage == "simulating":
    st.markdown("---")
    gf_yaml = st.session_state.get("gf_netlist_yaml")
    if not gf_yaml:
        st.error("No GDSFactory netlist available for simulation.")
        st.session_state.stage = "done"
    else:
        with st.status("Running layout and simulation...", expanded=True) as status_widget:
            _render_events(
                run_layout_simulation(gf_netlist_yaml=gf_yaml),
                status_widget,
            )

    if st.session_state.stage == "done":
        st.rerun()

# ---------------------------------------------------------------------------
# Display results (stage == "done")
# ---------------------------------------------------------------------------
di: DesignIntent | None = st.session_state.design_intent

if di is not None:
    st.markdown("---")
    st.header("Design Intent")

    st.subheader(di.title)
    st.markdown(di.brief_summary)

    col_table, col_graph = st.columns([3, 2])

    with col_table:
        st.subheader("Components")
        rows = []
        for c in di.components:
            specs_str = ", ".join(f"{s.key}={s.value}" for s in c.specs) if c.specs else ""
            rows.append({
                "ID": c.id,
                "Description": c.description,
                "Role": c.role or "",
                "Ports": c.port_config or "",
                "Specs": specs_str,
                "Confidence": f"{c.confidence:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with col_graph:
        st.subheader("Preschematic")
        preschematic = st.session_state.preschematic_dot or st.session_state.dot_string
        if preschematic:
            try:
                st.graphviz_chart(preschematic, use_container_width=True)
            except Exception as e:
                st.error(f"Graphviz rendering failed: {e}")
                st.code(preschematic, language="dot")
        else:
            st.caption("No preschematic available.")

    if di.connections:
        st.subheader("Connections")
        conn_rows = []
        for cn in di.connections:
            conn_rows.append({
                "From": cn.from_component,
                "To": cn.to_component,
                "Description": cn.description,
                "Confidence": f"{cn.confidence:.1f}",
            })
        st.dataframe(pd.DataFrame(conn_rows), use_container_width=True, hide_index=True)

    if di.ambiguities:
        st.subheader("Remaining Ambiguities")
        for amb in di.ambiguities:
            st.markdown(f"- {amb}")

    selection = st.session_state.get("component_selection")
    if selection is not None:
        st.markdown("---")
        st.subheader("Component Mappings")
        mappings = selection.get("mappings", [])
        if mappings:
            map_rows = [
                {
                    "Component ID": m.get("component_id", ""),
                    "PDK Module": m.get("pdk_module", ""),
                    "Match Quality": m.get("match_quality", ""),
                    "Port Config": m.get("port_config", ""),
                }
                for m in mappings
            ]
            st.dataframe(pd.DataFrame(map_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No mappings.")

        st.subheader("Circuit Schematic")
        schematic_dot = st.session_state.get("final_dot")
        if schematic_dot:
            try:
                st.graphviz_chart(schematic_dot, use_container_width=True)
            except Exception as e:
                st.error(f"Graphviz rendering failed: {e}")
                st.code(schematic_dot, language="dot")
            st.download_button(
                label="Download circuit schematic DOT",
                data=schematic_dot,
                file_name="circuit_schematic.dot",
                mime="text/plain",
                key="dl_final_dot",
            )
        else:
            st.caption("No schematic DOT available.")

        gf_yaml = st.session_state.get("gf_netlist_yaml")
        if gf_yaml:
            st.subheader("GDSFactory Netlist")
            with st.expander("YAML netlist", expanded=False):
                st.code(gf_yaml, language="yaml")
            st.download_button(
                label="Download GDSFactory netlist (YAML)",
                data=gf_yaml,
                file_name="gf_netlist.yaml",
                mime="text/yaml",
                key="dl_gf_netlist",
            )

    # -- GDS Layout + SAX Simulation results (Phase 8) -------------------------
    gds_b64 = st.session_state.get("gds_fig_b64")
    sax_b64 = st.session_state.get("sax_fig_b64")

    if gds_b64 or sax_b64:
        st.markdown("---")
        st.subheader("Layout and Simulation")

        if gds_b64:
            st.markdown("**GDS Layout**")
            st.image(base64.b64decode(gds_b64), use_container_width=True)

        if sax_b64:
            st.markdown("**S-Parameter Simulation**")
            st.image(base64.b64decode(sax_b64), use_container_width=True)

        dl_col1, dl_col2, dl_col3 = st.columns(3)
        gds_path = st.session_state.get("gds_file_path")
        if gds_path:
            try:
                with open(gds_path, "rb") as f:
                    gds_bytes = f.read()
                with dl_col1:
                    st.download_button(
                        label="Download GDS file",
                        data=gds_bytes,
                        file_name="circuit_output.gds",
                        mime="application/octet-stream",
                        key="dl_gds_file",
                    )
            except FileNotFoundError:
                with dl_col1:
                    st.caption("GDS file not found on disk.")

        s_params = st.session_state.get("s_params_json")
        if s_params:
            with dl_col2:
                st.download_button(
                    label="Download S-parameters (JSON)",
                    data=json.dumps(s_params, indent=2),
                    file_name="s_parameters.json",
                    mime="application/json",
                    key="dl_s_params",
                )

    st.markdown("---")

    col_json, col_dot = st.columns(2)

    with col_json:
        with st.expander("Full DesignIntent JSON", expanded=False):
            full_json = json.dumps(di.full(), indent=2, ensure_ascii=False)
            st.code(full_json, language="json")

        st.download_button(
            label="Download JSON",
            data=json.dumps(
                {"summary": di.summary(), "full": di.full(), "legacy_pretemplate": di.to_pretemplate()},
                indent=2, ensure_ascii=False,
            ),
            file_name="design_intent.json",
            mime="application/json",
        )

    with col_dot:
        pre_dot = st.session_state.preschematic_dot or st.session_state.dot_string or ""
        with st.expander("DOT Source (Preschematic)", expanded=False):
            st.code(pre_dot, language="dot")

        st.download_button(
            label="Download Preschematic DOT",
            data=pre_dot,
            file_name="preschematic.dot",
            mime="text/plain",
        )

    with st.expander("Agent Trace Log", expanded=False):
        for i, event in enumerate(st.session_state.run_log):
            etype = event["type"]
            if etype == "phase":
                lbl = PHASE_LABELS.get(event["phase"], event["phase"])
                st.markdown(f"**[{i}] {lbl}** — {event['detail']}")
            elif etype == "concepts":
                st.markdown(f"[{i}] Concepts: {event['components']} | "
                            f"Params: {event['parameters']} | Specs: {event['specs']}")
            elif etype == "tool_call":
                st.markdown(f"[{i}] 🔧 `{event['name']}({json.dumps(event['args'])})`")
            elif etype == "tool_result":
                preview = event["result"][:200] + "..." if len(event["result"]) > 200 else event["result"]
                st.markdown(f"[{i}] → `{preview}`")
            elif etype == "agent_text":
                st.markdown(f"[{i}] Agent: {event['content'][:200]}")
            elif etype == "grounding":
                st.markdown(f"[{i}] ⚠️ Grounding: {event['detail']}")
            elif etype == "stats":
                st.markdown(f"[{i}] Stats: {event['total']} calls ({event['pdk']} PDK, {event['kg']} KG)")
            elif etype == "clarification":
                req = event["request"]
                n_q = len(req.get("questions", []))
                st.markdown(f"[{i}] 🤔 Disambiguation: {n_q} question(s), "
                            f"ready_to_proceed={req.get('ready_to_proceed', True)}")
            elif etype == "critic":
                v = event["verdict"]
                status = "PASSED" if v["passed"] else "FAILED"
                st.markdown(f"[{i}] 🧐 Critic {status} (attempt {event.get('attempt', '?')}): "
                            f"{v['summary'][:150]}")
            elif etype == "done":
                st.markdown(f"[{i}] ✅ Done")
            elif etype == "pipeline_phase":
                lbl = PHASE_LABELS.get(event.get("phase", "?"), event.get("phase", "?"))
                st.markdown(f"[{i}] **{lbl}**")
            elif etype == "selection_done":
                n = len(event.get("selection", {}).get("mappings", []))
                st.markdown(f"[{i}] Component selection: {n} mapping(s)")
            elif etype == "dot_draft":
                st.markdown(f"[{i}] DOT draft generated")
            elif etype == "edge_routing_done":
                st.markdown(f"[{i}] Edge routing done")
            elif etype == "layout_done":
                n = len(event.get("positions", {}))
                st.markdown(f"[{i}] Layout done ({n} positions)")
            elif etype == "feedback":
                issues = event.get("issues", [])
                st.markdown(f"[{i}] Feedback: {len(issues)} issue(s)")
            elif etype == "pipeline_done":
                st.markdown(f"[{i}] Pipeline complete")
            elif etype == "gds_rendered":
                routing = "with" if event.get("routing_ok") else "without"
                st.markdown(f"[{i}] GDS rendered ({routing} routing)")
            elif etype == "layout_sim_done":
                _res = event.get("result", {})
                _drc = _res.get("drc_clean")
                if _drc is True:
                    _drc_txt = " — DRC clean"
                elif _drc is False:
                    _drc_txt = f" — DRC {_res.get('drc_violations')} violation(s)"
                else:
                    _drc_txt = ""
                st.markdown(f"[{i}] Layout and simulation complete{_drc_txt}")
            elif etype == "circuit_updated":
                n_c = event.get("component_count", 0)
                n_e = event.get("connection_count", 0)
                st.markdown(f"[{i}] Circuit updated: {n_c} components, {n_e} connections")
            elif etype == "validation_retry":
                st.markdown(f"[{i}] 🔄 {event.get('detail', 'Validation retry')}")
            elif etype == "validation_failed":
                st.markdown(f"[{i}] ❌ Validation failed: {event.get('message', '')}")
            elif etype == "error":
                st.markdown(f"[{i}] ❌ {event['message']}")
