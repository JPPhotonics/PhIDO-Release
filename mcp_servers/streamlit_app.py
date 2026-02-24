"""
Streamlit UI for the PhIDO Interpreter Agent.

Streams the agent's tool-calling loop in real time, displays the structured
DesignIntent output, and renders a preschematic DOT graph.

Supports interactive disambiguation: after the agent explores, it may ask
clarification questions before finalizing the design.

Run with:
    streamlit run mcp_servers/streamlit_app.py
"""

import json
import pandas as pd
import streamlit as st

from mcp_servers.interpreter_agent import explore_and_ask, finalize_stream
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
        options=["o3-mini", "gpt-4o", "gpt-4o-mini", "o1", "o1-mini"],
        index=0,
        help="OpenAI model for the agent loop. Must support tool calling + structured output.",
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
    st.markdown(
        "**Tools available:**\n"
        "- `search_pdk` — PDK component search\n"
        "- `validate_ports` — port config check\n"
        "- `get_component_info` — full PDK details\n"
        "- `search_knowledge_graph` — KG semantic search\n"
        "- `resolve_function` — function → components\n"
        "- `get_concept_neighborhood` — KG graph traversal\n"
        "- `get_component_properties` — KG properties"
    )

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
    "stage": "initial",          # initial | exploring | disambiguating | finalizing | done
    "design_intent": None,
    "dot_string": None,
    "run_log": [],
    "explore_state": None,       # saved state from explore_and_ask (messages, tool_log, etc.)
    "clarification_req": None,   # ClarificationRequest dict
    "user_prompt": "",
}
for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Phase labels for display
PHASE_LABELS = {
    "extraction": ("0", "Concept Extraction"),
    "exploration": ("1", "Agentic Exploration"),
    "grounding": ("1.5", "KG Grounding Gate"),
    "disambiguation": ("1.75", "Disambiguation"),
    "clarification_update": ("1.75+", "Clarification Update"),
    "structuring": ("2", "Structured Output"),
    "critic": ("3", "Critic Review"),
}


# ---------------------------------------------------------------------------
# Helper: render a stream of agent events inside a status widget
# ---------------------------------------------------------------------------
def _render_events(event_stream, status_widget):
    """Consume an event generator and render each event in the Streamlit status widget."""
    for event in event_stream:
        etype = event["type"]
        st.session_state.run_log.append(event)

        if etype == "phase":
            num, label = PHASE_LABELS.get(event["phase"], ("?", event["phase"]))
            st.write(f"**Phase {num} — {label}:** {event['detail']}")

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
            st.session_state.stage = "done"
            status_widget.update(
                label=f"Done — {len(di.components)} components, "
                      f"{len(di.connections)} connections",
                state="complete",
                expanded=False,
            )

        elif etype == "error":
            st.error(event["message"])
            status_widget.update(label="Failed", state="error")


# ---------------------------------------------------------------------------
# Stage: INITIAL → start exploration
# ---------------------------------------------------------------------------
if run_btn and prompt.strip():
    for key, default in _DEFAULTS.items():
        st.session_state[key] = default
    st.session_state.user_prompt = prompt.strip()
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
                        options_list = ["(use agent default)"] + q["options"] + ["(custom...)"]
                        choice = st.selectbox(
                            "Your answer",
                            options=options_list,
                            key=f"crit_{i}",
                        )
                        if choice == "(custom...)":
                            custom = st.text_input(
                                "Enter custom answer",
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
                        options_list = ["(use agent default)"] + q["options"] + ["(custom...)"]
                        choice = st.selectbox(
                            "Your answer",
                            options=options_list,
                            key=f"help_{i}",
                        )
                        if choice == "(custom...)":
                            custom = st.text_input(
                                "Enter custom answer",
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
# Stage: FINALIZING — run Phases 2 → 3
# ---------------------------------------------------------------------------
if st.session_state.stage == "finalizing":
    st.markdown("---")

    explore_state = st.session_state.explore_state
    clarifications = st.session_state.get("clarifications")

    label = "Finalizing design"
    if clarifications:
        label += f" (with {len(clarifications)} clarification(s))"
    label += "..."

    with st.status(label, expanded=True) as status_widget:
        _render_events(
            finalize_stream(
                messages=explore_state["messages"],
                tool_log=explore_state["tool_log"],
                user_prompt=st.session_state.user_prompt,
                extracted_dict=explore_state["extracted"],
                model=model,
                max_tool_rounds=max_rounds,
                max_critic_rounds=max_critic,
                clarifications=clarifications,
            ),
            status_widget,
        )

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
        dot_str = st.session_state.dot_string
        try:
            st.graphviz_chart(dot_str, use_container_width=True)
        except Exception as e:
            st.error(f"Graphviz rendering failed: {e}")
            st.code(dot_str, language="dot")

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
        with st.expander("DOT Source", expanded=False):
            st.code(st.session_state.dot_string, language="dot")

        st.download_button(
            label="Download DOT",
            data=st.session_state.dot_string,
            file_name="preschematic.dot",
            mime="text/plain",
        )

    with st.expander("Agent Trace Log", expanded=False):
        for i, event in enumerate(st.session_state.run_log):
            etype = event["type"]
            if etype == "phase":
                st.markdown(f"**[{i}] Phase: {event['phase']}** — {event['detail']}")
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
            elif etype == "error":
                st.markdown(f"[{i}] ❌ {event['message']}")
