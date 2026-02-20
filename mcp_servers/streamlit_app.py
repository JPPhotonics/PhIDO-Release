"""
Streamlit UI for the PhIDO Interpreter Agent.

Streams the agent's tool-calling loop in real time, displays the structured
DesignIntent output, and renders a preschematic DOT graph.

Run with:
    streamlit run mcp_servers/streamlit_app.py
"""

import json
import pandas as pd
import streamlit as st

from mcp_servers.interpreter_agent import interpret_stream
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
# State
# ---------------------------------------------------------------------------
if "design_intent" not in st.session_state:
    st.session_state.design_intent = None
if "dot_string" not in st.session_state:
    st.session_state.dot_string = None
if "run_log" not in st.session_state:
    st.session_state.run_log = []

# ---------------------------------------------------------------------------
# Run the agent
# ---------------------------------------------------------------------------
if run_btn and prompt.strip():
    # Reset previous results
    st.session_state.design_intent = None
    st.session_state.dot_string = None
    st.session_state.run_log = []

    st.markdown("---")

    # Phase progress
    phase_labels = {
        "extraction": ("0", "Concept Extraction"),
        "exploration": ("1", "Agentic Exploration"),
        "grounding": ("1.5", "KG Grounding Gate"),
        "structuring": ("2", "Structured Output"),
        "critic": ("3", "Critic Review"),
    }

    with st.status("Running interpreter agent...", expanded=True) as status_widget:
        current_phase = ""
        tool_counter = 0

        for event in interpret_stream(
            user_prompt=prompt.strip(),
            model=model,
            max_tool_rounds=max_rounds,
            max_grounding_rounds=max_grounding,
            max_critic_rounds=max_critic,
        ):
            etype = event["type"]
            st.session_state.run_log.append(event)

            if etype == "phase":
                phase_key = event["phase"]
                num, label = phase_labels.get(phase_key, ("?", phase_key))
                current_phase = phase_key
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
                tool_counter += 1
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
                # Render concepts as a compact tag-like grid
                n_cols = 4
                cols = st.columns(n_cols)
                for i, concept in enumerate(concepts):
                    cols[i % n_cols].markdown(f"`{concept}`")

            elif etype == "stats":
                st.write(
                    f"**Tool call summary:** {event['total']} total "
                    f"({event['pdk']} PDK, {event['kg']} KG)"
                )

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
# Display results
# ---------------------------------------------------------------------------
di: DesignIntent | None = st.session_state.design_intent

if di is not None:
    st.markdown("---")
    st.header("Design Intent")

    # Title and summary
    st.subheader(di.title)
    st.markdown(di.brief_summary)

    # Two-column layout: table + graph
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

    # Connections table
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

    # Ambiguities
    if di.ambiguities:
        st.subheader("Ambiguities & Assumptions")
        for amb in di.ambiguities:
            st.markdown(f"- {amb}")

    # Expandable sections
    st.markdown("---")

    col_json, col_dot = st.columns(2)

    with col_json:
        with st.expander("Full DesignIntent JSON", expanded=False):
            full_json = json.dumps(di.full(), indent=2, ensure_ascii=False)
            st.code(full_json, language="json")

        # Download button
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

    # Agent trace (collapsed by default)
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
            elif etype == "critic":
                v = event["verdict"]
                status = "PASSED" if v["passed"] else "FAILED"
                st.markdown(f"[{i}] 🧐 Critic {status} (attempt {event.get('attempt', '?')}): "
                            f"{v['summary'][:150]}")
            elif etype == "done":
                st.markdown(f"[{i}] ✅ Done")
            elif etype == "error":
                st.markdown(f"[{i}] ❌ {event['message']}")

