"""
Streamlit Chat UI for the KG Knowledge Agent.

Run with:
    streamlit run mcp_servers/kg_chat.py

Multi-turn conversational interface that streams tool calls and answers
from the KG agent in real time.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# Path setup so imports resolve from repo root
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp_servers.kg_agent import ask_stream, LOG_FILE

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Photonics KG Agent",
    page_icon="🔬",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # OpenAI message list (mutated by ask_stream)

if "chat_display" not in st.session_state:
    st.session_state.chat_display = []  # [{role, content, tool_trace?}]

_MODEL_OPTIONS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5.4",
    "o3-mini",
    "claude-sonnet-4-20250514",
    "claude-3.5-sonnet-20241022",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

if "model" not in st.session_state:
    st.session_state.model = "gpt-4o"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("Settings")
    st.session_state.model = st.selectbox(
        "Model",
        _MODEL_OPTIONS,
        index=_MODEL_OPTIONS.index(st.session_state.model)
        if st.session_state.model in _MODEL_OPTIONS else 0,
    )

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.chat_display = []
        st.rerun()

    st.divider()
    st.caption("Benchmark log")
    if LOG_FILE.exists():
        line_count = sum(1 for _ in open(LOG_FILE, encoding="utf-8"))
        st.metric("Logged interactions", line_count)
        st.code(str(LOG_FILE), language=None)
    else:
        st.write("No interactions logged yet.")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Photonics Knowledge Graph Agent")
st.caption("Ask questions about photonic components, architectures, and design principles.")

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for entry in st.session_state.chat_display:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("tool_trace"):
            with st.expander(f"Tool calls ({len(entry['tool_trace'])})", expanded=False):
                for tc in entry["tool_trace"]:
                    st.markdown(f"**{tc['name']}** `{json.dumps(tc['args'])}`")
                    if tc.get("summary"):
                        st.json(tc["summary"])

# ---------------------------------------------------------------------------
# Chat input + agent loop
# ---------------------------------------------------------------------------
TOOL_LABELS = {
    "search_concepts": "Searching KG",
    "get_concept_neighborhood": "Exploring neighborhood",
    "resolve_function": "Resolving function",
    "get_component_properties": "Looking up component",
    "get_ontology_schema": "Fetching ontology",
    "list_entity_types": "Listing entity types",
    "kg_stats": "Getting KG stats",
}

if prompt := st.chat_input("Ask about photonics..."):
    # Display user message immediately
    st.session_state.chat_display.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Run agent and stream results
    with st.chat_message("assistant"):
        status_container = st.status("Querying Knowledge Graph...", expanded=True)
        answer_placeholder = st.empty()

        tool_trace_display: list[dict] = []
        final_answer = ""

        for event in ask_stream(
            prompt,
            st.session_state.messages,
            model=st.session_state.model,
        ):
            etype = event["type"]

            if etype == "tool_call":
                label = TOOL_LABELS.get(event["name"], event["name"])
                status_container.update(label=f"{label}...", state="running")
                status_container.write(
                    f"**{event['name']}** `{json.dumps(event['args'])}`"
                )

            elif etype == "tool_result":
                summary = event.get("summary", {})
                tool_trace_display.append({
                    "name": event["name"],
                    "args": {},  # already shown in tool_call
                    "summary": summary,
                    "latency_s": event.get("latency_s"),
                })
                detail_parts = []
                if "num_hits" in summary:
                    detail_parts.append(f"{summary['num_hits']} hits")
                    if summary.get("top_score") is not None:
                        detail_parts.append(f"top score {summary['top_score']:.2f}")
                elif "total_neighbors" in summary:
                    detail_parts.append(f"{summary['total_neighbors']} neighbors")
                elif "num_components" in summary:
                    detail_parts.append(f"{summary['num_components']} components")
                elif summary.get("found") is False:
                    detail_parts.append("not found")

                detail_str = f" ({', '.join(detail_parts)})" if detail_parts else ""
                latency_str = f" [{event.get('latency_s', '?')}s]"
                status_container.write(f"  ↳ Result{detail_str}{latency_str}")

            elif etype == "answer":
                final_answer = event["content"]
                status_container.update(label="Done", state="complete", expanded=False)
                answer_placeholder.markdown(final_answer)

    st.session_state.chat_display.append({
        "role": "assistant",
        "content": final_answer,
        "tool_trace": tool_trace_display,
    })
