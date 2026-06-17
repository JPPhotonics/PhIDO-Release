"""
KG Knowledge Agent — conversational Q&A over the photonics Knowledge Graph.

A minimal ReAct-style agent that answers user questions by autonomously
querying the KG MCP tools (direct Python imports) and synthesising results.

Features:
  - Multi-turn conversation (mutable messages list)
  - Streaming events for real-time UI updates
  - JSONL interaction logging with retrieval metrics for GraphRAG benchmarking

Usage (single-turn):
    from mcp_servers.kg_agent import ask_stream
    msgs = []
    for event in ask_stream("What is a ring resonator?", msgs):
        print(event)

Usage (multi-turn):
    msgs = []
    for event in ask_stream("What is an MZI?", msgs):
        ...
    for event in ask_stream("What physical principles does it use?", msgs):
        ...
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from mcp_servers.llm_client import create_client

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOG_DIR = REPO_ROOT / "logs"
LOG_FILE = LOG_DIR / "kg_agent_log.jsonl"

# ---------------------------------------------------------------------------
# KG tool imports (direct, same pattern as interpreter_agent.py)
# ---------------------------------------------------------------------------
from mcp_servers.kg_server import (
    get_ontology_schema,
    list_entity_types,
    search_concepts,
    get_concept_neighborhood,
    resolve_function,
    get_component_properties,
    kg_stats,
)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a photonics knowledge expert with access to a Knowledge Graph (KG) built
from research literature on photonic integrated circuits. Your job is to answer
the user's questions accurately using ONLY information retrieved from your KG
tools. Do NOT rely on your training data for domain-specific claims.

Available tools:

1. **search_concepts** — Semantic vector search across the KG. Finds components,
   architectures, properties, physical principles, and design functions matching a
   natural-language query. START HERE for most questions.

2. **get_concept_neighborhood** — Graph traversal: given a concept, returns its
   relationships (sub-components, properties, principles, related entities) within
   N hops. Use to understand WHAT something is made of or connected to.

3. **resolve_function** — Reverse lookup: given a design function (e.g.
   "modulation", "wavelength filtering"), find which components or architectures
   can perform it.

4. **get_component_properties** — Detailed knowledge about a specific component:
   its properties, design functions, physical principles, and sub-components.

5. **get_ontology_schema** — The full PIC ontology (classes and relationships).
   Use only when the user asks about the KG structure itself.

6. **list_entity_types** — All relationship types in the KG. Use when the user
   asks about what kinds of relationships exist.

7. **kg_stats** — Summary statistics (node/edge counts). Use when the user asks
   how much knowledge is available.

Strategy:
- For "What is X?" questions: search_concepts → get_component_properties or
  get_concept_neighborhood to fill in details.
- For "What can do X?" questions: resolve_function → then drill into the results.
- For "How does X relate to Y?": search both, then use get_concept_neighborhood.
- Always cite which tool returned the information.

CRITICAL — when to refuse:
- If ALL of your KG tool calls return empty results, zero hits, or only errors,
  you MUST NOT attempt to answer from your training data. Instead, respond with
  exactly: "The Knowledge Graph does not contain information relevant to this
  question. Unable to provide an answer from the available KG data."
- Do NOT pad an empty retrieval with general knowledge. The answer must be
  grounded exclusively in KG-retrieved evidence. A honest "no data" response
  is always preferred over an ungrounded answer.
"""

# ---------------------------------------------------------------------------
# OpenAI tool definitions
# ---------------------------------------------------------------------------
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_concepts",
            "description": (
                "Semantic vector search across the KG. Finds components, architectures, "
                "properties, physical principles, and design functions matching a query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query, e.g. 'ring resonator modulator'",
                    },
                    "entity_type": {
                        "type": "string",
                        "description": (
                            "Optional filter: Components, Architectures, Properties, "
                            "Design_Functions, or Physical_Principles. Leave empty to search all."
                        ),
                        "default": "",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_concept_neighborhood",
            "description": (
                "Graph traversal: get a concept's relationships, sub-components, "
                "properties, principles, and connected entities within N hops."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "concept_name": {
                        "type": "string",
                        "description": "Concept name, e.g. 'MZI', 'Ring_Resonator'",
                    },
                    "max_hops": {
                        "type": "integer",
                        "description": "Number of relationship hops (1-3). Default 1.",
                        "default": 1,
                    },
                },
                "required": ["concept_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_function",
            "description": (
                "Reverse lookup: given a design function (e.g. 'modulation', "
                "'wavelength filtering'), find which components or architectures perform it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "function_description": {
                        "type": "string",
                        "description": "Design function, e.g. 'Modulation', 'Wavelength_Filter'",
                    },
                },
                "required": ["function_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_component_properties",
            "description": (
                "Get all known properties, design functions, physical principles, "
                "and sub-components for a specific component or architecture."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "component_name": {
                        "type": "string",
                        "description": "Component name, e.g. 'MZI', 'Ring_Resonator'",
                    },
                },
                "required": ["component_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ontology_schema",
            "description": "Get the full PIC ontology schema: classes, object properties, data properties.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_entity_types",
            "description": "Get all relationship types from the ontology and any discovered types in the KG.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "kg_stats",
            "description": "Get summary statistics about the KG: node counts by type, edge counts.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------
_TOOL_DISPATCH: dict[str, Any] = {
    "search_concepts": lambda a: search_concepts(a["query"], a.get("entity_type", "")),
    "get_concept_neighborhood": lambda a: get_concept_neighborhood(
        a["concept_name"], a.get("max_hops", 1)
    ),
    "resolve_function": lambda a: resolve_function(a["function_description"]),
    "get_component_properties": lambda a: get_component_properties(a["component_name"]),
    "get_ontology_schema": lambda a: get_ontology_schema(),
    "list_entity_types": lambda a: list_entity_types(),
    "kg_stats": lambda a: kg_stats(),
}

# ---------------------------------------------------------------------------
# JSONL logging helpers
# ---------------------------------------------------------------------------

def _summarise_tool_result(tool_name: str, raw_json: str) -> dict[str, Any]:
    """Extract compact retrieval metrics from a tool's raw JSON output."""
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return {"parse_error": True}

    if tool_name == "search_concepts":
        results = data.get("results", [])
        scores = [r.get("score", 0) for r in results if "score" in r]
        entity_types = list({r.get("entity_type", "") for r in results})
        return {
            "num_hits": len(results),
            "top_score": max(scores) if scores else None,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else None,
            "top_hit": results[0].get("name") if results else None,
            "entity_types_searched": entity_types,
            "num_errors": len(data.get("errors", [])),
        }

    if tool_name == "get_component_properties":
        return {
            "found": "error" not in data,
            "num_properties": len(data.get("properties", [])),
            "num_functions": len(data.get("design_functions", [])),
            "num_principles": len(data.get("physical_principles", [])),
            "num_sub_components": len(data.get("sub_components", [])),
        }

    if tool_name == "get_concept_neighborhood":
        return {
            "found": "error" not in data,
            "total_neighbors": data.get("total_neighbors", 0),
            "relationship_types": list(data.get("neighborhood", {}).keys()),
        }

    if tool_name == "resolve_function":
        return {
            "match_type": data.get("match_type"),
            "num_components": len(data.get("components", [])),
            "similarity": data.get("similarity"),
        }

    if tool_name == "kg_stats":
        return {
            "total_nodes": data.get("total_nodes", 0),
            "total_edges": data.get("total_edges", 0),
        }

    if tool_name == "get_ontology_schema":
        return {
            "num_classes": len(data.get("classes", [])),
            "num_object_properties": len(data.get("object_properties", [])),
            "num_data_properties": len(data.get("data_properties", [])),
        }

    if tool_name == "list_entity_types":
        return {
            "num_ontology_relationships": len(data.get("ontology_relationships", [])),
            "num_discovered_relationships": len(data.get("discovered_relationships", [])),
        }

    return {}


def _aggregate_retrieval_metrics(tool_trace: list[dict]) -> dict[str, Any]:
    """Compute cross-call retrieval metrics from the full tool trace."""
    tools_used = list({t["tool"] for t in tool_trace})
    search_entries = [t for t in tool_trace if t["tool"] == "search_concepts"]

    all_scores: list[float] = []
    total_hits = 0
    entity_types: set[str] = set()

    for entry in search_entries:
        summary = entry.get("result_summary", {})
        hits = summary.get("num_hits", 0)
        total_hits += hits
        top = summary.get("top_score")
        avg = summary.get("avg_score")
        if top is not None:
            all_scores.append(top)
        for et in summary.get("entity_types_searched", []):
            entity_types.add(et)

    return {
        "total_tool_calls": len(tool_trace),
        "tools_used": tools_used,
        "search_calls": len(search_entries),
        "total_kg_hits": total_hits,
        "max_similarity_score": round(max(all_scores), 4) if all_scores else None,
        "avg_similarity_score": (
            round(sum(all_scores) / len(all_scores), 4) if all_scores else None
        ),
        "entity_types_searched": sorted(entity_types),
    }


def _log_interaction(entry: dict[str, Any]) -> None:
    """Append a single JSON object to the JSONL log file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Streaming agent loop
# ---------------------------------------------------------------------------

# Event type alias for clarity
PipelineEvent = dict[str, Any]


def ask_stream(
    question: str,
    messages: list[dict[str, Any]],
    model: str = "gpt-4o",
    max_rounds: int = 10,
) -> Generator[PipelineEvent, None, None]:
    """Run a tool-calling ReAct loop over the KG and yield streaming events.

    Args:
        question: The user's natural-language question.
        messages: Mutable conversation history (OpenAI message list). Initialised
            with the system prompt on first call if empty. Mutated in-place so the
            caller can pass it back for multi-turn conversation.
        model: OpenAI model name.
        max_rounds: Maximum number of LLM round-trips (safety cap).

    Yields:
        Events of the form ``{"type": ..., ...}``:
        - ``{"type": "tool_call", "name": str, "args": dict}``
        - ``{"type": "tool_result", "name": str, "result": str, "summary": dict, "latency_s": float}``
        - ``{"type": "answer", "content": str}``
    """
    client = create_client(model)

    if not messages:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

    messages.append({"role": "user", "content": question})

    tool_trace: list[dict[str, Any]] = []
    t_start = time.perf_counter()
    tool_rounds = 0

    for _ in range(max_rounds):
        resp = client.complete(messages, tools=TOOLS)

        if resp.stop_reason == "stop" and not resp.tool_calls:
            answer = resp.content or ""
            messages.append({"role": "assistant", "content": answer})
            yield {"type": "answer", "content": answer}
            break

        messages.append(client.assistant_message(resp))
        tool_rounds += 1

        for tc in resp.tool_calls:
            fn_name = tc.name
            try:
                args = json.loads(tc.arguments)
            except json.JSONDecodeError:
                args = {}

            yield {"type": "tool_call", "name": fn_name, "args": args}

            t_tool = time.perf_counter()
            dispatch_fn = _TOOL_DISPATCH.get(fn_name)
            if dispatch_fn is None:
                result_str = json.dumps({"error": f"Unknown tool: {fn_name}"})
            else:
                try:
                    result_str = dispatch_fn(args)
                except Exception as exc:
                    result_str = json.dumps({"error": str(exc)})
            tool_latency = round(time.perf_counter() - t_tool, 3)

            summary = _summarise_tool_result(fn_name, result_str)
            tool_trace.append({
                "tool": fn_name,
                "args": args,
                "result_summary": summary,
                "latency_s": tool_latency,
            })

            yield {
                "type": "tool_result",
                "name": fn_name,
                "result": result_str,
                "summary": summary,
                "latency_s": tool_latency,
            }

            messages.append(client.tool_result_message(tc, result_str))
    else:
        answer = messages[-1].get("content", "") if messages else ""
        yield {"type": "answer", "content": answer}

    total_latency = round(time.perf_counter() - t_start, 3)

    # --- Log the interaction ---
    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "question": question,
        "answer": answer,
        "latency_s": total_latency,
        "tool_trace": tool_trace,
        "retrieval_metrics": _aggregate_retrieval_metrics(tool_trace),
        "tool_rounds": tool_rounds,
    }
    _log_interaction(log_entry)


def _serialise_assistant_message(msg: Any) -> dict[str, Any]:
    """Convert an OpenAI ChatCompletionMessage to a plain dict for the messages list."""
    out: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return out
