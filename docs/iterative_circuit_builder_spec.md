# Iterative Circuit Builder — Implementation Specification

## 1. Overview

The Iterative Circuit Builder is an agentic architecture for constructing complex photonic
integrated circuits (PICs) incrementally via LLM tool calls. Instead of generating the
entire circuit in a single structured-output call, the LLM places components one at a time
on a stateful `CircuitGraph` object, wires them, and receives immediate feedback after
every operation.

### Design Rationale

The single-shot extraction approach struggles with large circuits (e.g., 8×8 Clements
meshes with 28 MZIs, 1×64 splitter trees) because:

- The LLM must hold the entire topology in working memory at once.
- Port-level routing errors compound with no opportunity for correction.
- A prior hierarchical decomposition approach suffered from context loss at stitch points.

The iterative builder solves these by maintaining a mutable graph that enforces structural
invariants after every operation, giving the LLM immediate error feedback and wiring hints.

### Key Files

| File | Role |
|------|------|
| `mcp_servers/circuit_graph.py` | `CircuitGraph` — stateful graph object with validation |
| `mcp_servers/interpreter_agent.py` | `build_circuit_iterative()` — agentic tool-calling loop |
| `mcp_servers/pipeline_orchestrator.py` | Pipeline integration, port-preserving translation |
| `mcp_servers/models.py` | `Connection` model with `from_port`/`to_port` fields |
| `mcp_servers/streamlit_app.py` | UI: extraction mode selector, live DOT visualization |
| `tests/test_circuit_graph.py` | Unit tests (54 tests across 11 test classes) |

---

## 2. Data Structures

### 2.1 `Position`

Logical grid coordinate for a component.

```python
@dataclass
class Position:
    stage: int   # signal-flow axis (0 = input, increasing rightward)
    lane: int    # cross-axis (0 = top, increasing downward)
```

Each `(stage, lane)` pair must be unique across all components in the graph.

### 2.2 `PortInfo`

State of a single port on a component.

```python
@dataclass
class PortInfo:
    name: str                            # e.g. "o1", "o2", "o3", "o4"
    direction: str                       # "input" or "output"
    connected_to: Optional[str] = None   # e.g. "C3.o1" or None
```

Ports follow the GDSFactory convention. For an `NxM` port config:
- `o1..oN` are **inputs** (left side)
- `o(N+1)..o(N+M)` are **outputs** (right side)

### 2.3 `ComponentNode`

A placed component in the circuit graph.

```python
@dataclass
class ComponentNode:
    id: str                        # auto-assigned: "C1", "C2", ...
    component_type: str            # resolved PDK module name
    sub_type: Optional[str]
    port_config: str               # e.g. "2x2", "1x2"
    role: Optional[str]
    description: str
    position: Position
    ports: dict[str, PortInfo]
    specs: dict[str, str]
    pdk_name: Optional[str]        # human-readable PDK name
```

### 2.4 `ConnectionEdge`

A validated connection between two ports.

```python
@dataclass
class ConnectionEdge:
    from_id: str       # source component ID
    from_port: str     # source port name
    to_id: str         # target component ID
    to_port: str       # target port name
    description: str   # natural language description
```

### 2.5 `PDKEntry`

Lightweight PDK catalog entry for validation.

```python
@dataclass
class PDKEntry:
    module_name: str        # canonical PDK module name
    name: str               # human-readable name
    port_config: str        # e.g. "2x2"
    labels: list[str]
    aliases: list[str]
```

---

## 3. Physical Lane Model

Each port has a **physical lane** — its absolute vertical position in the circuit. This is
computed from the component's lane position and the port's role within the component.

### Formula

For a **2×2** component at lane `L`:

| Port | Direction | Physical Lane |
|------|-----------|---------------|
| `o2` | input (top) | `L` |
| `o1` | input (bottom) | `L + 1` |
| `o3` | output (top) | `L` |
| `o4` | output (bottom) | `L + 1` |

For a **1×2** component at lane `L`:

| Port | Direction | Physical Lane |
|------|-----------|---------------|
| `o1` | input | `L` |
| `o2` | output (top) | `L` |
| `o3` | output (bottom) | `L + 1` |

### Usage

Physical lanes are used to:

1. **Suggest lane-aligned connections** in `_find_connectable_ports()`: when a new component
   is placed, the system identifies which open output ports at adjacent stages share the same
   physical lane as the new component's input ports, and suggests those as `"same_lane"`
   connections.

2. **Enrich the wiring reminder**: when the LLM places components without connecting them,
   the injected reminder includes explicit `connect()` suggestions annotated with
   `# same physical_lane=N`.

3. **Annotate tool responses**: both `add_component` and `get_open_ports` include
   `physical_lane` in their return payloads, so the LLM can reason about spatial alignment.

---

## 4. CircuitGraph — Public API

### 4.1 `__init__(pdk_catalog=None)`

Creates an empty graph. If `pdk_catalog` (list of dicts or `PDKEntry` objects) is provided,
builds internal indexes for PDK validation:

- `_pdk_index`: `module_name` → `PDKEntry` (case-insensitive)
- `_pdk_alias_map`: `alias` → `module_name`

### 4.2 `add_component(component_type, port_config, stage, lane, ...)`

Place a component at a specific `(stage, lane)` position.

**Validation chain:**

1. **PDK validation** (when catalog loaded): resolves `component_type` against the catalog
   via exact match, case-insensitive match, alias lookup, or fuzzy substring match. Returns
   error with suggestions if not found. Rejects mismatched `port_config`.
2. **Format validation**: `port_config` must match `NxM` regex.
3. **Position validation**: `(stage, lane)` must be unoccupied.

**Returns:** Dict with `id`, `component_type`, `position`, `ports` (including `physical_lane`
for each port), `pdk_name`, and `connectable` (lane-aligned wiring hints from adjacent stages).

### 4.3 `connect(from_component, from_port, to_component, to_port, ...)`

Wire two ports together.

**Validation chain:**

1. **Component existence**: both must exist.
2. **Self-connection guard**: `from_component == to_component` → error.
3. **Port existence**: both ports must exist on their respective components.
4. **Double connection**: neither port may already be connected.
5. **Forward flow**: source stage must be ≤ target stage (unless `feedback=True`).
6. **Stage adjacency**: stages must differ by at most 1 (unless `skip=True` or `feedback=True`).
7. **Output-to-output guard**: connecting two output ports → error (hard block).
8. **Input-to-input**: warning only (non-blocking).
9. **Lane proximity**: `abs(lane_diff) > 3` → warning.

**Returns:** `{"status": "connected", "edge": "C1.o3 -> C2.o2"}` on success, with optional
`warnings` list.

### 4.4 `get_open_ports()`

Returns all unconnected ports grouped by stage. Each port entry includes `component`,
`port`, `direction`, `physical_lane`, `component_type`, `position`, and `description`.

Also returns `total_components`, `total_connections`, `furthest_stage`, and a
human-readable `summary`.

### 4.5 `get_state()`

Returns a full circuit summary: component/connection counts, grid layout (stage → lane →
component info), and per-component port maps showing connected vs open ports.

### 4.6 `replicate_stage(source_components, count, connect_from, connection_rule, ...)`

Bulk-replicate a pattern of components with automatic wiring. Designed for repeating
structures (mesh columns, tree levels, cascaded filters).

**Parameters:**
- `source_components`: list of component IDs to clone
- `count`: number of replications
- `connect_from`: port to connect from (on source component outputs)
- `connection_rule`: `"one_to_one"` or `"chain"`

Places new components at incrementing stages and wires them according to the rule.

### 4.7 `finalize(title, brief_summary, force=False, ...)`

Signal that construction is complete. Runs a wiring adequacy check: if the circuit has
too few connections relative to its component count (and `force` is `False`), returns
an error with a list of disconnected components.

Also runs BFS from the first component to detect disconnected subgraphs, returning
warnings for unreachable components.

Sets `_finalized = True` on acceptance.

### 4.8 `to_design_intent()`

Serializes the graph to a `DesignIntent` (the standard downstream format). Crucially,
the `Connection` objects include `from_port` and `to_port` fields, preserving the
exact port-level wiring for downstream translation.

### 4.9 `to_dot(highlight=True)`

Generates a Graphviz DOT string (directed graph, `rankdir=LR`) with:

- **Record-shaped nodes**: showing input ports (left), component ID + PDK name (center),
  and output ports (right).
- **Port ordering**: input ports are reversed so `o2` (top) renders above `o1` (bottom),
  matching GDSFactory visual conventions.
- **Directed edges** with `taillabel`/`headlabel` for port names.
- **`rank=same`** constraints grouping components by stage.
- **Highlighting**: newly added nodes (blue fill) and edges (blue, bold) when
  `highlight=True`.

---

## 5. Agentic Loop — `build_circuit_iterative()`

### 5.1 Initialization

1. Extracts the user prompt and concept summary from the exploration state.
2. Initializes `CircuitGraph(pdk_catalog=_PDK_CATALOG)` with the full PDK catalog.
3. Prepares the message list with the system prompt (`ITERATIVE_BUILD_PROMPT`) and a
   directive user message.
4. Merges builder tools (6) with PDK/KG tools (~11) into a single tool list.

### 5.2 Round Structure

Each round of the loop:

1. Calls `client.chat.completions.create(model, messages, tools)`.
2. If the response contains **tool calls**:
   - Dispatches each tool call to the appropriate handler (`_dispatch_builder_tool` for
     builder tools, PDK/KG tools for lookup).
   - Appends tool results as `role: tool` messages.
   - Emits `circuit_updated` events (with DOT string) after builder tool calls.
   - Tracks `adds_this_round` and `connects_this_round` for wiring enforcement.
   - If `finalize` is accepted, sets `finalized = True` and breaks.
3. If the response is **text only** (no tool calls):
   - Increments `consecutive_text_only` counter.
   - Injects escalating nudges to push the LLM back to tool calling.

### 5.3 Wiring Enforcement

After each round where `add_component` was called but `connect` was not:

1. Calls `graph.get_open_ports()` to get all unconnected ports with physical lane info.
2. For each output port, finds input ports at the same or next stage on the same
   physical lane — **excluding same-component matches** (preventing self-connection hints).
3. If wireable pairs exist, injects a user message with explicit `connect()` suggestions
   annotated with physical lane info.
4. If no wireable pairs exist (e.g., first stage placement), no reminder is injected.

### 5.4 Nudging for Text-Only Responses

When the LLM responds with text instead of tool calls:

- **Mild nudge** (1–2 consecutive text rounds): reminds the LLM to call tools with a
  progress hint from `get_state()`.
- **Strong nudge** (3+ consecutive text rounds): more forceful instruction to stop writing
  text and call tools immediately.

### 5.5 Context Compaction

After 3 consecutive text-only rounds, `_compact_messages()` removes redundant
assistant-text + user-nudge pairs to prevent context bloat from repeated
nudge/narration cycles.

### 5.6 Auto-Finalize

If `max_rounds` is reached without the LLM calling `finalize`, the system calls
`graph.finalize(..., force=True)` to force-accept the current state.

### 5.7 Logging

All tool calls, arguments, results, and round metadata are logged to
`iterative_builder.log` via Python's `logging` module at DEBUG level.

---

## 6. System Prompt

The `ITERATIVE_BUILD_PROMPT` instructs the LLM to:

1. **Only use tool calls** — never narrate or plan in text.
2. **Follow the (stage, lane) grid** — each component occupies a unique position.
3. **Understand port spatial conventions** — the prompt documents the GDSFactory port
   layout for 2×2 and 1×2 components, including the physical lane alignment model.
4. **Place then wire, stage by stage** — place a stage's components, immediately connect
   them, then move to the next stage.
5. **Use `replicate_stage`** for repeating structures.
6. **Never self-connect** — connecting a component to itself is physically impossible.
7. **Never call `finalize` prematurely** — check `get_open_ports` first.
8. **Batch tool calls** — include multiple `add_component` and `connect` calls per round.

### Builder Tools Available

| Tool | Purpose |
|------|---------|
| `add_component` | Place a component at (stage, lane) |
| `connect` | Wire two ports together |
| `get_open_ports` | View unconnected ports by stage |
| `get_state` | Full circuit summary |
| `replicate_stage` | Bulk-replicate patterns with auto-wiring |
| `finalize` | Signal construction complete |

The LLM also has access to PDK tools (`search_pdk`, `validate_ports`, `get_component_info`,
`get_module_params`, `get_pdk_cell_details`) and KG tools for domain lookups.

---

## 7. Pipeline Integration

### 7.1 Extraction Mode Selection

The user selects the extraction mode in the Streamlit sidebar:

- **Single-shot** (default): one structured-output call via `extract_design_intent()`.
- **Iterative builder**: incremental construction via `build_circuit_iterative()`.
- **Auto**: uses iterative if estimated component count ≥ 12 (the `_AUTO_ITERATIVE_THRESHOLD`).

### 7.2 Iterative Path in `run_pipeline_finalize`

When `extraction_mode="iterative"`:

1. Runs `build_circuit_iterative()`, streaming `circuit_updated` events for live visualization.
2. Receives the final `DesignIntent` (with port-level connections).
3. **Skips the LLM critic** (the iterative builder's own validation is considered sufficient).
4. Runs the **Clingo topology gate** for structural validation.
5. Runs **component selection** to map component types to PDK modules.
6. Runs **selection validation** with optional feedback loop.

### 7.3 Port-Preserving Translation

The iterative builder's port-level connections flow through to the final schematic:

```
CircuitGraph                    DesignIntent                 Circuit DSL
(ConnectionEdge with            (Connection with             (edges dict with
 from_port, to_port)  ────►     from_port, to_port)  ────►   "C1,o3: C3,o2")
```

1. **`Connection` model** (`models.py`): Has optional `from_port` and `to_port` fields
   (set by iterative builder, `None` for single-shot).

2. **`CircuitGraph.to_design_intent()`**: Populates `from_port` and `to_port` from
   `ConnectionEdge` fields.

3. **`_build_circuit_dsl()`** (`pipeline_orchestrator.py`): When connections have port info,
   populates the DSL `"edges"` dict directly (format: `"C1,o3: C3,o2"`).

4. **LLM edge routing bypass**: When the DSL already has edges, the pipeline injects them
   directly into the DOT string as undirected edges (`C1:o3 -- C3:o2;`) and **skips the
   `_run_edge_routing()` LLM call entirely**. This eliminates the LLM re-guessing the wiring.

The single-shot path is unaffected — it still gets empty edges and uses LLM edge routing.

### 7.4 Feedback Re-runs

When user feedback is classified as Tier 3 (architectural), `run_pipeline_with_feedback`
calls `run_pipeline_finalize` with the same `extraction_mode` the user originally selected.
This means iterative builder re-runs are correctly routed through the iterative path.

---

## 8. Streamlit UI

### 8.1 Live Visualization

During the iterative build, each `circuit_updated` event carries the current DOT string
from `CircuitGraph.to_dot()`. The UI renders this via `st.graphviz_chart()` in a
placeholder that updates in real-time, showing:

- Components appearing stage by stage.
- Connections being drawn as the LLM wires them.
- Newly added nodes/edges highlighted in blue.

### 8.2 Final Schematic

After the iterative build completes, the pipeline generates a detailed schematic DOT
(with proper GDSFactory component labels and port-accurate edges) that replaces the
incremental visualization. This schematic uses the same port-level connections established
during the build.

---

## 9. Validation Summary

### Structural Invariants

| Invariant | Enforced By | Behavior |
|-----------|-------------|----------|
| Unique `(stage, lane)` positions | `add_component` | Error |
| No self-connections | `connect` | Error |
| No output-to-output connections | `connect` | Error |
| Forward flow (source ≤ target stage) | `connect` | Error (bypass with `feedback=True`) |
| Stage adjacency (≤ 1 stage gap) | `connect` | Error (bypass with `skip=True`) |
| No double connections | `connect` | Error |
| Valid port config format | `add_component` | Error |
| PDK component validation | `add_component` | Error with fuzzy suggestions |
| PDK port config match | `add_component` | Error |
| Wiring adequacy at finalize | `finalize` | Error (bypass with `force=True`) |
| Subgraph connectivity at finalize | `finalize` | Warning |

### LLM Behavioral Guards

| Guard | Mechanism |
|-------|-----------|
| Prompt forbids self-connections | System prompt rule |
| Prompt forbids text-only responses | System prompt + escalating nudges |
| Wiring reminders with lane-aligned hints | Injected user messages after add-without-connect |
| Context compaction for narration loops | `_compact_messages()` after 3 text-only rounds |
| Premature finalize rejection | `finalize` wiring adequacy check |

---

## 10. Test Coverage

54 unit tests across 11 test classes in `tests/test_circuit_graph.py`:

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestMakePorts` | 4 | Port generation for various configs |
| `TestAddComponent` | 5 | Happy path, ID increment, collisions, bad config |
| `TestConnect` | 9 | Valid wiring, all error cases, bypass flags |
| `TestGetOpenPorts` | 3 | Open ports tracking, frontier after connect |
| `TestGetState` | 2 | Counts and grid layout |
| `TestFinalize` | 4 | Clean finalize, disconnected warnings, single-component |
| `TestToDesignIntent` | 2 | Round-trip serialization, specs preservation |
| `TestToDot` | 4 | Rank constraints, highlighting, rankdir |
| `TestReplicateStage` | 5 | One-to-one, chain, error cases |
| `TestPDKValidation` | 9 | Exact/alias/case matching, fuzzy, catalog modes |
| `TestDetailedDot` | 5 | Digraph format, ports, PDK names, highlight colors |

---

## 11. Known Limitations and Future Work

1. **Feedback injection**: Tier 3 architectural re-runs route through the iterative builder
   but do not currently inject the user's schematic feedback text into
   `build_circuit_iterative`. The feedback is only used by the single-shot extraction path.

2. **Live DOT during feedback re-runs**: The Streamlit feedback path does not pass a
   `graph_placeholder` to `_render_events`, so live iterative DOT updates are not visible
   during Tier 3 feedback re-runs (only during the initial finalize).

3. **LLM wiring quality**: While physical lane hints and self-connection guards significantly
   improve wiring behavior, the LLM may still make suboptimal routing choices for complex
   staggered topologies. Further improvements could include stronger same-lane enforcement
   or deterministic wiring templates for known architectures.

4. **`replicate_stage` adoption**: The LLM does not always use `replicate_stage` for
   repeating structures, preferring manual placement. Further prompt engineering or
   detection logic could encourage its use.
