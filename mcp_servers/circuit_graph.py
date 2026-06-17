"""
Iterative Circuit Builder — stateful graph object for incremental circuit construction.

The LLM builds a photonic circuit by calling methods on this object (via tool calls).
The graph enforces structural invariants after every operation and provides feedback.

When construction is complete, ``to_design_intent()`` serializes the internal state
to a standard ``DesignIntent`` that downstream pipeline stages consume unchanged.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from mcp_servers.models import (
    ComponentIntent,
    Connection,
    DesignIntent,
    SpecEntry,
)


# ---------------------------------------------------------------------------
# PDK catalog types
# ---------------------------------------------------------------------------

@dataclass
class PDKEntry:
    """Lightweight representation of a PDK component for validation."""
    module_name: str
    name: str
    port_config: str
    labels: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """Logical grid coordinate for a component."""
    stage: int
    lane: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.stage, self.lane)


@dataclass
class PortInfo:
    """State of a single port on a component."""
    name: str
    direction: str  # "input" or "output"
    connected_to: Optional[str] = None  # "C3.o1" or None


@dataclass
class ComponentNode:
    """A placed component in the circuit graph."""
    id: str
    component_type: str
    sub_type: Optional[str]
    port_config: str
    role: Optional[str]
    description: str
    position: Position
    ports: dict[str, PortInfo] = field(default_factory=dict)
    specs: dict[str, str] = field(default_factory=dict)
    pdk_name: Optional[str] = None


@dataclass
class ConnectionEdge:
    """A validated connection between two ports."""
    from_id: str
    from_port: str
    to_id: str
    to_port: str
    description: str


_PORT_CONFIG_RE = re.compile(r"^(\d+)x(\d+)$")


def _port_physical_lane(component_lane: int, port_name: str, port_config: str) -> int:
    """Compute the physical lane occupied by a port.

    For a 2x2 component at lane L:
      o2 (top input)    -> physical lane L
      o1 (bottom input) -> physical lane L+1
      o3 (top output)   -> physical lane L
      o4 (bottom output)-> physical lane L+1

    For a 1x2 component at lane L:
      o1 (input)        -> physical lane L  (centered)
      o2 (top output)   -> physical lane L
      o3 (bottom output)-> physical lane L+1

    For 1x1 components, the single port maps to lane L.
    """
    m = _PORT_CONFIG_RE.match(port_config)
    if not m:
        return component_lane
    n_in, n_out = int(m.group(1)), int(m.group(2))

    if n_in == 2 and n_out == 2:
        return component_lane + {
            "o2": 0, "o1": 1,   # inputs: top=+0, bottom=+1
            "o3": 0, "o4": 1,   # outputs: top=+0, bottom=+1
        }.get(port_name, 0)
    elif n_in == 1 and n_out == 2:
        return component_lane + {"o1": 0, "o2": 0, "o3": 1}.get(port_name, 0)
    elif n_in == 2 and n_out == 1:
        return component_lane + {"o1": 1, "o2": 0, "o3": 0}.get(port_name, 0)
    else:
        # Generic fallback: distribute ports across lanes
        port_idx = int(port_name[1:]) - 1 if port_name.startswith("o") else 0
        total = n_in + n_out
        return component_lane + (port_idx * max(1, total - 1) // max(1, total - 1))


def _make_ports(port_config: str) -> dict[str, PortInfo]:
    """Generate ports from an NxM port config string (GDSFactory convention)."""
    m = _PORT_CONFIG_RE.match(port_config)
    if not m:
        raise ValueError(f"Invalid port_config '{port_config}': must match NxM (e.g. '2x2')")
    n_in, n_out = int(m.group(1)), int(m.group(2))
    ports: dict[str, PortInfo] = {}
    for i in range(1, n_in + 1):
        ports[f"o{i}"] = PortInfo(name=f"o{i}", direction="input")
    for i in range(n_in + 1, n_in + n_out + 1):
        ports[f"o{i}"] = PortInfo(name=f"o{i}", direction="output")
    return ports


# ---------------------------------------------------------------------------
# CircuitGraph
# ---------------------------------------------------------------------------

class CircuitGraph:
    """Stateful, mutable circuit graph with enforced positional constraints.

    Provides the tool-callable methods that the LLM uses to build a circuit
    incrementally, plus serialization to ``DesignIntent`` and DOT.

    Parameters
    ----------
    pdk_catalog : list[PDKEntry] | list[dict] | None
        If provided, ``add_component`` will enforce that ``component_type``
        matches a known PDK module name and that ``port_config`` is valid
        for that module.  Pass ``None`` to disable enforcement (unit-test
        mode).
    """

    def __init__(self, pdk_catalog: list[PDKEntry] | list[dict] | None = None) -> None:
        self._components: dict[str, ComponentNode] = {}
        self._connections: list[ConnectionEdge] = []
        self._position_map: dict[tuple[int, int], str] = {}
        self._id_counter: int = 0

        # Finalization metadata (set by finalize())
        self._title: str = ""
        self._summary: str = ""
        self._architecture_type: Optional[str] = None
        self._n_value: Optional[int] = None
        self._ambiguities: list[str] = []
        self._finalized: bool = False

        # Highlight tracking for live visualization
        self._last_added_ids: set[str] = set()
        self._last_added_edges: set[tuple[str, str, str, str]] = set()

        # PDK validation index
        self._pdk_index: dict[str, PDKEntry] | None = None
        self._pdk_alias_map: dict[str, str] | None = None
        if pdk_catalog is not None:
            self._build_pdk_index(pdk_catalog)

    def _build_pdk_index(self, catalog: list[PDKEntry] | list[dict]) -> None:
        """Build fast lookup structures from the PDK catalog."""
        self._pdk_index = {}
        self._pdk_alias_map = {}
        for entry in catalog:
            if isinstance(entry, dict):
                mod = entry.get("module_name", "")
                entry = PDKEntry(
                    module_name=mod,
                    name=entry.get("name", mod),
                    port_config=str(entry.get("ports", "unknown")),
                    labels=entry.get("labels", []),
                    aliases=[
                        a.strip().lower()
                        for a in (entry.get("aka") or "").split(",") if a.strip()
                    ],
                )
            self._pdk_index[entry.module_name] = entry
            for alias in entry.aliases:
                self._pdk_alias_map[alias] = entry.module_name

    def _resolve_pdk(self, component_type: str) -> tuple[PDKEntry | None, str | None]:
        """Try to resolve *component_type* against the PDK index.

        Returns ``(entry, None)`` on success or ``(None, error_msg)`` on failure.
        The error message includes suggestions for fuzzy matches.
        """
        if self._pdk_index is None:
            return None, None  # no catalog → skip validation

        ct_lower = component_type.lower().strip()

        # Exact module name match
        if component_type in self._pdk_index:
            return self._pdk_index[component_type], None

        # Case-insensitive module match
        for mod, entry in self._pdk_index.items():
            if mod.lower() == ct_lower:
                return entry, None

        # Alias match
        if self._pdk_alias_map and ct_lower in self._pdk_alias_map:
            mod = self._pdk_alias_map[ct_lower]
            return self._pdk_index[mod], None

        # Fuzzy: substring matching
        candidates = [
            mod for mod in self._pdk_index
            if ct_lower in mod.lower() or mod.lower() in ct_lower
        ]
        if not candidates:
            candidates = [
                mod for mod, entry in self._pdk_index.items()
                if ct_lower in entry.name.lower()
            ]

        if candidates:
            suggestions = ", ".join(candidates[:5])
            return None, (
                f"Component '{component_type}' not found in PDK. "
                f"Did you mean one of: {suggestions}? "
                f"Use search_pdk to find the correct module name."
            )
        return None, (
            f"Component '{component_type}' not found in PDK. "
            f"Use search_pdk to find available components."
        )

    # -- Mutation helpers ---------------------------------------------------

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"C{self._id_counter}"

    def _clear_highlights(self) -> None:
        self._last_added_ids.clear()
        self._last_added_edges.clear()

    # ======================================================================
    # add_component
    # ======================================================================

    def add_component(
        self,
        component_type: str,
        port_config: str,
        stage: int,
        lane: int,
        role: Optional[str] = None,
        sub_type: Optional[str] = None,
        description: str = "",
        specs: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Add a component at a specific (stage, lane) position.

        Returns a dict with the assigned id and port map, or an error.
        When a PDK catalog was provided at init, validates ``component_type``
        against the catalog and resolves the canonical module name and port
        config automatically.
        """
        self._clear_highlights()

        # --- PDK validation (when catalog is loaded) -----------------------
        resolved_name = component_type
        pdk_entry: PDKEntry | None = None

        if self._pdk_index is not None:
            pdk_entry, err = self._resolve_pdk(component_type)
            if err:
                return {"status": "error", "message": err}
            if pdk_entry is not None:
                resolved_name = pdk_entry.module_name
                # Auto-correct port_config from PDK if the one given doesn't
                # match the PDK's canonical value
                if pdk_entry.port_config != "unknown" and port_config != pdk_entry.port_config:
                    return {
                        "status": "error",
                        "message": (
                            f"port_config '{port_config}' does not match PDK for "
                            f"'{resolved_name}' (expected '{pdk_entry.port_config}'). "
                            f"Use port_config='{pdk_entry.port_config}'."
                        ),
                    }

        # --- Format validation ---------------------------------------------
        if not _PORT_CONFIG_RE.match(port_config):
            return {"status": "error",
                    "message": f"Invalid port_config '{port_config}': must match NxM (e.g. '2x2')"}

        # --- Position validation -------------------------------------------
        pos = Position(stage, lane)
        key = pos.as_tuple()
        if key in self._position_map:
            occupant = self._position_map[key]
            return {"status": "error",
                    "message": f"Position ({stage}, {lane}) already occupied by {occupant}"}

        cid = self._next_id()
        ports = _make_ports(port_config)
        node = ComponentNode(
            id=cid,
            component_type=resolved_name,
            sub_type=sub_type,
            port_config=port_config,
            role=role,
            description=description,
            position=pos,
            ports=ports,
            specs=specs or {},
        )
        if pdk_entry is not None:
            node.pdk_name = pdk_entry.name
        self._components[cid] = node
        self._position_map[key] = cid
        self._last_added_ids.add(cid)

        result: dict[str, Any] = {
            "status": "ok",
            "id": cid,
            "component_type": resolved_name,
            "position": {"stage": stage, "lane": lane},
            "ports": {
                name: {
                    "direction": p.direction,
                    "connected": False,
                    "physical_lane": _port_physical_lane(lane, name, port_config),
                }
                for name, p in ports.items()
            },
        }
        if pdk_entry is not None:
            result["pdk_name"] = pdk_entry.name

        # Provide wiring hints: show open ports on nearby components that
        # could be connected to this new component's input ports.
        connectable = self._find_connectable_ports(cid)
        if connectable:
            result["connectable"] = connectable

        return result

    def _find_connectable_ports(self, target_id: str) -> list[dict]:
        """Find open output ports on adjacent-stage components that can feed into *target_id*'s inputs.

        Uses physical lane alignment to recommend the best connections first.
        """
        target = self._components.get(target_id)
        if not target:
            return []
        target_stage = target.position.stage
        target_inputs = [
            p for p in target.ports.values()
            if p.direction == "input" and p.connected_to is None
        ]
        if not target_inputs:
            return []

        # Build a map: physical_lane -> target input port name
        input_by_phys_lane: dict[int, str] = {}
        for p in target_inputs:
            pl = _port_physical_lane(target.position.lane, p.name, target.port_config)
            input_by_phys_lane[pl] = p.name

        hints: list[dict] = []
        for node in self._components.values():
            if node.id == target_id:
                continue
            stage_diff = target_stage - node.position.stage
            if stage_diff < 0 or stage_diff > 1:
                continue
            for p in node.ports.values():
                if p.direction == "output" and p.connected_to is None:
                    src_pl = _port_physical_lane(
                        node.position.lane, p.name, node.port_config)
                    # Find matching input on the same physical lane
                    matched_input = input_by_phys_lane.get(src_pl)
                    if matched_input:
                        hints.append({
                            "from": f"{node.id}.{p.name}",
                            "to": f"{target_id}.{matched_input}",
                            "physical_lane": src_pl,
                            "alignment": "same_lane",
                        })
                    else:
                        # No lane-aligned input; suggest all open inputs
                        all_inputs = [tp.name for tp in target_inputs]
                        hints.append({
                            "from": f"{node.id}.{p.name}",
                            "to_candidates": [f"{target_id}.{ip}" for ip in all_inputs],
                            "physical_lane": src_pl,
                            "alignment": "cross_lane",
                        })

        # Sort: same_lane hints first, then cross_lane
        hints.sort(key=lambda h: (0 if h.get("alignment") == "same_lane" else 1))
        return hints[:8]

    # ======================================================================
    # connect
    # ======================================================================

    def connect(
        self,
        from_component: str,
        from_port: str,
        to_component: str,
        to_port: str,
        description: str = "",
        feedback: bool = False,
        skip: bool = False,
    ) -> dict[str, Any]:
        """Connect two ports. Enforces forward flow and stage adjacency."""
        self._clear_highlights()

        # Component existence
        if from_component not in self._components:
            return {"status": "error",
                    "message": f"Component {from_component} does not exist"}
        if to_component not in self._components:
            return {"status": "error",
                    "message": f"Component {to_component} does not exist"}

        src = self._components[from_component]
        dst = self._components[to_component]

        # Self-connection guard
        if from_component == to_component:
            return {"status": "error",
                    "message": f"Cannot connect {from_component} to itself. "
                               f"Connect to a different component instead."}

        # Port existence
        if from_port not in src.ports:
            avail = ", ".join(sorted(src.ports.keys()))
            return {"status": "error",
                    "message": f"Port {from_port} does not exist on {from_component} "
                               f"(available: {avail})"}
        if to_port not in dst.ports:
            avail = ", ".join(sorted(dst.ports.keys()))
            return {"status": "error",
                    "message": f"Port {to_port} does not exist on {to_component} "
                               f"(available: {avail})"}

        src_port = src.ports[from_port]
        dst_port = dst.ports[to_port]

        # Double connection
        if src_port.connected_to is not None:
            return {"status": "error",
                    "message": f"Port {from_component}.{from_port} is already "
                               f"connected to {src_port.connected_to}"}
        if dst_port.connected_to is not None:
            return {"status": "error",
                    "message": f"Port {to_component}.{to_port} is already "
                               f"connected to {dst_port.connected_to}"}

        # Forward flow
        if not feedback and src.position.stage > dst.position.stage:
            return {"status": "error",
                    "message": f"Backward connection from stage {src.position.stage} "
                               f"to stage {dst.position.stage}. "
                               f"Use feedback=true for intentional feedback paths."}

        # Stage adjacency (feedback connections are exempt)
        stage_diff = abs(dst.position.stage - src.position.stage)
        if not skip and not feedback and stage_diff > 1:
            return {"status": "error",
                    "message": f"Connection spans {stage_diff} stages "
                               f"(from stage {src.position.stage} to {dst.position.stage}). "
                               f"Direct connections must be between adjacent stages. "
                               f"Use skip=true if intentional."}

        # Direction checks
        warnings: list[str] = []
        if src_port.direction == "input" and dst_port.direction == "input":
            warnings.append("Both ports are inputs — unusual connection.")
        if src_port.direction == "output" and dst_port.direction == "output":
            return {
                "status": "error",
                "message": (
                    f"Cannot connect two output ports: "
                    f"{from_component}.{from_port} (output) -> "
                    f"{to_component}.{to_port} (output). "
                    f"Connect an output to an input instead."
                ),
            }

        # Lane proximity warning
        lane_diff = abs(src.position.lane - dst.position.lane)
        if lane_diff > 3:
            warnings.append(f"Connection spans {lane_diff} lanes.")

        # Commit
        src_port.connected_to = f"{to_component}.{to_port}"
        dst_port.connected_to = f"{from_component}.{from_port}"
        edge = ConnectionEdge(from_id=from_component, from_port=from_port,
                              to_id=to_component, to_port=to_port,
                              description=description)
        self._connections.append(edge)
        self._last_added_edges.add((from_component, from_port, to_component, to_port))

        result: dict[str, Any] = {
            "status": "connected",
            "edge": f"{from_component}.{from_port} -> {to_component}.{to_port}",
        }
        if warnings:
            result["warnings"] = warnings
        return result

    # ======================================================================
    # get_open_ports
    # ======================================================================

    def get_open_ports(self) -> dict[str, Any]:
        """Return all unconnected ports grouped by stage."""
        by_stage: dict[int, list[dict]] = defaultdict(list)
        for node in self._components.values():
            for pname, pinfo in node.ports.items():
                if pinfo.connected_to is None:
                    by_stage[node.position.stage].append({
                        "component": node.id,
                        "port": pname,
                        "direction": pinfo.direction,
                        "physical_lane": _port_physical_lane(
                            node.position.lane, pname, node.port_config),
                        "component_type": node.component_type,
                        "position": {"stage": node.position.stage,
                                     "lane": node.position.lane},
                        "description": node.description,
                    })

        furthest = max(by_stage.keys()) if by_stage else 0
        total_open = sum(len(v) for v in by_stage.values())

        stage_summary_parts: list[str] = []
        for s in sorted(by_stage.keys()):
            ports = by_stage[s]
            lanes = {p["position"]["lane"] for p in ports}
            stage_summary_parts.append(
                f"{len(ports)} open port(s) at stage {s} across {len(lanes)} lane(s)"
            )

        return {
            "open_ports_by_stage": {str(s): ports for s, ports in sorted(by_stage.items())},
            "total_components": len(self._components),
            "total_connections": len(self._connections),
            "furthest_stage": furthest,
            "summary": "; ".join(stage_summary_parts) if stage_summary_parts else "No open ports",
        }

    # ======================================================================
    # get_state
    # ======================================================================

    def get_state(self) -> dict[str, Any]:
        """Return a compact summary of the entire circuit."""
        type_counts: dict[str, int] = defaultdict(int)
        grid: dict[str, dict[str, dict]] = defaultdict(dict)
        components_list: list[dict] = []

        for node in self._components.values():
            type_counts[node.component_type] += 1
            grid[str(node.position.stage)][str(node.position.lane)] = {
                "id": node.id,
                "type": node.component_type,
                "ports": node.port_config,
            }
            port_map: dict[str, str] = {}
            for pname, pinfo in node.ports.items():
                if pinfo.connected_to:
                    port_map[pname] = f"-> {pinfo.connected_to}"
                else:
                    port_map[pname] = "open"
            components_list.append({
                "id": node.id,
                "type": node.component_type,
                "port_config": node.port_config,
                "position": {"stage": node.position.stage,
                             "lane": node.position.lane},
                "ports": port_map,
            })

        open_in = sum(
            1 for n in self._components.values()
            for p in n.ports.values()
            if p.direction == "input" and p.connected_to is None
        )
        open_out = sum(
            1 for n in self._components.values()
            for p in n.ports.values()
            if p.direction == "output" and p.connected_to is None
        )

        return {
            "total_components": len(self._components),
            "total_connections": len(self._connections),
            "open_input_ports": open_in,
            "open_output_ports": open_out,
            "component_counts": dict(type_counts),
            "grid": dict(grid),
            "components": components_list,
        }

    # ======================================================================
    # replicate_stage
    # ======================================================================

    def replicate_stage(
        self,
        source_components: list[str],
        count: int,
        connect_from: list[str],
        connection_rule: str,
        start_stage: Optional[int] = None,
        description: str = "",
    ) -> dict[str, Any]:
        """Deterministically replicate a set of components.

        ``count`` is the multiplier: for each source component, ``count``
        copies are created, giving ``len(source_components) * count`` new
        components total.

        ``connection_rule``:
          - ``one_to_one``: zip ``connect_from`` ports with new component
            inputs in order.
          - ``broadcast``: each ``connect_from`` port feeds one copy.
          - ``chain``: connect output of copy *i* to input of copy *i+1*.
        """
        self._clear_highlights()

        # Validate sources
        templates: list[ComponentNode] = []
        for sid in source_components:
            if sid not in self._components:
                return {"status": "error",
                        "message": f"Source component {sid} does not exist"}
            templates.append(self._components[sid])

        if count < 1:
            return {"status": "error", "message": "count must be >= 1"}
        if connection_rule not in ("one_to_one", "broadcast", "chain"):
            return {"status": "error",
                    "message": f"Unknown connection_rule '{connection_rule}'. "
                               f"Use one_to_one, broadcast, or chain."}

        # Parse connect_from port references
        from_ports: list[tuple[str, str]] = []
        for ref in connect_from:
            parts = ref.split(".")
            if len(parts) != 2:
                return {"status": "error",
                        "message": f"Invalid port reference '{ref}': use 'C1.o2' format"}
            cid, pname = parts
            if cid not in self._components:
                return {"status": "error",
                        "message": f"Component {cid} in connect_from does not exist"}
            if pname not in self._components[cid].ports:
                return {"status": "error",
                        "message": f"Port {pname} does not exist on {cid}"}
            if self._components[cid].ports[pname].connected_to is not None:
                return {"status": "error",
                        "message": f"Port {cid}.{pname} is already connected"}
            from_ports.append((cid, pname))

        # Determine placement stage
        if start_stage is not None:
            place_stage = start_stage
        elif from_ports:
            place_stage = max(self._components[c].position.stage for c, _ in from_ports) + 1
        else:
            place_stage = (max(n.position.stage for n in self._components.values()) + 1
                          if self._components else 0)

        # Find first available lane at that stage
        occupied_lanes = {
            lane for (s, lane), _ in self._position_map.items() if s == place_stage
        }
        next_lane = 0
        while next_lane in occupied_lanes:
            next_lane += 1

        total_copies = len(templates) * count
        created_ids: list[str] = []
        created_positions: list[dict] = []

        for _copy_idx in range(count):
            for tmpl in templates:
                while next_lane in occupied_lanes:
                    next_lane += 1
                result = self.add_component(
                    component_type=tmpl.component_type,
                    port_config=tmpl.port_config,
                    stage=place_stage,
                    lane=next_lane,
                    role=tmpl.role,
                    sub_type=tmpl.sub_type,
                    description=tmpl.description if not description else description,
                    specs=dict(tmpl.specs) if tmpl.specs else None,
                )
                if result["status"] != "ok":
                    return result
                created_ids.append(result["id"])
                created_positions.append(result["position"])
                occupied_lanes.add(next_lane)
                next_lane += 1

        # Apply connection rule
        connections_made = 0

        if connection_rule == "one_to_one":
            for i, (src_cid, src_port) in enumerate(from_ports):
                if i >= len(created_ids):
                    break
                new_cid = created_ids[i]
                new_node = self._components[new_cid]
                first_input = next(
                    (p.name for p in new_node.ports.values() if p.direction == "input"),
                    None,
                )
                if first_input:
                    res = self.connect(src_cid, src_port, new_cid, first_input,
                                       description=f"replicate_stage one_to_one")
                    if res["status"] == "connected":
                        connections_made += 1

        elif connection_rule == "broadcast":
            for i, (src_cid, src_port) in enumerate(from_ports):
                if i >= len(created_ids):
                    break
                new_cid = created_ids[i]
                new_node = self._components[new_cid]
                first_input = next(
                    (p.name for p in new_node.ports.values() if p.direction == "input"),
                    None,
                )
                if first_input:
                    res = self.connect(src_cid, src_port, new_cid, first_input,
                                       description=f"replicate_stage broadcast")
                    if res["status"] == "connected":
                        connections_made += 1

        elif connection_rule == "chain":
            # Wire connect_from to first component, then chain outputs to next inputs
            if from_ports and created_ids:
                src_cid, src_port = from_ports[0]
                new_node = self._components[created_ids[0]]
                first_input = next(
                    (p.name for p in new_node.ports.values() if p.direction == "input"),
                    None,
                )
                if first_input:
                    res = self.connect(src_cid, src_port, created_ids[0], first_input,
                                       description="replicate_stage chain start")
                    if res["status"] == "connected":
                        connections_made += 1

            for i in range(len(created_ids) - 1):
                curr_node = self._components[created_ids[i]]
                next_node = self._components[created_ids[i + 1]]
                first_output = next(
                    (p.name for p in curr_node.ports.values()
                     if p.direction == "output" and p.connected_to is None),
                    None,
                )
                first_input = next(
                    (p.name for p in next_node.ports.values() if p.direction == "input"),
                    None,
                )
                if first_output and first_input:
                    res = self.connect(created_ids[i], first_output,
                                       created_ids[i + 1], first_input,
                                       description="replicate_stage chain link",
                                       skip=True)
                    if res["status"] == "connected":
                        connections_made += 1

        # Collect open ports on new components
        new_open: list[dict] = []
        for cid in created_ids:
            node = self._components[cid]
            for pname, pinfo in node.ports.items():
                if pinfo.connected_to is None:
                    new_open.append({
                        "component": cid, "port": pname,
                        "direction": pinfo.direction,
                        "position": {"stage": node.position.stage,
                                     "lane": node.position.lane},
                    })

        self._last_added_ids = set(created_ids)
        return {
            "status": "ok",
            "created": created_ids,
            "positions": created_positions,
            "connections_made": connections_made,
            "open_ports": new_open,
        }

    # ======================================================================
    # finalize
    # ======================================================================

    def finalize(
        self,
        title: str,
        brief_summary: str,
        architecture_type: Optional[str] = None,
        n_value: Optional[int] = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Signal construction complete. Run Tier 2 validation checks.

        If ``force`` is False (default) and the circuit has multiple components
        but very few connections, finalize is rejected with a message telling
        the caller to wire the circuit first.
        """
        n_comp = len(self._components)
        n_conn = len(self._connections)

        # Reject premature finalize: a multi-component circuit should have at
        # least roughly (n_comp - 1) connections to form a spanning tree.
        if not force and n_comp > 2 and n_conn < (n_comp - 1) * 0.5:
            # Gather the most useful info about what's unconnected
            disconnected = []
            for node in self._components.values():
                has_conn = any(p.connected_to is not None for p in node.ports.values())
                if not has_conn:
                    disconnected.append(node.id)
            return {
                "status": "error",
                "message": (
                    f"Circuit is severely under-connected: {n_comp} components "
                    f"but only {n_conn} connection(s). Expected at least "
                    f"{n_comp - 1}. {len(disconnected)} component(s) have no "
                    f"connections at all: {', '.join(disconnected[:10])}. "
                    f"Wire the circuit using connect() before calling finalize. "
                    f"Call get_open_ports() to see what needs wiring."
                ),
            }

        self._title = title
        self._summary = brief_summary
        self._architecture_type = architecture_type
        self._n_value = n_value
        self._finalized = True

        warnings: list[str] = []

        # Check for disconnected components (zero connections)
        for node in self._components.values():
            has_connection = any(
                p.connected_to is not None for p in node.ports.values()
            )
            if not has_connection and len(self._components) > 1:
                warnings.append(f"Component {node.id} has no connections.")

        # Check for disconnected subgraphs (BFS from first component)
        if len(self._components) > 1:
            adj: dict[str, set[str]] = defaultdict(set)
            for edge in self._connections:
                adj[edge.from_id].add(edge.to_id)
                adj[edge.to_id].add(edge.from_id)
            start = next(iter(self._components))
            visited: set[str] = set()
            queue = [start]
            while queue:
                curr = queue.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                queue.extend(adj[curr] - visited)
            unreachable = set(self._components.keys()) - visited
            if unreachable:
                warnings.append(
                    f"Disconnected subgraph: {', '.join(sorted(unreachable))} "
                    f"not reachable from {start}."
                )

        # Count open ports
        open_in = sum(
            1 for n in self._components.values()
            for p in n.ports.values()
            if p.direction == "input" and p.connected_to is None
        )
        open_out = sum(
            1 for n in self._components.values()
            for p in n.ports.values()
            if p.direction == "output" and p.connected_to is None
        )

        if open_in == 0 and len(self._components) > 0:
            warnings.append("No open input ports — circuit has no external inputs.")
        if open_out == 0 and len(self._components) > 0:
            warnings.append("No open output ports — circuit has no external outputs.")

        return {
            "status": "ok" if not warnings else "warning",
            "total_components": len(self._components),
            "total_connections": len(self._connections),
            "open_input_ports": open_in,
            "open_output_ports": open_out,
            "warnings": warnings,
            "design_intent_ready": True,
        }

    # ======================================================================
    # Serialization
    # ======================================================================

    def to_design_intent(self) -> DesignIntent:
        """Convert the graph to a standard DesignIntent."""
        components = [
            ComponentIntent(
                id=node.id,
                description=node.description,
                port_config=node.port_config,
                role=node.role,
                component_type=node.component_type,
                sub_type=node.sub_type,
                pdk_module=node.pdk_name,
                specs=[SpecEntry(key=k, value=str(v)) for k, v in node.specs.items()],
                confidence=1.0,
            )
            for node in self._components.values()
        ]
        connections = [
            Connection(
                from_component=edge.from_id,
                to_component=edge.to_id,
                from_port=edge.from_port,
                to_port=edge.to_port,
                description=edge.description,
                confidence=1.0,
            )
            for edge in self._connections
        ]
        return DesignIntent(
            title=self._title or "Untitled Circuit",
            brief_summary=self._summary or "",
            components=components,
            connections=connections,
            architecture_type=self._architecture_type,
            n_value=self._n_value,
            ambiguities=list(self._ambiguities),
        )

    def to_dot(self, highlight: bool = True) -> str:
        """Generate a Graphviz DOT string with record-shaped nodes showing ports.

        Nodes display the component ID, PDK module name (or type), and
        input/output ports on the left/right sides.  Edge labels show port
        names.  When ``highlight`` is True, recently-added elements are
        coloured for live visualization.
        """
        lines = [
            "digraph G {",
            "    rankdir=LR;",
            "    node [shape=record, style=rounded, fontsize=10];",
            "    edge [fontsize=8];",
            "",
        ]

        # Group by stage for rank constraints
        stages: dict[int, list[ComponentNode]] = defaultdict(list)
        for node in self._components.values():
            stages[node.position.stage].append(node)

        for stage_num in sorted(stages.keys()):
            nodes = stages[stage_num]
            ids = " ".join(n.id for n in nodes)
            lines.append(f"    {{ rank=same; {ids}; }}")

        lines.append("")

        # Nodes — record label with correct port spatial ordering.
        # GDSFactory convention: for 2x2, o2=top-left, o1=bottom-left,
        # o3=top-right, o4=bottom-right.  In Graphviz records the first
        # item in {a|b} renders on top, so we reverse inputs to get the
        # higher-numbered port on top.
        for node in self._components.values():
            in_ports = [p for p in node.ports.values() if p.direction == "input"]
            out_ports = [p for p in node.ports.values() if p.direction == "output"]

            # Reverse inputs so higher-numbered port is on top (o2 above o1)
            in_ports_ordered = list(reversed(in_ports))
            in_section = "|".join(f"<{p.name}>{p.name}" for p in in_ports_ordered) if in_ports else ""
            out_section = "|".join(f"<{p.name}>{p.name}" for p in out_ports) if out_ports else ""

            display_name = node.pdk_name or node.component_type
            center = f"{node.id}\\n{display_name}\\n({node.port_config})"

            if in_section and out_section:
                label = f"{{{{{in_section}}}|{center}|{{{out_section}}}}}"
            elif in_section:
                label = f"{{{{{in_section}}}|{center}}}"
            elif out_section:
                label = f"{{{center}|{{{out_section}}}}}"
            else:
                label = center

            attrs = f'label="{label}"'
            if highlight and node.id in self._last_added_ids:
                attrs += ', style="rounded,filled", fillcolor=lightblue'
            lines.append(f'    {node.id} [{attrs}];')

        lines.append("")

        # Edges — directed, with port labels
        for edge in self._connections:
            edge_attrs: list[str] = []
            edge_attrs.append(f'taillabel="{edge.from_port}"')
            edge_attrs.append(f'headlabel="{edge.to_port}"')
            if highlight and (edge.from_id, edge.from_port,
                              edge.to_id, edge.to_port) in self._last_added_edges:
                edge_attrs.append("penwidth=2.5")
                edge_attrs.append("color=steelblue")
            attr_str = ", ".join(edge_attrs)
            lines.append(
                f"    {edge.from_id}:{edge.from_port} -> "
                f"{edge.to_id}:{edge.to_port} [{attr_str}];"
            )

        lines.append("}")
        return "\n".join(lines)
