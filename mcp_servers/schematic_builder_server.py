"""
MCP Server: Schematic Builder
DOT graph generation, layout computation, and circuit DSL assembly tools.

Run with: python mcp_servers/schematic_builder_server.py
"""

import json
import re

import pygraphviz as pgv
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "PhIDO Schematic Builder",
    instructions="Build and validate photonic circuit schematics from circuit DSLs.",
)

@mcp.tool()
def circuit_dsl_to_dot(circuit_dsl_json: str) -> str:
    """Convert a circuit DSL (JSON) into a DOT graph string with port-level nodes.

    The circuit DSL must have:
      - "doc": {"name": "..."} or "doc": {"title": "..."}
      - "nodes": {"C1": {"component": "mzi_foo", "properties": {"ports": "2x2"}}, ...}

    Returns a DOT graph string using Graphviz record shapes with port labels.

    Args:
        circuit_dsl_json: JSON string of the circuit DSL dict.
    """
    try:
        circuit_dsl = json.loads(circuit_dsl_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    doc = circuit_dsl.get("doc", {})
    graph_name = doc.get("name", doc.get("title", "circuit"))
    graph_name = re.sub(r"[^a-zA-Z0-9_]", "_", graph_name)
    if graph_name and graph_name[0].isdigit():
        graph_name = f"_{graph_name}"

    nodes = circuit_dsl.get("nodes", {})
    dot_lines = [f"graph {graph_name} {{", "  rankdir=LR;", "  node [shape=record];"]

    for node_name, node_info in nodes.items():
        if isinstance(node_info, str):
            component_name = node_info
            ports_info = ""
            display_name = f"{node_name}: {component_name}"
        else:
            component_name = node_info.get("component", "")
            ports_info = node_info.get("properties", {}).get("ports", "")
            display_name = node_info.get("label", f"{node_name}: {component_name}")

        display_name = display_name.replace('"', '\\"')

        port_match = re.match(r"^(\d+)x(\d+)$", ports_info.strip()) if ports_info else None
        if port_match:
            inp, out = int(port_match.group(1)), int(port_match.group(2))
            input_labels = "|".join(
                f"<o{i}> o{i}" for i in range(inp, 0, -1)
            ) if inp > 0 else ""
            output_labels = "|".join(
                f"<o{i}> o{i}" for i in range(inp + 1, inp + out + 1)
            ) if out > 0 else ""

            if input_labels and output_labels:
                label = f"{{{{{input_labels}}} | {display_name} | {{{output_labels}}}}}"
            elif input_labels:
                label = f"{{{{{input_labels}}} | {display_name} }}"
            elif output_labels:
                label = f"{{ {display_name} | {{{output_labels}}}}}"
            else:
                label = display_name
        else:
            label = display_name

        dot_lines.append(f'  {node_name} [label="{label}"];')

    dot_lines.append("}")
    return "\n".join(dot_lines)

@mcp.tool()
def check_planarity(dot_string: str) -> str:
    """Check whether a DOT graph has crossing edges after layout.

    Runs the Graphviz 'dot' layout engine, extracts edge start/end
    coordinates, and checks all pairs for geometric intersection.

    Args:
        dot_string: A valid Graphviz DOT graph string (with edges).
    """
    try:
        lines = dot_string.strip().splitlines()
        if lines and lines[0].strip() == "dot":
            dot_string = "\n".join(lines[1:])

        graph = pgv.AGraph(string=dot_string)
        graph.layout(prog="dot")

        edges = []
        for edge in graph.edges():
            points = edge.attr["pos"].split()
            start = tuple(map(float, points[0].split(",")))
            end = tuple(map(float, points[-1].split(",")))
            edges.append((start, end))

        def _do_intersect(p1, q1, p2, q2):
            def _orient(p, q, r):
                val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
                if val == 0:
                    return 0
                return 1 if val > 0 else 2

            def _on_seg(p, q, r):
                return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0])
                        and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

            o1, o2 = _orient(p1, q1, p2), _orient(p1, q1, q2)
            o3, o4 = _orient(p2, q2, p1), _orient(p2, q2, q1)
            if o1 != o2 and o3 != o4:
                return True
            if o1 == 0 and _on_seg(p1, p2, q1):
                return True
            if o2 == 0 and _on_seg(p1, q2, q1):
                return True
            if o3 == 0 and _on_seg(p2, p1, q2):
                return True
            if o4 == 0 and _on_seg(p2, q1, q2):
                return True
            return False

        crossings = []
        for i, (p1, q1) in enumerate(edges):
            for j, (p2, q2) in enumerate(edges):
                if i < j and _do_intersect(p1, q1, p2, q2):
                    crossings.append({"edge_a": i, "edge_b": j})

        return json.dumps({
            "planar": len(crossings) == 0,
            "num_crossings": len(crossings),
            "crossings": crossings[:10],
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Planarity check failed: {e}"})
    
@mcp.tool()
def compute_layout(dot_string: str, footprints_json: str) -> str:
    """Compute x,y placements for components using Graphviz layout.

    Takes a DOT graph and a footprints dict (component_id → [dx, dy] in
    microns), injects physical sizes into the DOT nodes, runs the 'dot'
    layout engine, and returns center-point coordinates for each node.

    Args:
        dot_string: Graphviz DOT string (with or without edges).
        footprints_json: JSON object mapping node IDs to [dx_um, dy_um] arrays.
                         Example: {"C1": [300.5, 50.2], "C2": [100.0, 30.0]}
    """
    try:
        footprints = json.loads(footprints_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid footprints JSON: {e}"})

    try: 
        # Convert microns → inches (Graphviz uses inches)
        MICRON_TO_INCH = 0.01
        PADDING_FACTOR = 1.2
        PADDING_MARGIN = 1  # inches

        scaled = {
            node: (dx * MICRON_TO_INCH, dy * MICRON_TO_INCH)
            for node, (dx, dy) in footprints.items()
        }

        # Inject sizes into DOT
        lines = dot_string.split("\n")
        output_lines = []
        nodes_added = set()
        for line in lines:
            stripped = line.strip()
            if "label=" in stripped and stripped.endswith("];"):
                node_name = stripped.split("[")[0].strip()
                if node_name in scaled and node_name not in nodes_added:
                    w, h = scaled[node_name]
                    output_lines.append(
                        f"  {node_name} [width={w * PADDING_FACTOR + PADDING_MARGIN}, "
                        f"height={h * PADDING_FACTOR + PADDING_MARGIN}, "
                        f"shape=record, fixedsize=true];"
                    )
                    nodes_added.add(node_name)
            output_lines.append(line)

        sized_dot = "\n".join(output_lines)

        # Run layout
        graph = pgv.AGraph(string=sized_dot)
        graph.graph_attr["nodesep"] = "0.15"
        graph.graph_attr["ranksep"] = "0.25"
        graph.layout(prog="dot")

        # Extract positions (center of node).
        # Graphviz returns positions in points (1/72 inch). Node sizes were
        # injected in inches (microns * 0.01), so the coordinate space is
        # effectively points.  GDSFactory placements are in microns, so we
        # convert:  points * (100/72)  ≈  microns.
        POINTS_TO_MICRONS = 100.0 / 72.0
        positions = {}
        for node in graph.nodes():
            pos = node.attr.get("pos")
            if pos:
                x, y = map(float, pos.split(","))
                positions[str(node)] = {
                    "x": round(x * POINTS_TO_MICRONS, 3),
                    "y": round(y * POINTS_TO_MICRONS, 3),
                }

        return json.dumps({
            "positions": positions,
            "sized_dot": sized_dot,
        }, indent=2)

    except Exception as e:
        return json.dumps({"error": f"Layout computation failed: {e}"})
    
@mcp.tool()
def find_open_ports(dot_string: str) -> str:
    """Find unconnected ports in a DOT graph.

    Parses the DOT string for record-label port definitions (<o1>, <o2>, ...)
    and edge connections (C1:o1 -- C2:o3), then returns ports with no edge.

    Args:
        dot_string: Graphviz DOT string with record-shaped nodes and edges.
    """
    edges = []
    nodes_ports = {}

    for line in dot_string.splitlines():
        line = line.strip()
        if "--" in line:
            edge = line.strip(";").split(" -- ")
            edges.append(tuple(e.strip() for e in edge))
        elif '[label="' in line:
            node = line.split()[0]
            label = re.search(r'\[label="(.*)"\]', line)
            if label:
                ports = re.findall(r"<(o\d+)>", label.group(1))
                nodes_ports[node] = ports

    connected = set()
    for edge in edges:
        for endpoint in edge:
            if ":" in endpoint:
                connected.add(endpoint)

    open_ports = []
    for node, ports in nodes_ports.items():
        for port in ports:
            if f"{node}:{port}" not in connected:
                open_ports.append(f"{node}:{port}")

    # Also build the circuit-port mapping (o1, o2, ... for GDS export)
    port_dict = {f"o{i+1}": op.replace(":", ",") for i, op in enumerate(open_ports)}

    return json.dumps({
        "open_ports": open_ports,
        "circuit_ports": port_dict,
        "total_ports": sum(len(p) for p in nodes_ports.values()),
        "connected_count": len(connected),
    }, indent=2)
    
@mcp.tool()
def export_gf_netlist(circuit_dsl_json: str) -> str:
    """Convert a complete circuit DSL into a GDSFactory-compatible YAML netlist.

    The circuit DSL must be fully enriched: each node needs 'component',
    'properties', 'params', and 'placement'. Edges and ports must be present.

    Args:
        circuit_dsl_json: JSON string of the fully enriched circuit DSL.
    """
    try:
        circuit_dsl = json.loads(circuit_dsl_json)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    try:
        instances = {}
        for node_id, info in circuit_dsl["nodes"].items():
            instances[node_id] = {
                "component": info["component"],
                "info": info.get("properties", {}),
                "settings": info.get("params", {}),
            }

        # Group links by component pair so route_bundle handles each
        # pair independently instead of tangling all routes together.
        pair_links: dict[tuple[str, str], dict[str, str]] = {}
        for _edge_id, edge_info in circuit_dsl.get("edges", {}).items():
            link = edge_info["link"]
            source, target = link.split(": ")
            src_inst = source.split(",")[0]
            tgt_inst = target.split(",")[0]
            pair_key = (min(src_inst, tgt_inst), max(src_inst, tgt_inst))
            pair_links.setdefault(pair_key, {})[source] = target

        routes = {}
        route_settings = {
            "cross_section": "strip",
            "on_collision": "error",
            "sort_ports": True,
        }
        for i, ((inst_a, inst_b), links) in enumerate(pair_links.items()):
            routes[f"optical_{inst_a}_{inst_b}"] = {
                "settings": route_settings,
                "links": links,
            }

        placements = {}
        for node_id, info in circuit_dsl["nodes"].items():
            pl = info.get("placement", {})
            placements[node_id] = {
                "x": pl.get("x", 0),
                "y": pl.get("y", 0),
                "rotation": pl.get("rotation", 0),
            }

        gf_netlist = {
            "instances": instances,
            "routes": routes,
            "placements": placements,
            "ports": circuit_dsl.get("ports", {}),
        }

        import yaml
        return yaml.dump(gf_netlist, default_flow_style=False, sort_keys=False)

    except KeyError as e:
        return json.dumps({"error": f"Missing required field in circuit DSL: {e}"})
    except Exception as e:
        return json.dumps({"error": f"GF netlist export failed: {e}"})
    
if __name__ == "__main__":
    mcp.run()