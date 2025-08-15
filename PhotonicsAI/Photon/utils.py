"""Utility functions for the PhotonicsAI package."""

import ast
import glob
import os
import pathlib
import re

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygraphviz as pgv
from sax.saxtypes import Float, Model

from PhotonicsAI.config import PATH


def extract_docstring(file_path):
    """Extracts the module-level docstring of a Python file."""
    with open(file_path, encoding="utf-8") as file:
        module_content = file.read()

    parsed_module = ast.parse(module_content)

    # Extract the module-level docstring
    docstring = ast.get_docstring(parsed_module)

    # Extract the module name from the file path
    module_name = os.path.basename(file_path).replace(".py", "")

    return {"module_name": module_name, "docstring": docstring, "file_path": file_path}


def search_directory_for_docstrings(directory=PATH.pdk):
    """Uses glob to find Python files and extract their docstrings."""
    all_docs = []
    # Recursive glob pattern to find all Python files
    file_paths = glob.glob(f"{directory}/*.py", recursive=True)
    filtered_paths = [
        path for path in file_paths if not os.path.basename(path) == "__init__.py"
    ]

    for file_path in filtered_paths:
        doc = extract_docstring(file_path)
        all_docs.append(doc)

    sorted_all_docs = sorted(all_docs, key=lambda x: x["module_name"])
    return sorted_all_docs


def circuit_to_dot(circuit_dsl):
    """Converts a circuit DSL to a DOT graph string."""
    # data = yaml.safe_load(yaml_str)

    if "name" in circuit_dsl["doc"]:
        graph_name = circuit_dsl["doc"]["name"]
    else:
        graph_name = "graph_name_placeholder"
    nodes = circuit_dsl.get("nodes", {})

    dot_lines = [f"graph {graph_name} {{", "  rankdir=LR;", "  node [shape=record];"]

    for node_name, node_info in nodes.items():
        # Handle both string and dictionary node_info
        if isinstance(node_info, str):
            component_name = node_info
            ports_info = ""
        else:
            component_name = node_info.get("component", "")
            ports_info = node_info.get("properties", {}).get("ports", "")

        if "x" in ports_info:
            input_ports, output_ports = map(int, ports_info.split("x"))

            # Generate input and output port labels
            input_labels = (
                "|".join([f"<o{i}> o{i}" for i in range(input_ports, 0, -1)])
                if input_ports > 0
                else ""
            )
            output_labels = (
                "|".join(
                    [
                        f"<o{i}> o{i}"
                        for i in range(input_ports + 1, input_ports + output_ports + 1)
                    ]
                )
                if output_ports > 0
                else ""
            )

            # Format the label with conditional parts for input and output labels
            if input_labels and output_labels:
                label = f"{{{{{input_labels}}} | {node_name}: {component_name} | {{{output_labels}}}}}"
            elif input_labels:
                label = f"{{{{{input_labels}}} | {node_name}: {component_name} }}"
            elif output_labels:
                label = f"{{ {node_name}: {component_name} | {{{output_labels}}}}}"
            else:
                label = f"{node_name}: {component_name}"
        else:
            # handle cases without ports info
            label = f"{node_name}: {component_name}"

        dot_lines.append(f'  {node_name} [label="{label}"];')

    dot_lines.append("}")

    return "\n".join(dot_lines)


def edges_dot_to_yaml(session):
    """Converts a DOT graph string to a YAML dictionary."""
    # Regular expression to find the edges
    edge_pattern = re.compile(r"(\w+):(\w+) -- (\w+):(\w+);")

    # Find all matches in the DOT graph string
    edges = edge_pattern.findall(session["p300_dot_string"])

    # Format edges as required
    formatted_edges = [f"{edge[0]},{edge[1]}: {edge[2]},{edge[3]}" for edge in edges]

    circuit = session["p300_circuit_dsl"]
    circuit["edges"] = {}
    for i, edge in enumerate(formatted_edges):
        circuit["edges"][f"E{i+1}"] = {}
        circuit["edges"][f"E{i+1}"]["link"] = edge

    return circuit


def dsl_to_gf(circuit_dsl):
    """Converts a circuit DSL to a GDSFactory netlist."""
    # Create a new nodes dictionary with renamed keys
    new_nodes = {}
    for node_id, node_info in circuit_dsl["nodes"].items():
        new_nodes[node_id] = {
            "component": node_info["component"],
            "info": node_info["properties"],  # Rename properties to info
            "settings": node_info["params"],  # Rename params to settings
        }

    new_routes = {}
    for _edge_id, edge_info in circuit_dsl["edges"].items():
        link = edge_info["link"]
        source, target = link.split(": ")
        new_routes[source] = target

    placements = {}
    for node_id, node_info in circuit_dsl["nodes"].items():
        placements[node_id] = {}
        placements[node_id]["x"] = node_info["placement"]["x"]
        placements[node_id]["y"] = node_info["placement"]["y"]
        placements[node_id]["rotation"] = node_info["placement"]["rotation"]

    gf_netlist = {
        "instances": new_nodes,
        "routes": {
            "optical": {
                "links": new_routes,  # Move the list of links under routes: optical: links
            }
        },
        "placements": placements,
        "ports": circuit_dsl["ports"],
    }

    return gf_netlist


def get_graphviz_placements(dot_string):
    """Get the node positions from a DOT graph string."""
    # Create a graph from the DOT string
    graph = pgv.AGraph(string=dot_string)

    # Set node separation (horizontal) (inches)
    graph.graph_attr["nodesep"] = ".05"

    # Set rank separation (vertical) (inches)
    graph.graph_attr["ranksep"] = ".05"

    # Layout the graph using the dot layout
    graph.layout(prog="dot")

    # Render the graph to a file
    graph.draw(PATH.build / "graph.svg")

    # Get node positions ---> THIS IS THE CENTER OF THE NODES. BUT GDSFACTORY USES THE BOTTOM LEFT CORNER.
    positions = {}
    for node in graph.nodes():
        pos = node.attr["pos"]
        if pos:
            x, y = map(float, pos.split(","))
            positions[str(node)] = (x, y)
    return positions


def dot_add_node_sizes(dot_string, node_dimensions):
    """Add node sizes to a DOT graph string."""
    PADDING_FACTOR = 1.2  # in percent
    PADDING_MARGIN = 1  # ? in microns or inches?
    lines = dot_string.split("\n")
    output_lines = []
    nodes_added = set()

    for line in lines:
        stripped_line = line.strip()
        # if stripped_line.startswith("C") and "[" in stripped_line and stripped_line.endswith("];"):
        if ("label=" in stripped_line) and stripped_line.endswith("];"):
            node_name = stripped_line.split("[")[0].strip()
            if node_name in node_dimensions and node_name not in nodes_added:
                width, height = node_dimensions[node_name]
                size_line = f"  {node_name} [width={width*PADDING_FACTOR+PADDING_MARGIN}, height={height*PADDING_FACTOR+PADDING_MARGIN}, shape=record, fixedsize=true];"
                output_lines.append(size_line)
                nodes_added.add(node_name)
        output_lines.append(line)

    return "\n".join(output_lines)


def multiply_node_dimensions(node_dimensions, factor=0.01):
    """Multiply the dimensions of nodes by a factor."""
    return {
        node: (round(width * factor, 3), round(height * factor, 3))
        for node, (width, height) in node_dimensions.items()
    }


def add_placements_to_dsl(session):
    """Add node placements to the circuit DSL."""
    placements = session["p300_graphviz_node_coordinates"]
    circuit = session["p300_circuit_dsl"]

    # Add placements to the data
    # GRAPHVIZ returns CENTER OF THE NODES. BUT GDSFACTORY USES THE BOTTOM LEFT CORNER: (not sure which corner but this is working: )
    for key, value in placements.items():
        circuit["nodes"][key]["placement"] = {}
        circuit["nodes"][key]["placement"]["rotation"] = 0  # TODO
        circuit["nodes"][key]["placement"]["x"] = (
            value[0] - circuit["nodes"][key]["properties"]["dx"] / 2
        )
        circuit["nodes"][key]["placement"]["y"] = (
            value[1] - 0 * circuit["nodes"][key]["properties"]["dy"] / 2
        )

    # circuit['']['placements'] = {key: {'x': value[0]-circuit['nodes'][key]['info']['dx']/2,
    #                             'y': value[1]-0*circuit['nodes'][key]['info']['dy']/2} for key, value in placements.items()}

    return circuit


def add_final_ports(session):
    """Add final ports to the circuit DSL."""

    def find_open_ports(dot_source):
        edges = []
        nodes_ports = {}

        # Parse the graph source to identify edges and node labels
        for line in dot_source.splitlines():
            line = line.strip()
            if "--" in line:
                edge = line.strip(";").split(" -- ")
                edges.append(tuple(edge))
            elif '[label="' in line:
                node = line.split()[0]
                label = re.search(r'\[label="(.*)"\]', line).group(1)
                ports = re.findall(r"<(o\d+)>", label)
                nodes_ports[node] = ports

        # Extract connected ports
        connected_ports = set()
        for edge in edges:
            for endpoint in edge:
                node, port = endpoint.split(":")
                connected_ports.add(f"{node}:{port}")

        # Determine open ports
        open_ports = []
        for node, ports in nodes_ports.items():
            for port in ports:
                if f"{node}:{port}" not in connected_ports:
                    open_ports.append(f"{node}:{port}")

        return open_ports

    def create_port_dict(open_ports):
        port_dict = {}
        for i, open_port in enumerate(open_ports):
            port_dict[f"o{i+1}"] = open_port.replace(":", ",")
        return port_dict

    open_ports_list = find_open_ports(session["p300_dot_string"])
    circuit_ports_dict = create_port_dict(open_ports_list)

    session["p300_circuit_dsl"]["ports"] = circuit_ports_dict
    # print(yaml.dump(data, default_flow_style=False))
    return session["p300_circuit_dsl"]


def get_file_path(dir_string):
    """Get the absolute file path from a relative directory string."""
    current_dir = os.path.dirname(__file__)
    current_dir = os.path.join(
        current_dir, "..", "KnowledgeBase"
    )  # Use os.path.join for path construction
    file_path = os.path.join(
        current_dir, *dir_string.split(os.sep)
    )  # Use os.sep for splitting
    file_path = os.path.abspath(file_path)
    return file_path


matplotlib.use("Agg")

# Set the plot style to dark background
plt.style.use("dark_background")

# Set font type globally
font = {
    "family": "monospace",  # You can change this to any available font family
    "size": 11,
}
matplotlib.rc("font", **font)


def plot_dict_arrays(wl, data_dict):
    """Plot a dictionary of arrays."""
    # only keeping the s-params from port 1 to others:
    data_dict = {k: v for k, v in data_dict.items() if k[0] == "o1"}

    cols = 6  # Number of columns in the subplot grid

    num_plots = len(data_dict)
    rows = (num_plots // cols) + (
        num_plots % cols > 0
    )  # Calculate the number of rows needed

    # Calculate the figure size to maintain a 4:3 aspect ratio for each subplot
    subplot_width = 3
    subplot_height = 2
    fig_width = subplot_width * cols
    fig_height = subplot_height * rows

    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Flatten axes array for easy iteration if rows and cols are more than 1
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for _idx, (ax, (key, array)) in enumerate(zip(axes, data_dict.items())):
        ax.plot(
            wl,
            10 * np.log10(np.abs(array) ** 2),
            label=f"{key[0]}-{key[1]}",
            linewidth=2,
            color="darksalmon",
        )  # Make plot lines thicker
        ax.text(
            0.95,
            0.95,
            f"{key[0]}-{key[1]}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax.transAxes,
        )

    # Remove any unused subplots
    for ax in axes[len(data_dict) :]:
        fig.delaxes(ax)

    plt.tight_layout(pad=0.2)
    plt.subplots_adjust(wspace=0.2, hspace=0.1)

    plt.savefig(PATH.build / "plot_sax.png")
    plt.close()

    return fig


wl_cband = np.linspace(1.500, 1.600, 128)
PathType = str | pathlib.Path


def model_from_npz(
    filepath: PathType | np.ndarray,
    xkey: str = "wavelengths",
    xunits: float = 1,
) -> Model:
    """This is a modified version of the original function in gplugins/sax/read.py
    Returns a SAX Sparameters Model from a npz file.

    The SAX Model is a function that returns a SAX SDict interpolated over wavelength.

    Args:
        filepath: CSV Sparameters path or pandas DataFrame.
        xkey: key for wavelengths in file.
        xunits: x units in um from the loaded file (um). 1 means 1um.
    """
    sp = np.load(filepath) if isinstance(filepath, pathlib.Path | str) else filepath
    keys = list(sp.keys())

    if xkey not in keys:
        raise ValueError(f"{xkey!r} not in {keys}")

    x = jnp.asarray(sp[xkey] * xunits)
    wl = jnp.asarray(wl_cband)

    # make sure x is sorted from low to high
    idxs = jnp.argsort(x)
    x = x[idxs]
    sp = {k: v[idxs] for k, v in sp.items()}

    @jax.jit
    def model(wl: Float = wl):
        S = {}
        zero = jnp.zeros_like(x)

        for key in sp:
            if not key.startswith("wav"):
                port_mode0, port_mode1 = key.split(",")
                port0, _ = port_mode0.split("@")
                port1, _ = port_mode1.split("@")

                m = jnp.interp(wl, x, np.abs(sp.get(key, zero)))
                a = jnp.interp(wl, x, np.unwrap(np.angle(sp.get(key, zero))))
                S[(port0, port1)] = m * jnp.exp(1j * a)

        return S

    return model


def dot_crossing_edges(session):
    """Check if a Graphviz dot string has any crossing edges.

    Args:
        session: The Streamlit session object containing the dot string to check.
    """
    dot_string = session.p300_dot_string

    happy_flag = dot_planarity(dot_string)
    if happy_flag:
        return "No crossing edges found."
    else:
        return "Eroor: crossing edges found!"
    # , prompts["dot_verify"], session.p100_llm_api_selection


def dot_planarity(dot_string):
    """Check if a Graphviz dot string has any crossing edges.
    This function takes a dot string as input, applies a layout algorithm to position
    the nodes and edges, and then checks for any crossing edges in the graph.

    Args:
        session: The Streamlit session object containing the dot string to check.
    """
    # Remove any extraneous tokens at the start (e.g. an extra "dot" line)
    lines = dot_string.strip().splitlines()
    if lines and lines[0].strip() == "dot":
        dot_string = "\n".join(lines[1:])
        
    # Load the dot string
    graph = pgv.AGraph(string=dot_string)

    # Apply a layout to the graph
    graph.layout(prog="dot")

    # Get edge coordinates
    edges = []
    for edge in graph.edges():
        points = edge.attr["pos"].split()
        start = tuple(map(float, points[0].split(",")))
        end = tuple(map(float, points[-1].split(",")))
        edges.append((start, end))

    # Function to check if two line segments (p1, q1) and (p2, q2) intersect
    def do_intersect(p1, q1, p2, q2):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0
            return 1 if val > 0 else 2

        def on_segment(p, q, r):
            if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[
                1
            ] <= max(p[1], r[1]):
                return True
            return False

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    # Check each pair of edges for intersection
    crossings = []
    for i, (p1, q1) in enumerate(edges):
        for j, (p2, q2) in enumerate(edges):
            if i != j and do_intersect(p1, q1, p2, q2):
                crossings.append(((p1, q1), (p2, q2)))

    if len(crossings) == 0:
        return True
    else:
        return False
