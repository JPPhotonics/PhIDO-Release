import re
import warnings
from collections import OrderedDict

import pygraphviz as pgv
import yaml

from PhotonicsAI.Photon import llm_api


def is_valid_dot(dot_string):
    """Check if the dot string is a valid graph."""
    try:
        pgv.AGraph(string=dot_string)
        return True  # If no exception is raised, then dot_string is a valid graph representation
    except:
        return False


def dot_label(dot_string):
    try:
        graph = pgv.AGraph(string=dot_string)
    except Exception as e:
        return f"Invalid DOT string: {e}"

    # Regular expression to match the label syntax
    label_pattern = re.compile(r"^\{\{[<>\w\s|]+\} \| [\w: ]+ \| \{[<>\w\s|]+\}\}$")

    for node_name in graph.nodes():
        node = graph.get_node(node_name)
        label = node.attr["label"]
        if label:
            label = label.strip('"')
            if not label_pattern.match(label):
                return f"Invalid label syntax for node {node_name}: {label}"

    return True


def dot_ports(dot_string):
    """Parses a DOT string to extract port information from node labels,
    checks if the port lists are sorted in ascending order and are unique.

    If all labels are valid, returns True.
    If not, returns a list of invalid labels.
    """
    graph = pgv.AGraph(string=dot_string)

    invalid_labels = []
    label_pattern = re.compile(r"\{\{([^\}]+)\}\s*\|\s*([^\|]+)\|\s*\{([^\}]+)\}\}")

    for node in graph.nodes():
        label = node.attr["label"]
        if label:
            match = label_pattern.match(label.strip('"'))
            if match:
                left_ports = [
                    int(port[1:]) for port in re.findall(r"<(\w+)>", match.group(1))
                ]
                right_ports = [
                    int(port[1:]) for port in re.findall(r"<(\w+)>", match.group(3))
                ]
                concatenated_ports = list(reversed(left_ports)) + right_ports
                is_sorted_and_unique = concatenated_ports == sorted(
                    concatenated_ports
                ) and len(concatenated_ports) == len(set(concatenated_ports))

                if not is_sorted_and_unique:
                    invalid_labels.append(label.strip('"'))
            else:
                return f"Invalid label structure for node {node.name}: {label}"

    return True if not invalid_labels else invalid_labels


def dot_planarity(dot_string):
    """Check if a Graphviz dot string has any crossing edges.

    This function takes a dot string as input, applies a layout algorithm to position
    the nodes and edges, and then checks for any crossing edges in the graph.
    """
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


def dot(data: dict):
    message = ""

    if not is_valid_dot(data["dot_string"]):
        return "Invalid DOT string"

    if not dot_label(data["dot_string"]):
        return "Invalid node labels"

    if dot_ports(data["dot_string"]) is not True:
        data["dot_string"] = llm_api.repair_dot_ports(data)
        message += "Correction: ports orders. \n"

    if dot_planarity(data["dot_string"]) is False:
        data["dot_string"] = llm_api.repair_dot_edges(data)
        message += "Correction: links."

    return message, data["dot_string"]


def edges_dot_to_yaml(d):
    # Regular expression to find the edges
    edge_pattern = re.compile(r"(\w+):(\w+) -- (\w+):(\w+);")

    # Find all matches in the DOT graph string
    edges = edge_pattern.findall(d["dot_string"])

    # Format edges as required
    formatted_edges = [f"{edge[0]},{edge[1]}: {edge[2]},{edge[3]}" for edge in edges]

    netlist = yaml.safe_load(d["netlist2"])
    if "reasoning" in netlist:
        del netlist["reasoning"]
    netlist["routes"] = {}
    netlist["routes"]["optical"] = {}
    netlist["routes"]["optical"]["links"] = {}

    for edge in formatted_edges:
        source, target = edge.split(": ")
        netlist["routes"]["optical"]["links"][source] = target

    # Dump the updated netlist back to a YAML string
    updated_netlist = yaml.dump(netlist, sort_keys=False)

    return updated_netlist


def custom_warning_format(message, category, filename, lineno, line=None):
    """Custom warning formatter to add color and extra formatting."""
    # WARNING_COLOR = '\033[93m'  # Yellow
    WARNING_COLOR = "\033[38;2;255;165;0m"  # Orange
    RESET_COLOR = "\033[0m"  # Reset to default
    formatted_message = (
        f"{WARNING_COLOR}Warning: {message}{RESET_COLOR}\n"
        f"  Category: {category.__name__}\n"
        f"  File: {filename}, Line: {lineno}\n"
    )
    return formatted_message


# Set the custom formatter
warnings.formatwarning = custom_warning_format


def validate_cell_settings(settings, valid_args):
    """Validates the input settings against the predefined valid arguments.

    Parameters:
    settings (dict): A dictionary containing functional settings.
    valid_args (dict): A dictionary containing valid arguments with their ranges and default values.

    Returns:
    dict: A dictionary containing the validated settings, categorized by functional and geometrical.
    """
    validated_settings = {"functional": {}, "geometrical": {}}

    # Check for settings that do not exist in valid_args
    all_valid_keys = set()
    for category in valid_args.values():
        all_valid_keys.update(category.keys())

    for setting in settings.keys():
        if setting not in all_valid_keys:
            warnings.warn(
                f"Setting '{setting}' does not exist in valid_args.", UserWarning
            )

    # Validate and categorize settings
    for category, args in valid_args.items():
        for arg, props in args.items():
            value = settings.get(arg, props["default"])
            if not (props["range"][0] <= value <= props["range"][1]):
                warnings.warn(
                    f"Argument '{arg}' with value {value} is out of acceptable range {props['range']}. "
                    f"Using default value {props['default']}.",
                    UserWarning,
                )
                value = props["default"]
            validated_settings[category][arg] = value

    return validated_settings


def remove_redundant_from_yaml(yaml_string):  # not working well
    def ordered_load(stream, Loader=yaml.SafeLoader, object_pairs_hook=OrderedDict):
        class OrderedLoader(Loader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))

        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
        )
        return yaml.load(stream, OrderedLoader)

    def ordered_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
        class OrderedDumper(Dumper):
            pass

        def _dict_representer(dumper, data):
            return dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items()
            )

        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(data, stream, OrderedDumper, **kwds)

    # Load the YAML string into an OrderedDict
    data = ordered_load(yaml_string)

    # Convert the OrderedDict to a regular dict to remove duplicates
    # This will keep the last occurrence of any duplicate keys
    cleaned_data = dict(data)

    # Convert back to YAML string
    cleaned_yaml = ordered_dump(cleaned_data, default_flow_style=False)

    return cleaned_yaml


# ['name', 'instances', 'placements', 'connections', 'nets', 'ports', 'routes', 'settings', 'info', 'pdk', 'warnings', 'schema', 'schema_version']
def verify_and_filter_netlist(
    yaml_str,
    allowed_fields=None,
):
    """Verifies and filters a YAML string, keeping only the allowed high-level fields.

    :param yaml_str: The YAML string to verify and filter.
    :param allowed_fields: A list of high-level fields that are acceptable.
    :return: The verified and filtered YAML string.
    """
    # yaml_str = remove_redundant_from_yaml(yaml_str)
    if allowed_fields is None:
        allowed_fields = ["name", "instances", "routes", "placements", "connections"]
    yaml_str = llm_api.netlist_cleanup(yaml_str)

    try:
        # Load the YAML string into a dictionary
        data = yaml.safe_load(yaml_str)

        # Check if the data is a dictionary
        if not isinstance(data, dict):
            raise ValueError("The provided YAML is not a dictionary at the top level.")

        # Filter the dictionary to only include allowed fields
        filtered_data = {
            key: value for key, value in data.items() if key in allowed_fields
        }

        # Dump the filtered dictionary back to a YAML string
        verified_yaml_str = yaml.safe_dump(filtered_data)

        return verified_yaml_str
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML: {exc}")
    except Exception as e:
        raise ValueError(f"An error occurred: {e}")
