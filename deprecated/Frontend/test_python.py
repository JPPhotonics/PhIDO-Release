import time
from datetime import datetime

import numpy as np
import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from PhotonicsAI.Photon import llm_api, utils
from PhotonicsAI.Photon.DemoPDK import *


def my_print(text, top_margin=2, bottom_margin=2):
    console = Console()

    # Create the top and bottom margin strings
    top_margin_str = "\n" * top_margin
    bottom_margin_str = "\n" * bottom_margin

    # Combine margins with the text
    full_text = f"{top_margin_str}{text}{bottom_margin_str}"

    # Create a Markdown object
    markdown_text = Markdown(full_text)

    # Create a Panel object to frame the text
    panel = Panel(markdown_text, width=100)

    # Print the panel with the framed text
    console.print(panel)


with open("prompts.yaml") as file:
    prompts = yaml.safe_load(file)

db_docs = utils.search_directory_for_docstrings()
list_of_docs = [i["docstring"] for i in db_docs]
list_of_cnames = [i["module_name"] for i in db_docs]


def logger(prompt):
    with open("./st_prompt_logs.txt", "a") as file:
        file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {prompt}\n')


# Processing function
def process_prompt(prompt, llm_api_selection):
    start_time = time.time()
    # logger(prompt)
    d = {
        "prompt": prompt,
        #  'prompts': prompts,
        "llm_api_selection": llm_api_selection,
        "list_of_docs": list_of_docs,
        "list_of_cnames": list_of_cnames,
    }

    my_print(f"*{prompt}*")

    ###########
    # NEW interpreter
    d = llm_api.interpreter_classify(d)

    if d["interpret_classify_dict"]["cat"] == 4:  # irrelevant
        my_print(
            "**I cannot help you with this question. I only know about integrated photonic devices and circuits.**"
        )

    elif d["interpret_classify_dict"]["cat"] == 2:  # generic question
        my_print(
            "**I am still learning about integrated photonics. Soon I will be able to answer questions like this.**"
        )

    elif d["interpret_classify_dict"]["cat"] == 3:  # mode simulation
        my_print("**Mode simsulation is under development.**")

    else:
        ###########
        ########### interpret

        my_print("Fetching templates...")
        d = llm_api.interpreter_parse(d)

        my_print("Interpreter\n```yaml\n" + d["components_str"])

        ###########
        ########### Search
        d["db_component_list"], d["db_components_list_all"] = (
            llm_api.yaml_components_search(d)
        )

        my_print("Search\n```yaml\n" + d["db_component_list"])

        ###########
        ########### Netlist
        my_print("Initializing the netlist...")
        d["raw_netlist"] = llm_api.parsed_prompt_to_netlist(d)

        my_print("Parsing components settings...")
        d["netlist1"] = settings_netlist(d["raw_netlist"])
        d["netlist2"] = llm_api.apply_settings(d)

        my_print("```yaml\n" + d["netlist2"])

        ###########
        ########### DOT Graph
        d["dot_string"] = utils.netlist_to_dot(d["netlist2"])

        my_print("Working on the Circuit Diagram...")
        d["dot_string"] = llm_api.dot_add_edges(d)

        # st.graphviz_chart(d['dot_string'])

        my_print("Circuit Diageam\n```dot\n" + d["dot_string"])

        ###########
        ########### GDS
        d["valid_netlist"] = verifiers_depricated.edges_dot_to_yaml(d)

        d["footprints_dict"], d["netlist3"] = footprint_netlist(d["valid_netlist"])
        dot_string_scaled = utils.dot_add_node_sizes(
            d["dot_string"], utils.multiply_node_dimensions(d["footprints_dict"], 0.01)
        )

        d["graphviz_node_coordinates"] = utils.get_graphviz_placements(
            dot_string_scaled
        )
        graphviz_node_coordinates = utils.multiply_node_dimensions(
            d["graphviz_node_coordinates"], 100 / 72
        )
        d["netlist4"] = utils.add_placements_to_yaml(
            d["netlist3"], graphviz_node_coordinates
        )
        d["netlist5"] = utils.add_final_ports(d)
        # print('=====================')
        # print('FOOTPRINTS', d['footprints_dict'])
        # print('PLACEMENTS', d['graphviz_node_coordinates'])

        my_print("Netlist\n```yaml\n" + d["netlist5"])

        try:
            d = yaml_netlist_to_gds(d, ignore_links=False)
        except:
            d = yaml_netlist_to_gds(d, ignore_links=True)
            my_print(":red[Routing error.]")
            pass

        my_print(f'Required models:\n{d['required_models']}')

        my_print(f's-params:\n{d['sax_circuit']()}')

        wl = np.linspace(1.5, 1.6, 100)
        result = d["sax_circuit"](wl=wl)
        utils.plot_dict_arrays(result)

        # st.pyplot(d['gdsfig'])

        ###########
        ########### MODEL
        # recursive_netlist = sax.RecursiveNetlist.model_validate(gf_netlist_dict_recursive)
        # print('Required Models ==>', sax.get_required_circuit_models(recursive_netlist))
        # print('=====================')

        ###########
        ########### time
        end_time = time.time()
        my_print(f"{round(end_time - start_time,2)} seconds")


if __name__ == "__main__":
    # prompt = 'layout a 2x2 mmi'
    prompt = "Connect a 2x2 mzi with heater to a directional coupler with a length of 125 um, dy of 100 um, and dx of 100 um"

    process_prompt(prompt, llm_api_selection="gpt-4o")
