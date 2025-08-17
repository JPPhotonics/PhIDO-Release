import random
import time
from datetime import datetime

import streamlit as st
import yaml

from PhotonicsAI.Photon import llm_api, utils
from PhotonicsAI.Photon.DemoPDK import *

# import hmac
# import requests
# import json
# from PIL import Image
# from io import BytesIO
# import streamlit.components.v1 as components

# Set the page configuration to wide mode
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
        header {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)

with open("prompts.yaml") as file:
    prompts = yaml.safe_load(file)

db_docs = utils.search_directory_for_docstrings()
list_of_docs = [i["docstring"] for i in db_docs]
list_of_cnames = [i["module_name"] for i in db_docs]

# Initialize session state
if "current_message" not in st.session_state:
    st.session_state.current_message = ""
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "show_examples" not in st.session_state:
    st.session_state.show_examples = True
if "input_submitted" not in st.session_state:
    st.session_state.input_submitted = False


# Function to handle input submission
def check_input_change():
    if st.session_state.chat_input != st.session_state.last_input:
        if st.session_state.chat_input.strip():  # Check if input is not just whitespace
            st.session_state.current_message = st.session_state.chat_input
            st.session_state.show_examples = False
            st.session_state.input_submitted = True
        st.session_state.last_input = st.session_state.chat_input


llm_api_selection = st.sidebar.radio(
    "LLM:",
    (
        "gpt-4o",
        "llama-v3-70b-instruct",
        "gemini-1.5-flash",
    ),
)

st.sidebar.markdown(
    """
    <div style="margin-top: 40vh; text-align: center;">
    </div>
""",
    unsafe_allow_html=True,
)


# Display processed text
st.markdown("#### Photon Fury ⚡")
st.markdown("A Furious Photonic Chip Engineer")

st.markdown(
    """
    <div style="margin-top: 4vh; text-align: center;">
    </div>
""",
    unsafe_allow_html=True,
)

# Move the text input for chat input to the main page
if not st.session_state.input_submitted:
    st.markdown("Input text:")
    chat_input = st.text_input(
        "You: ",
        key="chat_input",
        on_change=check_input_change,
        label_visibility="collapsed",
    )


# Function to handle button clicks
def on_button_click(button_text):
    st.session_state.current_message = button_text
    st.session_state.show_examples = False
    st.session_state.input_submitted = True
    st.rerun()


example_prompts = [
    "What is a transceiver?",
    "connect four 1x1 fast amplitude modulators to a 4x1 WDM",
    "A 1x2 MMI for 1310 nm and a length of 40 um",
    "Connect a 2x2 mzi with heater to a directional coupler with a length of 125 um, dy of 40 um, and dx of 100 um",
    "A low loss 1x4 power splitter connected to four fast amplitude modulators.",
    "Two cascaded fast MZIs, each with two input and two output ports.",
    "A low loss 1x2 power splitter connected to two GHz modulators.",
    # "A high speed modulator connected to a VOA",
    "Layout 1x2 MMIs connected to each other to for a 1x8 splitter tree",
    "A 2x2 MZI with 1 GHz bandwidth",
    "Cascaded 2x2 MZIs to create a switch network with 8 outputs",
    "A 2d mesh of nine MZIs. Each MZI has two input and two outputs and they are back to back connected",
    "A 1x2 splitter connected to two amplitude modulators with 100 dB extinction ratio",
    "Eight low loss and low power thermo optic phase shifters. the phase shifters should be arranged in parallel in an array",
]

custom_css = """
<style>
    .stButton > button {
        width: 400px;
        height: 20px;
        border-radius: 10px;
        }
</style>
"""

# Example buttons in a container
if st.session_state.show_examples:
    # Add vertical space and center the container
    st.markdown(
        """
        <div style="margin-bottom: 10vh; text-align: center;">
        </div>
    """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("Try one of these:")
        col1, col2 = st.columns(2)
        with col1:
            for example in example_prompts:
                display_text = (
                    example[:45] + " ..."
                )  # Limit text to first 100 characters

                st.markdown(custom_css, unsafe_allow_html=True)
                if st.button(display_text):
                    on_button_click(example)
                    # process_prompt(example, llm_api_selection)
        with col2:
            if st.button("Mode Solver", disabled=True):
                on_button_click("Button 2 text")
            if st.button("Button 4", disabled=True):
                on_button_click("Button 4 text")


def logger(prompt):
    with open("./logs/st_prompt_logs.txt", "a") as file:
        file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {prompt}\n')


# Processing function
def process_prompt(prompt, llm_api_selection):
    start_time = time.time()
    logger(prompt)
    d = {
        "prompt": prompt,
        #  'prompts': prompts,
        "llm_api_selection": llm_api_selection,
        "list_of_docs": list_of_docs,
        "list_of_cnames": list_of_cnames,
    }

    with st.container(border=True):
        st.markdown(f"*{prompt}*")

    ###########
    # safeguard
    d["guard"] = llm_api.call_llm(d["prompt"], prompts["guard"], llm_api_selection)
    if d["guard"] == "False":
        d["prompt"] = random.choice(d["list_of_cnames"])
        st.write(
            f"Cheeky! I cannot help you with that. Enjoy this gorgeous {d['prompt']} instead."
        )
        # d['prompt'] = 'a waveguide'

    ###########
    # interpret
    d["raw_component_list"] = llm_api.call_llm(
        d["prompt"], prompts["parse0"], llm_api_selection
    )

    col1, col2 = st.columns(2)
    with col1:
        with st.container(height=300):
            st.write("Interpret\n```yaml\n" + d["raw_component_list"])

    ###########
    # Search
    d["db_component_list"], d["db_components_list_all"] = (
        llm_api.yaml_components_search(d)
    )
    with col2:
        with st.container(height=300):
            # st.write("Search\n```yaml\n"+d['db_component_list'])
            st.write("Search\n```yaml\n" + d["db_components_list_all"])

    ###########
    ########### Netlist
    with st.spinner("Initializing the netlist..."):
        d["raw_netlist"] = llm_api.parsed_prompt_to_netlist(d)

    with st.expander("Raw Netlist", expanded=False):
        st.write("```yaml\n" + d["raw_netlist"])

    with st.spinner("Parsing components settings..."):
        d["netlist1"] = settings_netlist(d["raw_netlist"])
        d["netlist2"] = llm_api.apply_settings(d)

    with st.expander("Netlist + Settings", expanded=False):
        st.write("```yaml\n" + d["netlist2"])

    ###########
    ########### DOT Graph
    d["dot_string"] = utils.netlist_to_dot(d["netlist2"])

    # with st.expander("dot-nodes", expanded=False):
    #     st.write("```dot\n"+d['dot_string'])

    with st.spinner("Working on the Circuit Diagram..."):
        d["dot_string"] = llm_api.dot_add_edges(d)
        # st.write("```dot\n"+d['dot_string'])

    st.graphviz_chart(d["dot_string"])

    # with st.spinner('Veryfying the graph...'):
    #     verify_message, d['dot_string'] = verifiers.dot(d)

    # if verify_message == '':
    #     st.markdown(f':green[Verified!]')
    # else:
    #     st.markdown(f':green[{verify_message}]')
    #     st.graphviz_chart(d['dot_string'])

    with st.expander("Circuit diagram"):
        st.write("```dot\n" + d["dot_string"])

    ###########
    ########### GDS
    d["valid_netlist"] = verifiers_depricated.edges_dot_to_yaml(d)

    d["footprints_dict"], d["netlist3"] = footprint_netlist(d["valid_netlist"])
    dot_string_scaled = utils.dot_add_node_sizes(
        d["dot_string"], utils.multiply_node_dimensions(d["footprints_dict"], 0.01)
    )

    d["graphviz_node_coordinates"] = utils.get_graphviz_placements(dot_string_scaled)
    graphviz_node_coordinates = utils.multiply_node_dimensions(
        d["graphviz_node_coordinates"], 100 / 72
    )
    d["netlist4"] = utils.add_placements_to_yaml(
        d["netlist3"], graphviz_node_coordinates
    )
    # print('=====================')
    # print('FOOTPRINTS', d['footprints_dict'])
    # print('PLACEMENTS', d['graphviz_node_coordinates'])

    with st.expander("Netlist"):
        st.write("```yaml\n" + d["netlist4"])

    try:
        (
            gds_fig,
            html_gds_fig,
            gds_netlist,
            gf_netlist_dict,
            gf_netlist_dict_recursive,
        ) = yaml_netlist_to_gds(d["netlist4"], ignore_links=False)
    except:
        (
            gds_fig,
            html_gds_fig,
            gds_netlist,
            gf_netlist_dict,
            gf_netlist_dict_recursive,
        ) = yaml_netlist_to_gds(d["netlist4"], ignore_links=True)
        st.markdown(":red[Routing error.]")
        pass
    st.pyplot(gds_fig)

    ###########
    ########### MODEL
    # recursive_netlist = sax.RecursiveNetlist.model_validate(gf_netlist_dict_recursive)
    # print('Required Models ==>', sax.get_required_circuit_models(recursive_netlist))
    # print('=====================')

    ###########
    ########### time
    end_time = time.time()
    st.write(f"{round(end_time - start_time,2)} seconds")


# Display only the most recent processed text on the main page
if st.session_state.current_message:
    process_prompt(st.session_state.current_message, llm_api_selection)
