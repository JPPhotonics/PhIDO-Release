import time
from datetime import datetime

import numpy as np
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


example_prompts_left = [
    "A four channel transceiver",
    "A five channel WDM",
    # 'Connect four 1x1 fast amplitude modulators to a 4x1 WDM',
    "A 1x2 MMI for 1310 nm and a length of 40 um",
    "Connect a 2x2 mzi with heater to a directional coupler with a length of 125 um, dy of 100 um, and dx of 100 um",
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

example_prompts_right = [
    "What is a transceiver?",
    "Simulate modes of a SiN wavguide, 400 nm width, 200 nm thick.",
    "What is the difference between a directional coupler and a MMI?",
    "Do you have access to internet?",
    "A four channel WDM connected to four grating couplers",
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
            for example in example_prompts_left:
                display_text = (
                    example[:45] + " ..."
                )  # Limit text to first 100 characters

                st.markdown(custom_css, unsafe_allow_html=True)
                if st.button(display_text):
                    on_button_click(example)

        with col2:
            for example in example_prompts_right:
                display_text = (
                    example[:45] + " ..."
                )  # Limit text to first 100 characters

                st.markdown(custom_css, unsafe_allow_html=True)
                if st.button(display_text):
                    on_button_click(example)


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
    # NEW interpreter
    d = llm_api.interpreter_classify(d)

    if d["interpret_classify_dict"]["cat"] == 4:  # irrelevant
        st.markdown(
            "**I cannot help you with this question. I only know about integrated photonic devices and circuits.**"
        )

    elif d["interpret_classify_dict"]["cat"] == 2:  # generic question
        st.markdown(
            "**I am still learning about integrated photonics. Soon I will be able to answer questions like this.**"
        )

    elif d["interpret_classify_dict"]["cat"] == 3:  # mode simulation
        st.markdown("**Mode simsulation is under development.**")

    else:
        ###########
        ########### interpret

        with st.spinner("Fetching templates..."):
            d = llm_api.interpreter_parse(d)

        col1, col2 = st.columns(2)
        # with col1:
        #     with st.container(height=300):
        #         st.write("Interpreter\n```yaml\n"+d['interpret_classify_str'])

        with col1:
            with st.container(height=300):
                st.write("Interpreter\n```yaml\n" + d["components_str"])

        # d['raw_component_list'] = call_llm(d['prompt'], prompts['parse0'], d['llm_api_selection'])

        ###########
        ########### Search
        d["db_component_list"], d["db_components_list_all"] = (
            llm_api.yaml_components_search(d)
        )
        with col2:
            with st.container(height=300):
                st.write("Search\n```yaml\n" + d["db_component_list"])
                # st.write(f"Search\n```yaml\n"+d['db_components_list_all'])

        ###########
        ########### Netlist
        with st.spinner("Initializing the netlist..."):
            d["raw_netlist"] = llm_api.parsed_prompt_to_netlist(d)

        # with st.expander("Raw Netlist", expanded=False):
        #     st.write("```yaml\n"+d['raw_netlist'])

        with st.spinner("Parsing components settings..."):
            d["netlist1"] = settings_netlist(d["raw_netlist"])
            d["netlist2"] = llm_api.apply_settings(d)
            d["netlist2"] = info_netlist(d)

        with st.expander("Netlist draft", expanded=False):
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

        with st.expander("Netlist"):
            st.write("```yaml\n" + d["netlist5"])

        try:
            d = yaml_netlist_to_gds(d, ignore_links=False)
        except:
            d = yaml_netlist_to_gds(d, ignore_links=True)
            st.markdown(":red[Routing error.]")
            pass

        ###########
        ########### circuit optimizer
        # circuit_optimizer(d)

        ###########
        ########### final GDS

        #### render GDS again

        st.pyplot(d["gdsfig"])

        with st.expander("Required SAX Models", expanded=False):
            st.write(d["required_models"])
            st.write(str(d["sax_circuit"]()))

        wl = np.linspace(1.5, 1.6, 100)
        result = d["sax_circuit"](wl=wl)
        d["sax_plot"] = utils.plot_dict_arrays(wl, result)
        st.pyplot(d["sax_plot"])

        ###########
        ########### MODEL
        # recursive_netlist = sax.RecursiveNetlist.model_validate(gf_netlist_dict_recursive)
        # print('Required Models ==>', sax.get_required_circuit_models(recursive_netlist))

        ###########
        ########### time
        end_time = time.time()
        st.write(f"{round(end_time - start_time,2)} seconds")


# Display only the most recent processed text on the main page
if st.session_state.current_message:
    process_prompt(st.session_state.current_message, llm_api_selection="gpt-4o")
