# ruff: noqa
import copy
import pickle
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
import yaml

from PhotonicsAI.config import PATH
from PhotonicsAI.Photon import llm_api, utils
from PhotonicsAI.Photon.DemoPDK import *

# Set the page configuration to wide mode
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_icon="icon.png",
)

st.markdown(
    """
    <style>
        header {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
session = st.session_state
# if 'project_data' not in session:
#     session.project_data = {}
# sdata = session.project_data
if "current_message" not in session:
    session.current_message = ""
if "last_input" not in session:
    session.last_input = ""
if "show_examples" not in session:
    session.show_examples = True
if "input_submitted" not in session:
    session.input_submitted = False
if "optimizer_mode" not in session:
    session.optimizer_mode = False 
if "template_selected" not in session:
    session.template_selected = False
if "components_selected" not in session:
    session.components_selected = False
if "user_specs" not in session:
    session.user_specs = {}
if "p100" not in session:
    session.p100 = True


with open(PATH.prompts) as file:
    prompts = yaml.safe_load(file)

components_list = utils.search_directory_for_docstrings()
list_of_docs = [i["docstring"] for i in components_list]
list_of_cnames = [i["module_name"] for i in components_list]

with open(PATH.templates) as file:
    templates_dict = yaml.safe_load(file)


# Function to handle input submission
def check_input_change():
    if session.chat_input != session.last_input:
        if session.chat_input.strip():  # Check if input is not just whitespace
            session.current_message = session.chat_input
            session.show_examples = False
            # session.input_submitted = True
        session.last_input = session.chat_input


# Display processed text
# st.markdown("#### Photon Fury ⚡")
# st.markdown("A Furious Photonic Chip Engineer")
st.markdown("#### Phido ⚡")
st.markdown("PHotonic Intelligent Design & Optimization")

st.markdown(
    """
    <div style="margin-top: 4vh; text-align: center;">
    </div>
""",
    unsafe_allow_html=True,
)

# Move the text input for chat input to the main page
if not session.input_submitted:
    # st.markdown('⇨ Describe a photonic circuit:')
    chat_input = st.text_input(
        "You: ",
        placeholder="⇨ Describe a photonic circuit and hit Enter!",
        key="chat_input",
        on_change=check_input_change,
        label_visibility="collapsed",
    )
    session.input_submitted = True

col3, col4 = st.columns([4, 1])
with col4:
    session.optimizer_mode = st.toggle("Enable optimizer mode", key="optimizer_mode_toggle")
    session.input_submitted = False

# Function to handle button clicks
def on_button_click(button_text):
    session.current_message = button_text
    session.show_examples = False
    session.input_submitted = True
    st.rerun()


example_prompts_left = [
    "A 2x2 MZI",
    "A wavelength division demultiplexer",
    "A transceiver with four wavelength channels",
    "A four channel WDM",
    # 'Connect four 1x1 fast amplitude modulators to a 4x1 WDM',
    # "A 1x2 MMI for 1310 nm and a length of 40 um",
    # 'Connect a 2x2 mzi with heater to a directional coupler with a length of 125 um, dy of 100 um, and dx of 100 um',
    # "A low loss 1x4 power splitter connected to four fast amplitude modulators.",
    # "Two cascaded fast MZIs, each with two input and two output ports.",
    "A power splitter connected to two MZIs with thermo-optic phase shifters each with a path difference 100 um",
    "A low loss 1x2 power splitter connected to two GHz modulators each with a delta length of 100 um.",
    # "A high speed modulator connected to a VOA",
    "Layout 1x2 MMIs connected to each other to for a 1x8 splitter tree",
    "A 2x2 MZI with 1 GHz bandwidth",
    "A 2d mesh of nine MZIs. Each MZI has two input and two outputs and they are back to back connected",
    "A 1x2 splitter connected to two amplitude modulators with 100 dB extinction ratio",
    "Eight low loss and low power thermo optic phase shifters. the phase shifters should be arranged in parallel in an array",
]

example_prompts_right = [
    "Cascaded 2x2 MZIs to create a switch tree network with 8 outputs",
    "Coupler with a 300 um distance between the two input ports",
    "coupler with sbend height of 300 and sbend length of 300",
    "Connect a 2x2 mzi with heater to a directional coupler with a length of 125 um, dy of 100 um, and dx of 100 um",
    "A low loss 1x4 power splitter connected to four fast amplitude modulators.",
    "Two cascaded fast MZIs, each with two input and two output ports.",
    "three 1x2 mmi use a 20um bend radius for routing",
    # 'What is a transceiver?',
    # 'Simulate modes of a SiN wavguide, 400 nm width, 200 nm thick.',
    # 'What is the difference between a directional coupler and a MMI?',
    # 'Do you have access to internet?',
    # 'A four channel WDM connected to four grating couplers',
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
if session.show_examples:
    # Add vertical space and center the container
    st.markdown(
        """
        <div style="margin-bottom: 10vh; text-align: center;">
        </div>
    """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("Or try one of these:")
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


def get_next_log_filename(
    directory=PATH.logs, prefix="log_", extension=".pickle", digits=4
):
    # Convert directory to a Path object
    dir_path = Path(directory)

    # Get the current date and time in the desired format
    current_time = datetime.now().strftime("d%Y%m%d_t%H%M%S")

    # Generate a random 5-digit number and check if it already exists
    while True:
        random_number = random.randint(
            0, 10**digits - 1
        )  # Generates a number between 00000 and 99999
        formatted_number = str(random_number).zfill(
            digits
        )  # Pad with zeros if necessary

        # Format the next log filename
        next_filename = f"{prefix}{current_time}_{formatted_number}{extension}"

        # Check if the file already exists
        if not (dir_path / next_filename).exists():
            break  # If the filename doesn't exist, exit the loop

    # Return the complete path as a string
    return str(dir_path / next_filename), random_number


def pickleable(obj):
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, AttributeError, TypeError):
        return False
    return True


def logger():
    session_data = session.to_dict()

    # Remove or transform non-pickleable objects
    for key, value in session_data.items():
        if not pickleable(value):  # You may implement a helper function to check this.
            session_data[key] = "non-pickleable"

    session_data.pop(
        "p400_gdsfig", None
    )  # remove gdsfig, it's about 5 MB in size, not sure why

    with open(session.log_filename, "wb") as file:
        pickle.dump(session_data, file)


def on_template_select(template_id):
    session.template_selected = True
    session.p100 = False
    session.p200_selected_template = template_id


def on_component_select(c_selected_idx):
    session.components_selected = True
    session.p100 = False
    session.p200_selected_components = c_selected_idx


def display_templates_columns():
    # Determine the number of columns based on the length of the list
    num_columns = min(len(session.p200_retreived_templates), 2)

    # Create the columns
    columns = st.columns(num_columns)

    # Populate the columns with items
    for i, item in enumerate(session.p200_retreived_templates):
        with columns[i % num_columns]:
            with st.container(height=400):
                t = item[1]
                id_ = item[0]
                st.write(t["doc"]["title"] + "\n" + t["doc"]["description"])
                st.markdown(f"[reference]({t['doc']['reference']})")
                st.button(
                    id_,
                    use_container_width=True,
                    on_click=on_template_select,
                    args=(id_,),
                )


def display_components_columns():
    c_idx = []
    scores = []
    for c in session.p200_componenets_search_r:
        c_idx.append(c.match_list)
        scores.append(c.match_scores)

    c_selected_idx = [None] * len(c_idx)
    session.p200_selected_components = []

    with st.form(key="my_form"):
        st.write("Picking components:")
        for i in range(len(c_idx)):
            with st.container(border=True):
                st.write(f"**{session.p200_pretemplate['components_list'][i]}**")

                # options = [f"{list_of_cnames[j]} ({j}) ({scores[i][k]})" for k, j in enumerate(c_idx[i])]
                for _ii in range(len(c_idx)):
                    options = []
                    for k, j in enumerate(c_idx[i]):
                        name = list_of_cnames[j]
                        score = scores[i][k]
                        if score == "exact":
                            # option = f"{name} [:green[{score}], {j}] "
                            option = f"{name} :green[/{score}/]"
                        elif score == "partial":
                            # option = f"{name} [:orange[{score}], {j}]"
                            option = f"{name} :orange[/{score}/]"
                        elif score == "poor":
                            # option = f"{name} [:red[{score}], {j}]"
                            option = f"{name} :red[/{score}/]"
                        else:
                            # option = f"{name} [:grey[{score}], {j}]"
                            option = f"{name} :grey[/{score}/]"
                        options.append(option)

                c_selected_idx[i] = st.radio(
                    label=str(i), options=options, label_visibility="collapsed"
                )

        c_selected_idx = [s.split(" :")[0] for s in c_selected_idx]
        st.form_submit_button(
            label="Submit", on_click=on_component_select, args=(c_selected_idx,)
        )


html_banner = """
<div style="
    border: 0px;
    background-color: dimgray;
    padding: 10px;
    border-radius: 5px;
    width: 100%;
    text-align: center;
    margin: 0 auto;
">
    <p style="font-size: 18px; color: white; margin: 0;">
        {content}
    </p>
</div>
"""

html_small = """<p style='font-size:13px; color: grey;'>{content}</p>"""

#
#
#
#
# INTERPRETER
##############
############################
##########################################
########################################################

if session.p100 & (session.current_message != ""):
    session.log_filename, session.log_id = get_next_log_filename()
    logger()

    st.markdown(
        f"""
        <style>
            .small-rectangle {{
                position: fixed;
                top: 10px;
                right: 10px;
                width: 100px;
                height: 50px;
                z-index: 9999;
                padding: 10px;
                text-align: right;
                font-family: monospace;
                color: grey;
                font-size: 10px;
            }}
        </style>
        <div class="small-rectangle">
            <p>{session.log_id}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="text-align: right; font-size: 18px; font-family: monospace;">200 interpreter</div>',
        unsafe_allow_html=True,
    )

    session.p100_start_time = time.time()

    # llm_api_selection = "gpt-3.5-turbo-0125"
    # llm_api_selection = "gpt-4o-2024-05-13"
    llm_api_selection = "gpt-4o-2024-08-06"
    #llm_api_selection = "nvidia/nemotron-4-340b-instruct"
    #llm_api_selection = "o1-2024-12-17"
    # llm_api_selection = "o1-mini-2024-09-12"
    # llm_api_selection = "deepseek-reasoner"
    # llm_api_selection= "gemini-2.0-flash"
    # llm_api_selection= "gemini-1.5-pro"

    session.p100_llm_api_selection = llm_api_selection

    prompt = session.current_message
    session.p100_prompt = prompt

    session.p100_list_of_docs = list_of_docs
    session.p100_list_of_cnames = list_of_cnames

    with st.container(border=True):
        st.markdown(f"*{prompt}*")

    # classify input as a photonic layout prompt or not
    session["input_prompt"] = prompt
    interpreter_cat = llm_api.intent_classification(prompt)

    """ if interpreter_cat.category_id != 1:
        st.markdown(f"**{interpreter_cat.response}**") """

    if True:
        with st.spinner("Entity extraction ..."):
            # session.p200_clarity = llm_api.verify_input_clarity(prompt)
            # if not session.p200_clarity['input_clarity']:
            #     st.write('```yaml\n'+yaml.dump(session.p200_clarity))

            session.p200_pretemplate = llm_api.entity_extraction(prompt)
            session.p200_pretemplate_copy = copy.deepcopy(session.p200_pretemplate)
            session.p200_preschematic = llm_api.preschematic(session.p200_pretemplate)

            col1, col2 = st.columns(2)
            with col2:
                try:
                    st.write("Initial schematic:")
                    st.graphviz_chart(session.p200_preschematic)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.write("Failed to render:\n```dot\n" + session.p200_preschematic)

            with col1:
                st.write("Extracted entities:")
                st.write(
                    "```yaml\n"
                    + yaml.dump(session.p200_pretemplate, sort_keys=False, width=55)
                )

                # session.p200_pretemplate = llm_api.refine_pretemplate(session)

                # edited_yaml_str = st.text_area('Edit pretemplate',
                #                                yaml.dump(session.p200_pretemplate,
                #                                          sort_keys=False,
                #                                          indent=4,
                #                                          width=55),
                #                                 height=300,
                #                                 label_visibility='collapsed')
                # if st.button('update'):
                #     yaml_data = yaml.safe_load(edited_yaml_str)

        with st.spinner("Searching design library ..."):
            if len(session.p200_pretemplate["components_list"]):
                # search templates
                session.p200_templates_search_r = llm_api.llm_search(
                    prompt, list(templates_dict.values())
                )
                session.p200_retreived_templates = [
                    (list(templates_dict.items())[i])
                    for i in session.p200_templates_search_r.match_list
                ]

                # search components
                session.p200_componenets_search_r = []
                # session.p200_retreived_components = []
                for c in session.p200_pretemplate["components_list"]:
                    r = llm_api.llm_search(c, session.p100_list_of_docs)
                    session.p200_componenets_search_r.append(r)
                    # session['p200_retreived_components'].append(r.match_list)

            if "p200_componenets_search_r" in session:
                st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
                col3, col4 = st.columns(2)
                with col3:
                    st.markdown(
                        html_banner.format(content="🛠️ build a new circuit"),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div style='height: 20px;'></div>", unsafe_allow_html=True
                    )
                    display_components_columns()
                    # st.markdown(html_small.format(content=session.p200_componenets_search_r.comment_str), unsafe_allow_html=True)

            if session["p200_retreived_templates"]:
                with col4:
                    st.markdown(
                        html_banner.format(content="🧩 use a template"),
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        "<div style='height: 20px;'></div>", unsafe_allow_html=True
                    )
                    display_templates_columns()
                    # st.markdown(html_small.format(content=session.p200_templates_search_r.comment_str), unsafe_allow_html=True)
    logger()

if not session.p100:
    st.markdown(
        f"""
        <style>
            .small-rectangle {{
                position: fixed;
                top: 10px;
                right: 10px;
                width: 100px;
                height: 50px;
                z-index: 9999;
                padding: 10px;
                text-align: right;
                font-family: monospace;
                color: grey;
                font-size: 10px;
            }}
        </style>
        <div class="small-rectangle">
            <p>{session.log_id}</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="text-align: right; font-size: 18px; font-family: monospace;">200 interpreter</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(f"*{session.current_message}*")

    col1, col2 = st.columns(2)
    with col1:
        st.write("```yaml\n" + yaml.dump(session.p200_pretemplate, width=55))

    with col2:
        try:
            st.graphviz_chart(session.p200_preschematic)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Failed to render:\n```dot\n" + session.p200_preschematic)

def map_pretemplate_to_template():
    pretemplate_dict = session.p200_pretemplate

    link = "(link)"
    labels = [""]

    template_dict = {
        "doc": {
            "title": pretemplate_dict.get("title", ""),
            "description": pretemplate_dict.get("brief_summary", ""),
            "reference": link,
            "labels": labels,
        },
        "nodes": {},
        "edges": pretemplate_dict.get("circuit_instructions", ""),
        "properties": {},
    }

    # Mapping components_list to nodes
    components = pretemplate_dict.get("components_list", [])
    for i, component in enumerate(components, start=1):
        node_label = f"N{i}"
        template_dict["nodes"][node_label] = {"component": component}

    session.p300_circuit_dsl = template_dict


if session.components_selected:
    session.p200_pretemplate["components_list"] = session.p200_selected_components
    map_pretemplate_to_template()
    # st.write('```yaml\n'+yaml.dump(session.p300_circuit_dsl))
    session["p300"] = True

# if 'p200_selected_template' in session:
if session.template_selected:
    template_id = session["p200_selected_template"]
    with st.container(border=True):
        st.markdown(f"Selection: *{template_id}*\n\n")
        st.write(
            templates_dict[template_id]["doc"]["title"]
            + "\n"
            + templates_dict[template_id]["doc"]["description"]
        )
        st.markdown(f"[reference]({templates_dict[template_id]['doc']['reference']})")

        st.write(templates_dict[template_id]["doc"]["title"])
        st.write("For this item, these specifications are required:")
        # st.write(str(templates_dict[template_id]['properties']['specs']))

        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                specs_dict = templates_dict[template_id]["properties"]["specs"]
                for key, item in specs_dict.items():
                    # specs_dict[key]['value'] = st.text_input(f"{key} ({item['comment']})", item['value'])
                    user_input = st.text_input(
                        f"{key} ({item['comment']})", item["value"]
                    )
                    session.user_specs[key] = {
                        "value": user_input,
                        "comment": item["comment"],
                    }

                if st.button("Update"):
                    session.updated_specs = yaml.dump(
                        session.user_specs, default_flow_style=False
                    )

if "updated_specs" in session:
    # if session.updated_yaml:
    session["p200_user_specs"] = session.updated_specs
    llm_parsed_user_spec = llm_api.parse_user_specs(session)

    # st.write(f"⇨ {llm_parsed_user_spec}")

    parsed_spec = yaml.safe_load(llm_parsed_user_spec)
    if "Error" in parsed_spec:
        st.write(parsed_spec)
        st.write("Let's try again!")
        del session.user_specs
        # time.sleep(5)
        # st.rerun()
    else:
        st.write("Got the specs!")

        session["p300_circuit_dsl"] = templates_dict[template_id]
        session["p300_circuit_dsl"]["properties"]["specs"] = parsed_spec

        if "TEMPLATE" in template_id:
            st.write("Looking for components...")
            # look for compoenents
            for key, value in session["p300_circuit_dsl"]["nodes"].items():
                try:
                    user_specs = session["p300_circuit_dsl"]["properties"]["specs"]

                    # iterate over the user_specs and remove comments
                    for spec_key in user_specs:
                        if "comment" in user_specs[spec_key]:
                            del user_specs[spec_key]["comment"]
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    user_specs = ""

                # r is a list of ids from 100_list_of_docs
                r = llm_api.llm_retrieve(
                    value + f"\n({str(user_specs)})",
                    session["p100_list_of_docs"],
                    session["p100_llm_api_selection"],
                )

                all_retrieved = [session["p100_list_of_cnames"][i] for i in r]
                st.write(f"For {value}, found: " + "\n".join(all_retrieved))

                selected_component = session["p100_list_of_cnames"][r[0]]
                session["p300_circuit_dsl"]["nodes"][key] = {}
                session["p300_circuit_dsl"]["nodes"][key]["component"] = {}
                session["p300_circuit_dsl"]["nodes"][key]["component"] = (
                    selected_component
                )
                # session['p300_circuit']['instances'][key]['component'] = selected_component

        session["p300"] = True

#
#
#
#
# SCHEMATIC
##############
############################
##########################################
########################################################

if "p300" in session:
    logger()
    st.markdown(
        '<div style="text-align: right; font-size: 18p`x; font-family: monospace;">300 schematic</div>',
        unsafe_allow_html=True,
    )

    session["p300_circuit_dsl"] = get_ports_info(session["p300_circuit_dsl"])
    print(session["p300_circuit_dsl"])
    session["p300_circuit_dsl"] = get_params(session["p300_circuit_dsl"])

    if not session.template_selected:
        session["p300_circuit_dsl"] = llm_api.apply_settings(session)

    with st.expander("Circuit draft", expanded=False):
        st.write("```yaml\n" + yaml.dump(session["p300_circuit_dsl"]))

    with st.spinner("Working on the schematic..."):
        session["p300_dot_string_draft"] = utils.circuit_to_dot(
            session["p300_circuit_dsl"]
        )

        if not session.template_selected:
            if len(session["p300_circuit_dsl"]["nodes"]) > 0:
                session["p300_dot_string"] = llm_api.dot_add_edges(session)
                session["p300_dot_string"] = llm_api.dot_verify(
                    session
                )  # TODO: Can we do this without LLM?

                #######
                # A hacky solution to avoid crossing edges TODO: come up with a better solution?
                for attempt in range(4):
                    print("\n\n+++++++++++++++++++++2")
                    print(session["p300_dot_string"])
                    print("+++++++++++++++++++++2\n\n")
                    happy_flag = utils.dot_planarity(session["p300_dot_string"])
                    if happy_flag:
                        break  # Exit the loop if happy_flag is True
                    else:
                        st.markdown(
                            ":red[Crossing edges found! Redoing the graph edges...]"
                        )
                        # st.graphviz_chart(session["p300_dot_string"])
                        # session["p300_dot_string"] = ""
                        session["p300_dot_string"] = llm_api.dot_add_edges_errorfunc(session)
                        session["p300_dot_string"] = llm_api.dot_verify(
                            session
                        )  # TODO: Can we do this without LLM?
                #######
        else:
            session["p300_dot_string"] = llm_api.dot_add_edges_templates(session)

        session["p300_dot_string"] = llm_api.dot_verify(
            session
        )  # TODO: Can we do this without LLM?

    with st.expander("Schematic diagram", expanded=False):
        st.write("```dot\n" + session["p300_dot_string"])

    st.graphviz_chart(session["p300_dot_string"])
    session["p300_circuit_dsl"] = utils.edges_dot_to_yaml(session)

    # get initial placements from dot
    session["p300_footprints_dict"], session["p300_circuit_dsl"] = footprint_netlist(
        session["p300_circuit_dsl"]
    )
    session["p300_dot_string_scaled"] = utils.dot_add_node_sizes(
        session["p300_dot_string"],
        utils.multiply_node_dimensions(session["p300_footprints_dict"], 0.01),
    )
    session["p300_graphviz_node_coordinates"] = utils.get_graphviz_placements(
        session["p300_dot_string_scaled"]
    )
    session["p300_graphviz_node_coordinates"] = utils.multiply_node_dimensions(
        session["p300_graphviz_node_coordinates"], 100 / 72
    )

    session["p300_circuit_dsl"] = utils.add_placements_to_dsl(session)
    session["p300_circuit_dsl"] = utils.add_final_ports(session)

    with st.expander("Circuit draft, updated", expanded=False):
        st.write("```yaml\n" + yaml.dump(session["p300_circuit_dsl"]))

    session["p300_dot_string"] = session["p300_dot_string"]
    session["p300_circuit_dsl"] = session["p300_circuit_dsl"]
    
    if session.optimizer_mode:
        session["p600"] = True
    else:
        session["p400"] = True
    
    logger()

#
#
#
#
# LAYOUT
##############
############################
##########################################
########################################################

if "p400" in session:
    logger()
    st.markdown(
        '<div style="text-align: right; font-size: 18px; font-family: monospace">400 layout</div>',
        unsafe_allow_html=True,
    )

    session["p400_gf_netlist"] = utils.dsl_to_gf(session["p300_circuit_dsl"])

    with st.expander("GDS-Factory Netlist", expanded=False):
        st.write("```yaml\n" + yaml.dump(session["p400_gf_netlist"]))

    with st.spinner("Rendering the GDS ..."):
        # GDS render
        try:
            d = yaml_netlist_to_gds(session, ignore_links=False)
            routing_flag = True
        except Exception as e:
            st.error(f"An error occurred: {e}")
            d = yaml_netlist_to_gds(session, ignore_links=True)
            st.markdown(":red[Routing error.]")
            routing_flag = False
            pass

    st.pyplot(session["p400_gdsfig"])
    # st.image('plot_gds.png')

    with st.spinner("Simulating s-parameters ..."):
        # SAX
        wl = np.linspace(1.5, 1.6, 200)
        result = session["p400_sax_circuit"](wl=wl)
        print(result[('o1', 'o1')])
        p400_sax_fig = utils.plot_dict_arrays(wl, result)
        st.pyplot(p400_sax_fig)
        # st.image(str(PATH.build / "plot_sax.png"))

    logger()

    ###########
    ########### circuit optimizer

    optimize_flag = False
    if "properties" in session["p300_circuit_dsl"]:
        if "optimizer" in session["p300_circuit_dsl"]["properties"]:
            if "error_fn" in session["p300_circuit_dsl"]["properties"]["optimizer"]:
                if (
                    "free_params"
                    in session["p300_circuit_dsl"]["properties"]["optimizer"]
                ):
                    if (
                        "sparam"
                        in session["p300_circuit_dsl"]["properties"]["optimizer"]
                    ):
                        if routing_flag:
                            optimize_flag = 0  # True

    if optimize_flag:
        with st.spinner("Optimizing circuit..."):
            session["p400_gf_netlist"] = circuit_optimizer(session)

        with st.expander("OPTIMIZED GDS-Factory Netlist", expanded=False):
            st.write("```yaml\n" + yaml.dump(session["p400_gf_netlist"]))

        try:
            d = yaml_netlist_to_gds(session, ignore_links=False)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            d = yaml_netlist_to_gds(session, ignore_links=True)
            st.markdown(":red[Routing error.]")
            pass

        # SAX
        wl = np.linspace(1.53, 1.57, 500)
        result = session["p400_sax_circuit"](wl=wl)
        p400_sax_fig = utils.plot_dict_arrays(wl, result)
        # st.pyplot(session['p400_sax_plot'])
        st.image(str(PATH.build / "plot_sax.png"))

    ###########
    ########### time
    session.p100_end_time = time.time()
    session.p100_runtime = session.p100_end_time - session.p100_start_time
    st.write(f"run time: {round(session.p100_runtime,2)} seconds")
    logger()

#
#
#
#
# OPTIMIZER
##############
############################
##########################################
########################################################

if "p600" in session:

    logger()
    st.markdown(
        '<div style="text-align: right; font-size: 18px; font-family: monospace">600 Device Optimizer</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Optimizing Device ..."):
        """ try: """
        res, sparams = device_optimizer(session)
        routing_flag = True
        """ except Exception as e:
            st.error(f"An error occurred: {e}")
            routing_flag = False
            pass """
    
    st.markdown(
        '<div style="text-align: left; font-size: 18px">Convergance Plot</div>',
        unsafe_allow_html=True,
    )
        
    st.image("build/opt_convergance.png")

    st.markdown(
        '<div style="text-align: left; font-size: 18px">Optimized Parameters</div>',
        unsafe_allow_html=True,
    )

    st.image("build/opt_varparams.png")

    st.markdown(
        '<div style="text-align: left; font-size: 18px">Optimized Scattering Parameters</div>',
        unsafe_allow_html=True,
    )

    st.image("build/opt_finalsparam.png")

    logger()