import hmac
import random
from datetime import datetime

import streamlit as st
import yaml

from PhotonicsAI.Photon import llm_api, utils

with open("prompts.yaml") as file:
    prompts = yaml.safe_load(file)


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


# if not check_password():
#     st.stop()  # Do not continue if check_password is not True.


# st.sidebar.markdown('Service Status:')
# status = get_backend_status()
# st.sidebar.markdown(f' {status}', unsafe_allow_html=True)

# Define custom CSS to adjust the sidebar width
st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 350px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        header {visibility: hidden;}
    </style>
""",
    unsafe_allow_html=True,
)

llm_api_selection = st.sidebar.radio(
    "LLM:",
    (
        "openai: gpt-4o",
        "fireworks: llama-v3-70b-instruct",
        "google: gemini-1.5-flash",
        #  'groq: llama3-70b'
    ),
)

st.markdown("### Photon Fury ⚡")
st.markdown("### A Furious Photonic Chip Engineer")
st.chat_message("assistant").write("How can I help?")

db_docs = utils.search_directory_for_docstrings()
list_of_docs = [i["docstring"] for i in db_docs]
list_of_cnames = [i["class_name"] for i in db_docs]


def logger(prompt):
    with open("./logs/st_prompt_logs.txt", "a") as file:
        file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {prompt}\n')


def verify_netlist(netlist, verified_components_list):
    print("------------")
    data_db = yaml.safe_load(verified_components_list)

    c_ = ""
    for key, _value in data_db.items():
        c_ += f""""{data_db[key]['component']}" with {data_db[key]['ports']} ports; """

    prompt_verify_netlist = prompts["verify_netlist"]
    prompt_verify_netlist.format_map({"netlist": netlist, "c_": c_})
    r = llm_api.call_llm(prompt_verify_netlist, "", llm_api_selection)
    print()
    print(r)
    return r


def process_prompt(prompt, llm_api_selection):
    logger(prompt)
    d = {
        "prompt": prompt,
        #  'prompts': prompts,
        "llm_api_selection": llm_api_selection,
        "list_of_docs": list_of_docs,
        "list_of_cnames": list_of_cnames,
    }

    st.chat_message("user").write(prompt)

    message = st.chat_message("assistant")

    d["guard"] = llm_api.call_llm(d["prompt"], prompts["guard"], llm_api_selection)
    if d["guard"] == "False":
        d["prompt"] = random.choice(d["list_of_cnames"])
        message.write(
            f"Cheeky! I cannot help you with that. Enjoy this gorgeous {d['prompt']} instead."
        )
        # d['prompt'] = 'a waveguide'

    d["raw_component_list"] = llm_api.call_llm(
        d["prompt"], prompts["classify"], llm_api_selection
    )

    message.write("List of components:\n```yaml\n" + d["raw_component_list"])

    d["db_component_list"], d["modified_yaml_data_all"] = (
        llm_api.yaml_components_search(d)
    )

    # message.write(f"Total items in db: {len(list_of_docs)}. Complete list of retrieved items:\n```yaml\n"+d['modified_yaml_data_all'])
    message.write(
        "Complete list of retrieved items:\n```yaml\n" + d["modified_yaml_data_all"]
    )
    message.write("Matched components from db:\n```yaml\n" + d["db_component_list"])

    r_netlist = llm_api.parsed_prompt_to_netlist(d)
    message.write("YAML Netlist:\n```yaml\n" + r_netlist)

    response2 = llm_api.call_llm(r_netlist, prompts["dot"], llm_api_selection)
    message.write("Circuit graph:\n```dot\n" + response2)
    message.graphviz_chart(response2)

    # st.chat_message("assistant").write("Fetching GDS...")
    # response = requests.post(PHOTONICDESIGNER_API+'/get_gds', json={"main_r": r.json()['main_r']})
    # st.image(img, caption='Generated Plot', use_column_width=True)

    # st.chat_message("assistant").write("Fetching scatter model...")
    # response = requests.post(PHOTONICDESIGNER_API+'/get_splot', json={"main_r": r.json()['main_r']})
    # st.image(img, caption='Generated Plot', use_column_width=True)


# Custom CSS to align button text to the left
st.markdown(
    """
    <style>
    .stButton>button {
        text-align: left;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

example_prompts = [
    # "A splitter with four ports. each port is connected to a modulator.",
    "A 1x2 MMI for 1310 nm",
    "A high speed modulator connected to a VOA",
    "A low loss 1x2 power splitter connected to two GHz modulators.",
    "A low loss 1x4 power splitter connected to four thermal modulators.",
    "Five cascaded MZIs, each with two input and two output ports.",
    "Cascaded 2x2 MZIs to create a switch network with 16 outputs",
    "A 2d mesh of nine MZIs. Each MZI has two input and two outputs and they are back to back connected",
    "Layout 1x2 MMIs connected to each other to for a 1x8 splitter tree",
    "a 1x2 splitter connected to two amplitude modulators with 100 dB extinction ratio",
    "eight low loss and low power thermo optic phase shifters. the phase shifters should be arranged in parallel in an array",
]

# Initialize session state for the selected prompt
if "selected_prompt" not in st.session_state:
    st.session_state.selected_prompt = None

st.sidebar.markdown("### Examples:")
for example in example_prompts:
    display_text = example[:35] + " ..."  # Limit text to first 100 characters
    if st.sidebar.button(display_text):
        process_prompt(example, llm_api_selection)


if prompt := st.chat_input():
    process_prompt(prompt, llm_api_selection)
