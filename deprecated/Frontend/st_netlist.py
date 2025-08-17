import requests
import streamlit as st
from fireworks.client import Fireworks
from groq import Groq
from openai import OpenAI

st.markdown("### Photonic Netlist Generator")

st.chat_message("assistant").write("How can I help?")

# Sidebar radio buttons to select API
api_selection = st.sidebar.radio(
    "Select API:",
    (
        "fireworks: llama-v3-70b-instruct",
        "openai: gpt-4o",
        "openai: gpt-3.5",
        "groq: llama3-70b",
        "groq: llama3-8b",
        "fireworks: qwen2-72b-instruct",
    ),
)

sys_prompt_classify = """You are a helpful assistant to a photonic engineer.
Extract the described components from the input text. This might be only one component, or many different components.
If the input text describes many copies of the same component, only count it once.
Output the extracted component(s) as a YAML object. For each component, include all provided specifications (if any) in the component name.

Here's an example:
Input Text: Three nodes: a 1x2 beamplitter with loss insersion loss, and two MZI modulators with GHz bandwidths. Ports o2 and o3 of the beamsplitter are connected to modulators.
components:
  C1: "1x2 beamplitter with loss insertion loss"
  C2: "MZI modulator with GHz bandwidth"
"""

sys_prompt_json = """Based on the input text, generate a JSON netlist for a circuit. Do not explain; only output the JSON.
Name ports as o1, o2, o3, ..., for all nodes.

Here's an example:
Input Text: Three nodes: a 1x2 beamplitter, and two MZI modulators. Ports o2 and o3 of the beamsplitter are connected to modulators.
{
  "connections": {
    "C1,o2": "C2,o1",
    "C1,o3": "C3,o1"
  },
  "placements": {
    "C1": {"x": 0, "y": 0,},
    "C2": {"x": 50, "y": 50,}
    "C3": {"x": 50, "y": -50,}
  },
  "instances": {
    "C1": {"component": "1x2 beamplitter"},
    "C2": {"component": "MZI modulator"},
    "C3": {"component": "MZI modulator"}
  },
  "name": "new_circuit"
}
INPUT TEXT: """

sys_prompt_dot = """Based on the input text, generate a Graphviz dot graph object for a circuit. Do not explain; only output the graph.
Name ports as o1, o2, o3, ..., for all nodes. Use undirected graphs. Do not preamble with "```dot".

Here's an example:
Three nodes: a 1x2 beamplitter, and two MZI modulators. Ports o2 and o3 of the beamsplitter are connected to modulators.
graph G {
    rankdir=LR;
    node [shape=record];

    beamsplitter [label="{<o1> o1 | 1x2 beamplitter | {<o2> o2 | <o3> o3}}"];
    mzi1 [label="{<i1> o1 | MZI Modulator | <o2> o2}"];
    mzi2 [label="{<i1> o1 | MZI Modulator | <o2> o2}"];

    beamsplitter:o2 -- mzi1:o1;
    beamsplitter:o3 -- mzi2:o1;
}
"""

sys_prompt_dot_simple = """Based on the input text, generate a Graphviz dot graph object for a circuit. Do not explain; only output the graph.
Use undirected graphs. Do not preamble with "```dot".

Here's an example:
Three nodes: a 1x2 beamplitter, and two MZI modulators. The beamsplitter is connected to both modulators.
graph G {
    beamsplitter [label="1x2 Beamsplitter"];
    modulator1 [label="MZI Modulator 1"];
    modulator2 [label="MZI Modulator 2"];

    beamsplitter -- modulator1;
    beamsplitter -- modulator2;
}
"""

sys_prompt2 = """Convert this JSON to a Graphviz dot object.
Make sure to preserve the number of nodes.
Use an undirected graph.
Do not explain.
INPUT JSON: """


def call_fireworks(prompt, sys_prompt, _model="llama-v3-70b-instruct"):
    if _model == "llama-v3-70b-instruct":
        model = "accounts/fireworks/models/llama-v3-70b-instruct"
    elif _model == "qwen2-72b-instruct":
        model = "accounts/fireworks/models/qwen2-72b-instruct"

    client = Fireworks(api_key="n4SbOjxiQ8xaemar8YsHAsugFQl13GSakT7RuoWdnywSrPX7")
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def call_groq(prompt, sys_prompt, model="llama3-8b-8192"):
    client = Groq(api_key="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")
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def call_openai(prompt, sys_prompt, model="gpt-4o"):
    client = OpenAI(api_key="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")
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def call_llm(prompt, sys_prompt, api_selection):
    if api_selection == "openai: gpt-4o":
        return call_openai(prompt, sys_prompt, "gpt-4o")
    elif api_selection == "openai: gpt-3.5":
        return call_openai(prompt, sys_prompt, "gpt-3.5-turbo")
    elif api_selection == "groq: llama3-70b":
        return call_groq(prompt, sys_prompt, "llama3-70b-8192")
    elif api_selection == "groq: llama3-8b":
        return call_groq(prompt, sys_prompt, "llama3-8b-8192")
    elif api_selection == "fireworks: llama-v3-70b-instruct":
        return call_fireworks(prompt, sys_prompt, "llama-v3-70b-instruct")
    elif api_selection == "fireworks: qwen2-72b-instruct":
        return call_fireworks(prompt, sys_prompt, "qwen2-72b-instruct")


def fetch_svg(graphviz_dot):
    url = "https://quickchart.io/graphviz"
    response = requests.get(url, params={"graph": graphviz_dot})
    if response.status_code == 200:
        return response.text
    else:
        return None


def process_prompt(prompt, api_selection):
    st.chat_message("user").write(prompt)

    response0 = call_llm(prompt, sys_prompt_classify, api_selection)
    st.chat_message("assistant").write("List of components:\n```yaml\n" + response0)

    response1 = call_llm(prompt, sys_prompt_json, api_selection)
    # st.chat_message("assistant").write(response1)
    # st.json(response1)
    # st.markdown(f"```json\n{response1}\n```")
    st.chat_message("assistant").write("Netlist in JSON:\n```json\n" + response1)

    # Step 3: Call LLM with sys_prompt_dot
    response2 = call_llm(prompt, sys_prompt_dot_simple, api_selection)
    st.chat_message("assistant").write("Graph in dot:\n```dot\n" + response2)
    # st.chat_message("assistant").markdown(f"```graphviz\n{response2}\n```")

    st.graphviz_chart(response2)

    # svg = fetch_svg(response2)
    # if svg:
    #     st.image(svg, use_column_width=True)


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
    "A splitter with four ports. each port is connected to a modulator.",
    "Five cascaded MZIs, each with two input and two output ports.",
    "A 2d mesh of 4x4 MZIs. Each MZI has two input and two outputs and they are back to back connected",
    "Layout 1x2 MMIs connected to each other to for a 1x8 splitter tree",
    "A 1x2 mzi with 200 dB extinction ratio",
]

st.sidebar.markdown("### Example Prompts")
for example in example_prompts:
    if st.sidebar.button(example):
        process_prompt(example, api_selection)

if prompt := st.chat_input():
    process_prompt(prompt, api_selection)
