import base64
import io
import pickle
import subprocess
import time
from datetime import datetime

import graphviz
import numpy as np
import yaml
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

################################
# fury
from PhotonicsAI.Photon import llm_api, utils
from PhotonicsAI.Photon.DemoPDK import *

with open("prompts.yaml") as file:
    prompts = yaml.safe_load(file)

db_docs = utils.search_directory_for_docstrings()
list_of_docs = [i["docstring"] for i in db_docs]
list_of_cnames = [i["module_name"] for i in db_docs]

################################

app = FastAPI()

API_KEY = "iTL4i1LiqqxoY7W5f4hmoNZIFrZAjSn"  # Replace with your actual API key
API_KEY_NAME = "key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(api_key_header: str | None = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <html>
        <head>
            <title>Host Server</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Courier+Prime&display=swap');

                body {
                    background-color: #1e1e1e;
                    color: #d4d4d4;
                    font-family: 'Courier Prime', Courier, monospace;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    padding: 20px;
                    box-sizing: border-box;
                }
                .container {
                    text-align: center;
                }
                a {
                    color: #d16b9b; /* dirty pink */
                    text-decoration: none;
                    border: 2px solid #d16b9b; /* dirty pink */
                    padding: 10px 20px;
                    border-radius: 5px;
                    margin: 10px;
                    display: inline-block;
                    transition: background-color 0.3s, color 0.3s;
                }
                a:hover {
                    background-color: #d16b9b; /* dirty pink */
                    color: #1e1e1e;
                }
                h1 {
                    margin-bottom: 40px;
                    font-size: 2.5em;
                }
                p {
                    margin: 20px 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <p><a href="/fastapi/docs">api</a></p>
                <p><a href="/photon/">webapp</a></p>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/photon/")
async def redirect_streamlit_root(request: Request):
    host = request.headers.get("host", "localhost:8080")
    return RedirectResponse(url=f"http://{host}:8080/")


class InputData(BaseModel):
    prompt: str = "a 1x2 mmi"
    parameter1: int = 2
    parameter2: bool = False


def logger(prompt):
    with open("./logs/api_prompt_logs.txt", "a") as file:
        file.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} {prompt}\n')


def fig_to_png(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format="png")
    buf.seek(0)
    return buf.getvalue()


async def process_input_and_generate_response(input_data: InputData):
    start_time = time.time()
    d = {
        "prompt": input_data.prompt,
        #  'prompts': prompts,
        "llm_api_selection": "gpt-4o",
        "list_of_docs": list_of_docs,
        "list_of_cnames": list_of_cnames,
    }

    logger(d["prompt"])

    d = llm_api.interpreter_classify(d)
    # yield {"data": {"result": f"{d['interpret_classify_dict']}"}}

    if d["interpret_classify_dict"]["cat"] == 4:  # irrelevant
        yield "data: **I cannot help you with this question. I only know about integrated photonic devices and circuits.**"

    elif d["interpret_classify_dict"]["cat"] == 2:  # generic question
        yield "data: **I am still learning about integrated photonics. Soon I will be able to answer questions like this.**"

    elif d["interpret_classify_dict"]["cat"] == 3:  # mode simulation
        yield "data: **Mode simsulation is under development.**"

    elif d["interpret_classify_dict"]["cat"] == 1:  # layout
        ###########
        ########### interpret

        d = llm_api.interpreter_parse(d)

        # yield {"data": {"result": f"Interpreter\n```yaml\n{d['components_str']}"}}

        ###########
        ########### Search
        d["db_component_list"], d["db_components_list_all"] = (
            llm_api.yaml_components_search(d)
        )
        # yield {"data": {"result": "Search\n```yaml\n"+d['db_component_list']}}

        ###########
        ########### Netlist
        d["raw_netlist"] = llm_api.parsed_prompt_to_netlist(d)

        d["netlist1"] = settings_netlist(d["raw_netlist"])
        d["netlist2"] = llm_api.apply_settings(d)

        # yield {"data": {"result": "```yaml\n"+d['netlist2']}}

        ###########
        ########### DOT Graph
        d["dot_string"] = utils.netlist_to_dot(d["netlist2"])
        d["dot_string"] = llm_api.dot_add_edges(d)

        graph = graphviz.Source(d["dot_string"], format="png")
        graph_png = graph.pipe(format="png")
        # graph_base64 = base64.b64encode(graph_png).decode('utf-8')
        # yield {"data": {"image": graph_base64}}

        # yield {"data": {"result": "```dot\n"+d['dot_string']}}

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

        # yield {"data": {"result": "```yaml\n"+d['netlist5']}}

        try:
            d = yaml_netlist_to_gds(d, ignore_links=False)
        except:
            d = yaml_netlist_to_gds(d, ignore_links=True)
            # yield {"data": {"result": 'Routing error.'}}
            pass

        gds_png = fig_to_png(d["gdsfig"])
        # gds_png_base64 = base64.b64encode(gds_png.getvalue()).decode('utf-8')
        # yield {"data": {"image": gds_png_base64}}

        # yield {"data": {"result": d['required_models']}}

        wl = np.linspace(1.5, 1.6, 100)
        result = d["sax_circuit"](wl=wl)
        d["sax_plot"] = utils.plot_dict_arrays(wl, result)

        sax_png = fig_to_png(d["sax_plot"])
        # sax_png_base64 = base64.b64encode(sax_png.getvalue()).decode('utf-8')
        # yield {"data": {"image": sax_png_base64}}

        end_time = time.time()

        response = {}
        response["prompt"] = d["prompt"]
        response["interpret_classify_dict"] = d["interpret_classify_dict"]
        response["components_dict"] = d["components_dict"]
        response["dot_graph"] = d["dot_string"]
        response["netlist"] = d["netlist5"]
        response["graph_png"] = graph_png
        response["gds_png"] = gds_png
        response["sax_png"] = sax_png
        response["time_seconds"] = round(end_time - start_time, 2)

        response_binary = pickle.dumps(response)
        response_encoded = base64.b64encode(response_binary).decode("utf-8")
        yield {"data": {"pickled_response": response_encoded}}


# async def process_input_and_generate_response2(input_data: InputData):
#     steps = [
#         ("Analyzing input", 2),
#         ("Generating text response", 3),
#         ("Creating image", 4),
#         ("Finalizing response", 2)
#     ]

#     for step, delay in steps:
#         yield {"event": "update", "data": f"{step}..."}
#         await asyncio.sleep(delay)

#     yield {"event": "result", "data": {
#         "processed_text": input_data.text.upper(),
#         "parameter1_result": input_data.parameter1 * 2,
#         "parameter2_result": input_data.parameter2 * 3.14
#     }}

#     # Generate and yield a sample image
#     img = Image.new('RGB', (100, 100), color = (73, 109, 137))
#     img_byte_arr = io.BytesIO()
#     img.save(img_byte_arr, format='PNG')
#     img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

#     yield {"event": "image", "data": {"image": img_base64}}


@app.get("/fastapi/{path:path}", dependencies=[Depends(get_api_key)])
async def redirect_fastapi(path: str, request: Request):
    host = request.headers.get("host", "localhost:8080")
    return RedirectResponse(url=f"http://{host}:80/{path}")


@app.post("/photonapi", dependencies=[Depends(get_api_key)])
async def process_and_stream(input_data: InputData):
    return EventSourceResponse(process_input_and_generate_response(input_data))


if __name__ == "__main__":
    import uvicorn

    # Start the Streamlit app as a subprocess
    subprocess.Popen(
        ["streamlit", "run", "st_photonic_0.2.py", "--server.port", "8080"]
    )

    uvicorn.run("photon_api:app", host="0.0.0.0", port=80, reload=True)
