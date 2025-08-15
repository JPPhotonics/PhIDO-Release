# import glob
import importlib
import sys

# from PIL import Image
# import mpld3
import uuid
from pathlib import Path

import gdsfactory as gf
import jax.numpy as jnp

# from io import BytesIO
import matplotlib.pyplot as plt
import sax
import yaml
from bayes_opt import BayesianOptimization
from gdsfactory.generic_tech import get_generic_pdk

from PhotonicsAI.config import PATH

# import copy
from PhotonicsAI.Photon import utils

genericPDK = get_generic_pdk()

LAYER = genericPDK.layers
layer_views = genericPDK.layer_views
layer_stack = genericPDK.layer_stack
cross_sections = genericPDK.cross_sections

# import sys
# sys.path.append("/Users/vahid/Downloads/PhotonicsAI_Project")


def list_python_files(directory):
    """Lists all Python files in the given directory using pathlib, excluding __init__.py.

    Args:
        directory: The directory to search for Python files.
    """
    return [f.stem for f in Path(directory).glob("*.py") if f.name != "__init__.py"]


def import_modules(module_names):
    """Imports the specified function from each module in the given package and returns them in a dictionary.

    Args:
        module_names: List of module names to import.
    """
    functions = {}
    for module_name in module_names:
        full_module_name = f"PhotonicsAI.KnowledgeBase.DesignLibrary.{module_name}"
        module = importlib.import_module(full_module_name)
        func = getattr(module, module_name)
        globals()[module_name] = func
        functions[module_name] = func
    return functions


def import_models(module_names):
    """Imports the specified function from each module in the given package and returns them in a dictionary.

    Args:
        module_names: List of module names to import.
    """
    models_dict = {}
    for module_name in module_names:
        full_module_name = f"PhotonicsAI.KnowledgeBase.DesignLibrary.{module_name}"
        module = importlib.import_module(full_module_name)
        func = module.get_model
        models_dict.update(func())
        # globals()[module_name] = func()
    return models_dict


# module_names = list_python_files("../KnowledgeBase/DesignLibrary/")
module_names = list_python_files(PATH.pdk)
cells = import_modules(module_names)
all_models = import_models(module_names)

DemoPDK = gf.Pdk(
    name="DemoPDK",
    layers=LAYER,
    cross_sections=cross_sections,
    cells=cells,
    layer_views=layer_views,
)
DemoPDK.activate()

DemoPDK = gf.Pdk(
    name="DemoPDK",
    layers=LAYER,
    cross_sections=cross_sections,
    cells=cells,
    layer_views=layer_views,
)
DemoPDK.activate()


def generate_unique_identifier(length: int = 32) -> str:
    """Generate a unique identifier of the specified length.

    Args:
        length: The length of the unique identifier.
    """
    # Generate a UUID
    unique_id = str(uuid.uuid4()).replace("-", "")
    # Truncate or repeat to get the desired length
    if length <= len(unique_id):
        return unique_id[:length]
    else:
        return (unique_id * (length // len(unique_id) + 1))[:length]


def yaml_netlist_to_gds(session, ignore_links=False):
    """Converts a YAML netlist to a GDS file.

    Args:
        session: The streamlit session object containing the YAML netlist.
        ignore_links: Whether to ignore the optical links in the netlist.
    """
    data = session["p400_gf_netlist"]

    # data = yaml.safe_load(netlist)

    if ignore_links:
        if "routes" in data:
            del data["routes"]

    if "name" in data:
        _id = generate_unique_identifier()
        data["name"] = f"{data['name']}_{_id}"

    if "reasoning" in data:
        del data["reasoning"]

    if "comments" in data:
        del data["comments"]

    if "placements" not in data:
        data["placements"] = {}
        x = y = 0
        for instance in data["instances"]:
            data["placements"][instance] = {"x": x, "y": y}
            x += 10
            y += 10

    modified_netlist = yaml.dump(data, default_flow_style=False, sort_keys=False)

    c = gf.read.from_yaml(modified_netlist)
    # gf_netlist_dict = c.get_netlist(recursive=False)

    try:
        gf_netlist_dict_recursive = c.get_netlist(recursive=True)
        required_models = sax.get_required_circuit_models(gf_netlist_dict_recursive)

        for name in required_models:
            if name not in all_models:
                print(
                    f"+++++++ MODEL ERROR: {name} is not available in all_models dictionary"
                )

    except Exception:
        print("++++++++++++ Recursive netlist failed")
        gf_netlist_dict_recursive = c.get_netlist(recursive=False)
        required_models = sax.get_required_circuit_models(gf_netlist_dict_recursive)
        required_models.append("ERROR: recursive netlist failed")
        pass

    _circuit, info = sax.circuit(
        gf_netlist_dict_recursive, all_models, backend="default"
    )

    gdsfig = c.plot(return_fig=True, show_labels=True)
    plt.savefig("build/plot_gds.png")
    plt.close(gdsfig)
    # js_fig = mpld3.fig_to_html(gdsfig)

    session["p400_gdsfig"] = gdsfig
    # session['p400_js_fig'] = js_fig
    # session['modified_netlist'] = modified_netlist
    # session['gf_netlist_dict'] = gf_netlist_dict
    # session['gf_netlist_dict_recursive'] = gf_netlist_dict_recursive
    session["p400_required_models"] = required_models
    session["p400_sax_circuit"] = _circuit

    return c, session


def footprint_netlist(circuit_dsl):
    """Get the footprint of each component in the circuit DSL and add it to the circuit DSL.

    Args:
        circuit_dsl: The circuit DSL in YAML containing the components.
    """
    nodes = circuit_dsl.get("nodes", {})
    instance_components = [
        [instance, details["component"]] for instance, details in nodes.items()
    ]
    foorprint = {}

    for _i, c in enumerate(instance_components):
        c_size = [
            DemoPDK.get_component(c[1]).dxsize,
            DemoPDK.get_component(c[1]).dysize,
        ]
        foorprint[c[0]] = (c_size[0], c_size[1])
        circuit_dsl["nodes"][c[0]]["properties"]["dx"] = c_size[0]
        circuit_dsl["nodes"][c[0]]["properties"]["dy"] = c_size[1]

    return foorprint, circuit_dsl


def get_params(circuit_dsl):
    """Get the parameters of each component in the circuit DSL and add it to the circuit DSL.

    Args:
        circuit_dsl: The circuit DSL in YAML containing the components.
    """
    for key, value in circuit_dsl["nodes"].items():
        gf_netlist = {}
        gf_netlist["instances"] = {}
        gf_netlist["instances"][key] = {}
        gf_netlist["instances"][key]["component"] = value["component"]

        c = gf.read.from_yaml(yaml.dump(gf_netlist))
        gf_netlist_updated = c.get_netlist(recursive=False)

        circuit_dsl["nodes"][key]["params"] = {}
        circuit_dsl["nodes"][key]["params"] = gf_netlist_updated["instances"][key][
            "settings"
        ]

    return circuit_dsl


db_docs = utils.search_directory_for_docstrings()
list_of_docs = [i["docstring"] for i in db_docs]
list_of_cnames = [i["module_name"] for i in db_docs]


def get_ports_info(circuit_dsl):
    """Get the ports information from the docstrings and add it to the circuit_dsl.

    Args:
        circuit_dsl: The circuit DSL in YAML containing the components.
    """
    for _c_name, c_data in circuit_dsl["nodes"].items():
        if "properties" not in c_data:
            c_data["properties"] = {}
        index = list_of_cnames.index(
            c_data["component"]
        )  # find the index of the component in the list
        # print(docs[index])
        docstring_dict = yaml.safe_load(list_of_docs[index].split("---", 1)[-1])
        c_data["properties"]["ports"] = docstring_dict["ports"]
        # c_data['info']['specs'] = docstring_dict['Specs']
        # c_data['info']['args'] = docstring_dict['Args']
        # c_data['info']['transmission_fn'] = docstring_dict['transmission_fn']

    return circuit_dsl


def info_netlist(d):
    """Get the information from the docstrings and add it to the netlist.

    Args:
        d: The netlist in YAML format.
    """
    netlist_dict = yaml.safe_load(d["netlist2"])
    docs = d["list_of_docs"]
    cnames = d["list_of_cnames"]

    try:
        for _c_name, c_data in netlist_dict["instances"].items():
            index = cnames.index(
                c_data["component"]
            )  # find the index of the component in the list
            docstring_dict = yaml.safe_load(docs[index])
            c_data["info"]["specs"] = docstring_dict["Specs"]
            c_data["info"]["args"] = docstring_dict["Args"]
            # c_data['info']['transmission_fn'] = docstring_dict['transmission_fn']

        updated_netlist = yaml.dump(netlist_dict, default_flow_style=False)
    except Exception:
        print("- could not parse compoenents docstrings")
        updated_netlist = d["netlist2"]

    return updated_netlist


def cosine(wav, **kwargs):
    """Returns the transmission function of a cosine filter.

    Args:
        wav: Wavelength in microns.
        kwargs: Dictionary of parameters.
    """
    lambda_0 = kwargs.get("design_wavelength")
    fsr = kwargs.get("fsr")
    arg = (2 * jnp.pi * lambda_0**2) / (wav * fsr)
    T = 0.5 * (1 + jnp.cos(arg))
    return T


def circuit_optimizer(session):
    """Optimize the circuit using Bayesian optimization.

    Args:
        session: The streamlit session object containing the circuit DSL and other information.
    """

    def fom_func(specs_dict, optparams, netlist_yaml):
        # get SAX simulation
        # c = mzi2(**optparams)

        netlist_dict = yaml.safe_load(netlist_yaml)

        if "routes" in netlist_dict:
            del netlist_dict["routes"]

        if "name" in netlist_dict:
            _id = generate_unique_identifier()
            netlist_dict["name"] = f"{netlist_dict['name']}_{_id}"

        # update dict settings (optparams)
        for k in optparams.keys():
            instance_name = k.split("____")[0]
            param_name = k.split("____")[1]
            netlist_dict["instances"][instance_name]["settings"][param_name] = float(
                optparams[k]
            )

        # for c_name, c_data in netlist_dict['instances'].items():
        #     for k in optparams.keys():
        #         instance_name = k.split('____')[0]
        #         param_name = k.split('____')[1]
        #         c_data['settings'][k] = float(optparams[k])
        #     break

        netlist_yaml = yaml.dump(netlist_dict, default_flow_style=False)

        c = gf.read.from_yaml(netlist_yaml)
        recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
        _c, info = sax.circuit(recnet, models=all_models)
        wl = jnp.linspace(1.51, 1.59, 200)
        S = _c(wl=wl)

        # Get the transmission function and pass the entire specs_dict
        transmission_fn_name = specs_dict.get("error_fn")
        # transmission_fn = locals()[transmission_fn_name]
        current_module = sys.modules[__name__]
        func = getattr(current_module, transmission_fn_name)

        T = func(wav=wl, **specs_dict)
        # plt.plot(wl, T)
        # plt.plot(wl, jnp.abs(S["o1", "o3"])**2, 'o')
        # now = datetime.datetime.now()
        # plt.savefig(f'T_{now}.png')
        # plt.close()
        # error = jnp.sum( jnp.abs(jnp.abs(S["o1", "o3"])**2 - T) )**2
        error = jnp.sum(jnp.abs(jnp.abs(S["o1", "o3"]) ** 2 - T) ** 2)

        return -error

    def objective(**kwargs):
        """Objective function for Bayesian optimization.

        Args:
            kwargs: Dictionary of parameters.
        """
        for param in kwargs:
            optparams[param] = kwargs[param]
        return fom_func(specs_dict, optparams, netlist_yaml)

    ##########

    circuit_dsl = session["p300_circuit_dsl"]
    netlist_yaml = yaml.dump(session["p400_gf_netlist"])

    # if user is specifying - LLM

    optimizable_specs = ["design_wavelength", "fsr"]
    specs_dict = {}
    specs_dict["error_fn"] = circuit_dsl["properties"]["optimizer"]["error_fn"]
    for key, value in circuit_dsl["properties"]["specs"].items():
        if key in optimizable_specs:
            specs_dict[key] = value["value"]

    params_dict = {}
    for i in circuit_dsl["properties"]["optimizer"]["free_params"]["passive"]:
        params_dict[i.replace(".", "____")] = {}
        if "length" in i:
            params_dict[i.replace(".", "____")] = (0, 200)
        if "volt" in i:
            params_dict[i.replace(".", "____")] = (0, 10)

    if not specs_dict:
        return None

    pbounds = params_dict
    optparams = {param: 0 for param in pbounds.keys()}

    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )

    optimizer.maximize(
        init_points=3,
        n_iter=3,
    )

    optimized_gf_netlist = session["p400_gf_netlist"]
    for k in optparams.keys():
        instance_name = k.split("____")[0]
        param_name = k.split("____")[1]
        optimized_gf_netlist["instances"][instance_name]["settings"][param_name] = (
            float(optimizer.max["params"][k])
        )

    return optimized_gf_netlist


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    # print(DemoPDK.cells)
    # DemoPDK.activate()

    netlist = """
    instances:
        C1:
            component: mzi_2x2_pn_diode
            info: {ports: 2x2}
            settings:
                dy: 100
                length: 400
        C2:
            component: _gc
            info: {ports: 1x0}
            settings:
                taper_length: 20
        C3:
            component: wdm_mzi1x4
            info: {ports: 1x4}
            settings:
                dy: 100
    placements:
        C1:
            x: 0
            y: 0
            mirror: True
        C2:
            mirror: True
            x: 1000
            y: 0
            rotation: 45
        C3:
            x: 0
            y: 1000
            rotation: 0
            mirror: True
    name: new_circuit
    """

    c = gf.read.from_yaml(netlist)
    c.plot()
    plt.show()

    # gds = yaml_netlist_to_gds(netlist)
