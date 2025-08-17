import os
import yaml
import re
import math
import pickle
import importlib
from typing import Any
from functools import cached_property

import gdsfactory as gf
from gdsfactory.generic_tech import get_generic_pdk
from gdsfactory.technology import LayerStack
from gplugins.common.base_models.component import LayeredComponentBase
from gplugins.tidy3d.util import get_port_normal, sort_layers

from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_gaussian_process

import tidy3d as td
from tidy3d.components.geometry.base import from_shapely

from misc_tests.Optimization.device_opt.SimulationSettings import SimulationSettingsTiny3DFdtd, SIMULATION_SETTINGS_LUMERICAL_TINY3D_DEFAULT

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

import tiktoken
import openai
# Configure OpenAI API Key
OPENAI_API_KEY = "ENTER API KEY"
client = openai.OpenAI(api_key=OPENAI_API_KEY)

import pathlib
PathType = pathlib.Path | str

########################################################################################################################################################

class Tidy3DSimulator():

    def __init__(self, component, settings: SimulationSettingsTiny3DFdtd = SIMULATION_SETTINGS_LUMERICAL_TINY3D_DEFAULT):

        self.settings= settings

        #self.comp_base = LayeredComponentBase()
        
        self.component = component

        #self.material_mapping = self.settings.material_mapping
        #self.reference_plane = "middle"
        
        self.simulation = None

        self.comp_base = LayeredComponentBase(component=self.component, layer_stack=self.settings.layer_stack, extend_ports=self.settings.extend_ports,
                                              port_offset=self.settings.port_offset, pad_xy_inner=self.settings.pad_xy_inner, pad_xy_outer=self.settings.pad_xy_outer,
                                              pad_z_inner=self.settings.pad_z, pad_z_outer=self.settings.pad_z)


    @cached_property
    def polyslabs(self) -> dict[str, tuple[td.Geometry, ...]]:
        """Returns a dictionary of PolySlab instances for each layer in the component.

        Returns:
            dict[str, tuple[td.PolySlab, ...]]: A dictionary mapping layer names to tuples of PolySlab instances.
        """
        slabs = {}
        layers = sort_layers(self.comp_base.geometry_layers, sort_by="mesh_order", reverse=True)
        for name, layer in layers.items():
            bbox = self.comp_base.get_layer_bbox(name)
            shape = self.comp_base.polygons[name].buffer(distance=0.0, join_style="mitre")
            geom = from_shapely(
                shape,
                axis=2,
                slab_bounds=(bbox[0][2], bbox[1][2]),
                dilation=0,
                sidewall_angle=np.deg2rad(layer.sidewall_angle),
                reference_plane="middle",
            )
            slabs[name] = geom

        return slabs

    @cached_property
    def structures(self) -> list[td.Structure]:
        """Returns a list of Structure instances for each PolySlab in the component.

        Returns:
            list[td.Structure]: A list of Structure instances.
        """
        structures = []
        for name, poly in self.polyslabs.items():
            structure = td.Structure(
                geometry=poly,
                medium=self.settings.material_mapping[self.comp_base.geometry_layers[name].material],
                name=name,
            )
            structures.append(structure)

        return structures

    def get_ports(self, mode_spec: td.ModeSpec) -> list[td.plugins.smatrix.Port]:

        ports = []
        for port in self.comp_base.ports:
        
            if port.port_type != "optical":
                continue

            center = self.comp_base.get_port_center(port)
            center = np.round(center, abs(int(np.log10(1e-6)))) # round to the nearest micron
            
            axis, direction = get_port_normal(port)

            match self.settings.port_size_mult:
                case float():
                    size = np.full(3, self.settings.port_size_mult * port.dwidth)
                case tuple():
                    size = np.full(3, self.settings.port_size_mult[0] * port.dwidth)
                    size[2] = self.settings.port_size_mult[1] * port.dwidth
            size[axis] = 0

            ports.append(
                td.plugins.smatrix.Port(
                    center=tuple(center),
                    size=tuple(size),
                    direction=direction,
                    mode_spec=mode_spec,
                    name=port.name,
                )
            )
        return ports
    
    def get_char_ports(self, mode_spec: td.ModeSpec) -> list[td.plugins.smatrix.Port]:

        "List of the first and last port. First port will be used as input, last port will be used as output for characterization"

        ports = []
        for port in [self.comp_base.ports[0], self.comp_base.ports[-1]]:
        
            if port.port_type != "optical":
                continue

            center = self.comp_base.get_port_center(port)
            center = np.round(center, abs(int(np.log10(1e-6)))) # round to the nearest micron
            
            axis, direction = get_port_normal(port)

            match self.settings.port_size_mult:
                case float():
                    size = np.full(3, self.settings.port_size_mult * port.dwidth)
                case tuple():
                    size = np.full(3, self.settings.port_size_mult[0] * port.dwidth)
                    size[2] = self.settings.port_size_mult[1] * port.dwidth
            size[axis] = 0

            ports.append(
                td.plugins.smatrix.Port(
                    center=tuple(center),
                    size=tuple(size),
                    direction=direction,
                    mode_spec=mode_spec,
                    name=port.name,
                )
            )
        return ports
    
    def create_simulation(self,
                          sources: tuple[Any, ...] | None = None,
                          monitors: tuple[Any, ...] | None = None
                          ) -> td.Simulation:

        sim_center = (*self.comp_base.center[:2], self.settings.layer_stack['core'].thickness/2) 
        sim_size = (*self.comp_base.size[:2], self.settings.layer_stack['core'].thickness/2 + 2*self.settings.pad_z)  

        grid_spec = self.settings.grid_spec

        boundary_spec = self.settings.boundary_spec

        return td.Simulation(
            center=sim_center,
            size=sim_size,
            grid_spec=grid_spec,
            medium = self.settings.material_mapping["sio2"],

            structures=self.structures,
            sources=[],
            monitors=[] if monitors is None else monitors,
            boundary_spec=boundary_spec,

            run_time=self.settings.run_time
        )

    def create_modesimulation(self):

        ldas = np.linspace(self.settings.wavelength - self.settings.bandwidth/2, self.settings.wavelength + self.settings.bandwidth/2, 101)  # wavelength range
        
        freq0 = td.C_0/self.settings.wavelength
        freqs = td.C_0 / ldas  # frequency range
        fwidth = 0.5 * (np.max(freqs) - np.min(freqs))
            

        char_ports = self.get_char_ports(td.ModeSpec(num_modes=self.settings.num_modes))

        source_port = char_ports[0]
        monitor_port = char_ports[1]

        mode_source = td.ModeSource(
                        center=source_port.center,
                        size=source_port.size,
                        source_time=td.GaussianPulse(freq0=freq0, fwidth=fwidth),
                        direction="+",
                        mode_spec=source_port.mode_spec,
                        mode_index=0,
                        )   

        # add a mode monitor to measure transmission at the output waveguide
        mode_monitor = td.ModeMonitor(
                        center=monitor_port.center,
                        size=monitor_port.size,
                        freqs=freqs,
                        mode_spec=monitor_port.mode_spec,
                        name="mode",
                        )

        # add a field monitor to visualize field distribution at z=t/2
        field_monitor = td.FieldMonitor(
            center=(0, 0, self.settings.layer_stack['core'].thickness/2), size=(td.inf, td.inf, 0), freqs=[freq0], name="field"
        )

        mode_sim = self.create_simulation(sources=[mode_source], monitors=[mode_monitor, field_monitor])

        return mode_sim

    def create_fdtdsimulation(self):
        
        ldas = np.linspace(self.settings.wavelength - self.settings.bandwidth/2, self.settings.wavelength + self.settings.bandwidth/2, 101)  # wavelength range
        
        freq0 = td.C_0/self.settings.wavelength
        freqs = td.C_0 / ldas  # frequency range
        fwidth = 0.5 * (np.max(freqs) - np.min(freqs))
        
        field_monitor = td.FieldMonitor(
            center=(0, 0, self.settings.layer_stack['core'].thickness/2), size=(td.inf, td.inf, 0), freqs=[freq0], name="field"
        )
        
        fdtd_sim = self.create_simulation(monitors=[field_monitor])
        
        return fdtd_sim

########################################################################################################################################################

tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_prompt(prompt, max_tokens=120000):
    # Tokenize the input prompt
    tokens = tokenizer.encode(prompt)
    
    # Check if the prompt exceeds the maximum allowed tokens
    if len(tokens) > max_tokens:
        # Truncate the prompt by keeping only the last `max_tokens` tokens
        tokens = tokens[-max_tokens:]
        
        # Decode tokens back to string
        truncated_prompt = tokenizer.decode(tokens)
        return truncated_prompt
    return prompt

def call_openai(prompt, sys_prompt='', model='gpt-4o'):
    '''
    calling openai. This is my persoanl account.
    '''
    prompt = truncate_prompt(prompt)
    
    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        # model="gpt-3.5-turbo",
        # model="gpt-4",
        # model="gpt-4-turbo",
        model=model,
        temperature= 0.1,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    return response.choices[0].message.content

def run_llm(simulator: Tidy3DSimulator, log_filepath):
    LOG_MAIN = log_filepath 
    with open(LOG_MAIN, "r", encoding="utf-8") as file:
        log_data = file.read()
    
    SYS_PROMPT = os.getcwd() + "\\SYS_PROMPT.txt" 
    with open(SYS_PROMPT, "r", encoding="utf-8") as file:
        sys_prompt = file.read()

    #print("################################################################################")
    #print("System Prompt Sent to OPENAI")
    #print("################################################################################")
    #print(sys_prompt)
    #print("################################################################################\n\n")

    data_to_send = "\n CURRENT VALUES \n min_steps_per_wvl: %f \n extend_ports: %f \n port_offset: %f \n pad_xy_inner: %f \n pad_xy_outer: %f \n pad_z: %f \n run_time: %s \n shutoff: %s \n wavelength: %f" % (simulator.settings.min_steps_per_wvl, simulator.settings.extend_ports, simulator.settings.port_offset, simulator.settings.pad_xy_inner, simulator.settings.pad_xy_outer, simulator.settings.pad_z, str(simulator.settings.run_time), str(simulator.settings.shutoff), simulator.settings.wavelength )
    
    data_to_send = log_data + data_to_send
    #print("################################################################################")
    #print("User Prompt Sent to OPENAI")
    #print("################################################################################")
    #print(data_to_send)
    #print("################################################################################\n\n")
    
    openai_response = call_openai(data_to_send, sys_prompt)
    
    print("################################################################################")
    print("Response from OPENAI")
    print("################################################################################")
    print(openai_response)
    print("################################################################################\n\n")

    suggested_param = openai_response[7:(len(openai_response)-3)]

    #print(ans)

    suggested_param_yaml = yaml.safe_load(suggested_param)

    #print(yaml_content)

    return suggested_param_yaml

########################################################################################################################################################

def run_modesimulation(simulator: Tidy3DSimulator):

    ldas = np.linspace(simulator.settings.wavelength - simulator.settings.bandwidth/2, simulator.settings.wavelength + simulator.settings.bandwidth/2, 101)  # wavelength range
    
    freqs = td.C_0 / ldas  # frequency range

    modesim = simulator.create_modesimulation()

    mode_spec = td.ModeSpec(num_modes=simulator.settings.num_modes)

    char_ports = simulator.get_char_ports(mode_spec=mode_spec)
    source_port = char_ports[0]

    mode_solver = td.plugins.mode.ModeSolver(
        simulation=modesim,
        plane=td.Box(center=source_port.center, size=source_port.size),
        mode_spec=mode_spec,
        freqs=freqs
    )

    mode_data = td.web.run(mode_solver, task_name="mode_solver", verbose=True)

    return modesim, mode_solver, mode_data

def run_modesimulation_llm(simulator: Tidy3DSimulator):

    try:
        modesim, mode_solver, mode_data = run_modesimulation(simulator)

        LOG_MAIN = os.getcwd() + "\\LOG_MAIN.txt"

        warning_found = False

        with open(LOG_MAIN, "r") as file:
            for line in file:
                if "WARNING" in line:
                    warning_found = True


        if warning_found == True:

            suggested_param_yaml = run_llm(simulator, LOG_MAIN)

            sim_settings_new = simulator.settings

            for key, val in suggested_param_yaml.items():
                if hasattr(sim_settings_new, key):
                    setattr(sim_settings_new, key, val)

            tinycomp_new = Tidy3DSimulator(component=simulator.component, settings=sim_settings_new)
                        
            modesim, mode_solver, mode_data = run_modesimulation(simulator=tinycomp_new)  

    except:
        print("\n\n################################################################################\n SIM FAILED, ASKING OPENAI \n################################################################################\n\n")

        suggested_param_yaml = run_llm(simulator, os.getcwd() + "\\LOG_MAIN.txt")

        #print(yaml_content)

        sim_settings_new = simulator.settings

        for key, val in suggested_param_yaml.items():
            if hasattr(sim_settings_new, key):
                setattr(sim_settings_new, key, val)

        #sim_settings = load_simulation_settings("SimulationSettingsGPT.py", "SimulationSettingsTiny3DFdtd")

        tinycomp_new = Tidy3DSimulator(component=simulator.component, settings=sim_settings_new)
        
        #tinycomp_new.settings = sim_settings_new
        
        modesim, mode_solver, mode_data = run_modesimulation(simulator=tinycomp_new)

    return modesim, mode_solver, mode_data

########################################################################################################################################################

def run_fdtdsimulation(simulator: Tidy3DSimulator, filepath: PathType | None = None):

    ldas = np.linspace(simulator.settings.wavelength - simulator.settings.bandwidth/2, simulator.settings.wavelength + simulator.settings.bandwidth/2, 101) 
    
    freqs = td.C_0 / ldas 

    fdtdsim = simulator.create_fdtdsimulation()

    ports = simulator.get_ports(td.ModeSpec(num_modes=1, filter_pol='te'))

    fdtd_solver = td.plugins.smatrix.ComponentModeler(simulation=fdtdsim, ports=ports, freqs=freqs, verbose=True, path_dir="build/data")

    smatrix = fdtd_solver.run()

    sp = {}

    for port_in in smatrix.port_in.values:
        for port_out in smatrix.port_out.values:
            for mode_index_in in smatrix.mode_index_in.values:
                for mode_index_out in smatrix.mode_index_out.values:
                    sp[f"{port_in}@{mode_index_in},{port_out}@{mode_index_out}"] = (
                        smatrix.sel(
                            port_in=port_in,
                            port_out=port_out,
                            mode_index_in=mode_index_in,
                            mode_index_out=mode_index_out,
                        ).values
                    )

    if (filepath==None):
        filepath = os.getcwd() + "/sparams.npz"
    else:
        filepath = filepath + "/sparams.npz"

    print("###################")
    print(filepath)

    frequency = smatrix.f.values
    sp["wavelengths"] = td.constants.C_0 / frequency
    np.savez_compressed(filepath, **sp)

    return fdtd_solver, sp

def run_fdtdsimulation_llm(simulator: Tidy3DSimulator, filepath: PathType | None = None):
    try:
        _, sp = run_fdtdsimulation(simulator, filepath)

        LOG_MAIN = os.getcwd() + "\\LOG_MAIN.txt"
        
        warning_found = False

        with open(LOG_MAIN, "r") as file:
            for line in file:
                if "WARNING" in line:
                    warning_found = True


        if warning_found == True:
            suggested_param_yaml = run_llm(simulator, LOG_MAIN)

            sim_settings_new = simulator.settings

            for key, val in suggested_param_yaml.items():
                if hasattr(sim_settings_new, key):
                    setattr(sim_settings_new, key, val)

            tinycomp_new = Tidy3DSimulator(component=simulator.component, settings=sim_settings_new)
                        
            fdtd_solver, smatrix = run_fdtdsimulation(simulator=tinycomp_new)  
            
    except:
        print("\n\n################################################################################\n SIM FAILED, ASKING OPENAI \n################################################################################\n\n")

        suggested_param_yaml = run_llm(simulator, os.getcwd() + "\\LOG_MAIN.txt")

        sim_settings_new = simulator.settings

        for key, val in suggested_param_yaml.items():
            if hasattr(sim_settings_new, key):
                setattr(sim_settings_new, key, val)

        #sim_settings = load_simulation_settings("SimulationSettingsGPT.py", "SimulationSettingsTiny3DFdtd")

        tinycomp_new = Tidy3DSimulator(component=simulator.component, settings=sim_settings_new)
        
        #tinycomp_new.settings = sim_settings_new
        
        sp_new = run_fdtdsimulation_llm(simulator=tinycomp_new, filepath=filepath)

    return sp_new

########################################################################################################################################################

class GP_BO:
    """Return optimized parameters for the given component based on the target figure of merit (FOM).

    Args:
        component: component to optimize
        const_params: constant parameters in the component
        var_params: parameters to optimize in the component
        param_bounds: bounds of the optimizable parameters
        fom: figure of merit function for the optimizer. It must always be a minimizing FOM.
        fom_args: arguments for the FOM
        acq_func: acquisition function for the Baysian Optimizer
        n_calls: total number of iterations for the optimizer
        n_random_starts: total number of random starts for the optimizer
        noise: noise added to the predicted model
        filepath: filepath to save FDTD simulation file, skopt optimized model .pickle file and sparameters .npz file
        graph_plot: if true, plots convergence and baysian optimizer progress graphs
        layer_stack: contains layer to thickness, zmin and material.
            Defaults to active pdk.layer_stack.
        simulation_settings: dataclass with all simulation_settings.
    """

    def __init__(
        self,
        component,
        const_params: list,
        var_params: list,
        param_bounds: list,
        fom,
        fom_args: list,
        acq_func: str,
        n_calls: int,
        n_random_starts: int,
        noise: float,
        filepath: str,
        graph_plot: bool,
        layer_stack: LayerStack | None = None,
        simulation_settings: SimulationSettingsTiny3DFdtd = SIMULATION_SETTINGS_LUMERICAL_TINY3D_DEFAULT,
    ):
        self.component = component
        self.const_params = const_params
        self.var_params = var_params
        self.param_bounds = param_bounds
        self.fom = fom
        self.fom_args = fom_args
        self.acq_func = acq_func
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.noise = noise
        self.filepath = filepath
        self.graph_plot = graph_plot
        self.layer_stack = layer_stack
        self.ss = simulation_settings

    def run_opt(self):
        """Skopt wrapper function implementing a Gaussian Process based Baysian Optimization.

        Return:
            skopt optimized model data structure saved as pickle file in the input filepath
            optimized sparameters saved as .npz file in the input filepath
        """
        self.res = gp_minimize(
            self.fdtd_sparams,
            self.param_bounds.values(),
            acq_func=self.acq_func,
            n_calls=self.n_calls,
            n_random_starts=self.n_random_starts,
            noise=self.noise**2,
            random_state=1234,
        )

        self.opt_sparams = self.get_sparams(self.res)

        if self.graph_plot:
            self.get_graphs()

        return self.res, self.opt_sparams

    def fdtd_sparams(self, param_set):
        """Helper function for the optimizer. Runs FDTD simulations on the input parameter set and return the figure of merit.

        Args:
            param_set: optimizable parameter set
        """
        #wg_width = self.const_params["wg_width"] or 0.41
        print("THIS IS THE FIX")
        for key, param in zip(self.var_params.keys(), param_set):
            self.var_params[key] = round(float(param), 2)

        c = self.component(**(self.const_params | self.var_params))

        custom_settings = SimulationSettingsTiny3DFdtd()

        tinycomp = Tidy3DSimulator(component=c, settings=custom_settings)

        _, smatrix = run_fdtdsimulation(tinycomp)

        opt = self.fom(smatrix, self.fom_args)
        
        print(opt)

        return opt

    def get_graphs(self):
        """Plots convergence graph for all cases. In case of a single optimizable parameter, plots baysian optimizer progress graphs."""
        if self.res.space.n_dims == 1:
            plt.figure(figsize=(3, 2))
            for n_iter in range(len(self.res.models)):
                # Plot true function.
                plt.subplot(len(self.res.models), 2, 2 * n_iter + 1)

                if n_iter == 0:
                    show_legend = True
                else:
                    show_legend = False

                ax = plot_gaussian_process(
                    self.res,
                    n_calls=n_iter,
                    noise_level=0.1,
                    show_legend=show_legend,
                    show_title=False,
                    show_next_point=False,
                    show_acq_func=False,
                )
                ax.set_ylabel("")
                ax.set_xlabel("")
                # Plot EI(x)
                plt.subplot(len(self.res.models), 2, 2 * n_iter + 2)
                ax = plot_gaussian_process(
                    self.res,
                    n_calls=n_iter,
                    show_legend=show_legend,
                    show_title=False,
                    show_mu=False,
                    show_acq_func=True,
                    show_observations=False,
                    show_next_point=True,
                )
                ax.set_ylabel("")
                ax.set_xlabel("")
            
            plt.savefig(self.filepath + "/opt_gaussian.png", dpi=300, bbox_inches='tight')

            plt.figure(figsize=(3, 2))
            plot_convergence(self.res)
            plt.savefig(self.filepath + "/opt_convergance.png", dpi=300, bbox_inches='tight')

        else:
            plt.figure(figsize=(3, 2))
            plot_convergence(self.res)
            plt.savefig(self.filepath + "/opt_convergance.png", dpi=300, bbox_inches='tight')

    def get_sparams(self, res):
        """Returns the final Sparameters of the optimized geometry.

        Args:
            res: skopt optimized model

        Return:
            sparams: Sparameter dictionary

        """
        #wg_width = self.const_params["wg_width"] or 0.5

        param_set = res["x"]

        for key, param in zip(self.var_params.keys(), param_set):
            self.var_params[key] = round(param, 2)

        c = self.component(**(self.const_params | self.var_params))

        custom_settings = SimulationSettingsTiny3DFdtd()

        tinycomp = Tidy3DSimulator(component=c, settings=custom_settings)

        _, smatrix = run_fdtdsimulation(tinycomp)

        np.savez(self.filepath + "/opt_finalsparams.npz", smatrix)

        return smatrix

def run_gpoptimizer(component, circuit_desl_yaml):

    circuit_desl = yaml.safe_load(circuit_desl_yaml)

    # Create FoM Fucntion

    fom_equation_str = circuit_desl["nodes"]["N1"]["FoM"]["equation"]
    fom_equation_direction = circuit_desl["nodes"]["N1"]["FoM"]["direction"]
    
    allowed_functions = {
        "abs": abs,
        "max": max,
        "min": min,
        "sqrt": np.sqrt,
        "real": np.real,
        "imag": np.imag,
        "conj": np.conj
    }

    sparam_map = {
        "S11": "o1@0,o1@0",
        "S12": "o2@0,o1@0",
        "S13": "o3@0,o1@0",
        "S14": "o4@0,o1@0",
        "S15": "o5@0,o1@0",

        "S21": "o1@0,o2@0",
        "S22": "o2@0,o2@0",
        "S23": "o3@0,o2@0",
        "S24": "o4@0,o2@0",
        "S25": "o5@0,o2@0",

        "S31": "o1@0,o3@0",
        "S32": "o2@0,o3@0",
        "S33": "o3@0,o3@0",
        "S34": "o4@0,o3@0",
        "S35": "o5@0,o3@0",

        "S41": "o1@0,o4@0",
        "S42": "o2@0,o4@0",
        "S43": "o3@0,o4@0",
        "S44": "o4@0,o4@0",
        "S45": "o5@0,o4@0",

        "S51": "o1@0,o5@0",
        "S52": "o2@0,o5@0",
        "S53": "o3@0,o5@0",
        "S54": "o4@0,o5@0",
        "S55": "o5@0,o5@0"
    }

    def fom(sparams, fom_equation_str):
        scope = dict(allowed_functions)  # safe built-ins
        
        # For each symbolic name, compute the mean over wavelengths
        for symbolic_name, dict_key in sparam_map.items():
            if dict_key not in sparams.keys():
                continue
            array = np.array(sparams[dict_key])
            mean_value = np.sum(np.abs(array) ** 2) / len(array)
            scope[symbolic_name] = mean_value
        
        if fom_equation_direction == "maximize":
            equation_str = f"-({equation_str})"

        return eval(fom_equation_str, {"__builtins__": {}}, scope)
    
    # Define Static Prameters and Variable Parameters with Bounds

    node = circuit_desl["nodes"]["N1"]
    params = node.get("params", {})
    opt_settings = node.get("opt_settings", {})

    variable_param_names = set(opt_settings.keys())

    variable_parameters = {k: round(v, 1) for k, v in params.items() if k in variable_param_names}

    static_parameters = {k: v for k, v in params.items() if k not in variable_param_names}

    variable_ranges = {k: (v["min"], v["max"]) for k, v in opt_settings.items()}

    # Run Optimizer
    
    ss = SimulationSettingsTiny3DFdtd()

    gp = GP_BO(
        component,
        static_parameters,
        variable_parameters,
        variable_ranges,  # the bounds on each dimension of x
        fom,
        fom_equation_str,
        acq_func="EI",  # the acquisition function
        n_calls=2,  # the number of evaluations of f
        n_random_starts=1,  # the number of random initialization points
        noise=0.1,  # the noise level (optional)
        filepath="build",
        graph_plot=True,
        simulation_settings=ss,
    )

    res, sparams = gp.run_opt()

    # Plotting

    sorted_pairs = {}
    print((variable_parameters.keys()))
    var_params = list(variable_parameters.keys())
    print(res['x_iters'])
    print(res['x_iters'][0])
    for i in range(0, len(res['x_iters'][0])):
        x = [item[i] for item in res['x_iters']]
        y = res['func_vals']
        print(var_params)
        sorted_pairs[var_params[i]] = sorted(zip(x, y))  # sorts by x
    
    print(sorted_pairs)

    if len(res['x_iters'][0]) > 1:
        fig, axs = plt.subplots(1, len(res['x_iters'][0]), figsize=(3*len(res['x_iters'][0]), 2))
        i = 0
        for label, xy in sorted_pairs.items():
            x_vals, y_vals = zip(*xy)
            axs[i].plot(x_vals, y_vals, marker='o', label=label)
            axs[i].set_title(f"FoM vs {label}")
            axs[i].grid(True)
            i = i+1
        plt.tight_layout()
        plt.savefig("build/opt_test_varparams.png", dpi=300)
        plt.close()
    else:
        x_vals, y_vals = zip(*sorted_pairs[var_params[0]])
        plt.figure(figsize=(3, 2))
        plt.plot(x_vals, y_vals, marker='o', label=(list(sorted_pairs.keys()))[0])
        plt.title(f"FoM vs {(list(sorted_pairs.keys()))[0]}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("build/opt_test_varparams.png", dpi=300)
        plt.close()

    #sparam_plot = {k: v for k, v in sparams.items() if k.split('@', 1)[0] == "o1"}
    sparam_plot = {k: v for k, v in sparams.items() if k != "wavelengths"}

    num_plots = len(sparam_plot)
    ncols = 2
    nrows = (num_plots + 1) // ncols

    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows), squeeze=False)
    axs = axs.flatten()

    for idx, (sparam, values) in enumerate(sparam_plot.items()):
        values = np.array(values)
        mag_db = 10 * np.log10(np.abs(values) + 1e-12)  # avoid log(0)
        
        axs[idx].plot(sparams["wavelengths"], mag_db)
        axs[idx].set_title(f"{sparam} (dB)")
        axs[idx].set_xlabel("Wavelength (μm)")
        axs[idx].set_ylabel("Magnitude (dB)")
        axs[idx].grid(True)

    # Hide any unused subplots
    for j in range(idx + 1, len(axs)):
        fig.delaxes(axs[j])

    fig.tight_layout()

    fig.savefig("build/opt_test_finalsparam.png")



    return res, sparams

########################################################################################################################################################

if __name__ == "__main__":

    yaml_input = """
        doc:
            description: The task is to provide a 1x2 MMI (Multi-Mode Interference) component
                with balanced optical outputs.
            labels:
            - ''
            reference: (link)
            title: 1x2 MMI with Balanced Outputs
        edges: ''
        nodes:
            N1:
                FoM:
                    direction: minimize
                    equation: abs(S21 - S31)
                component: _mmi1x2
                opt_settings:
                    width_mmi:
                        max: 4.0
                        min: 3.5
                    length_mmi: 
                        max: 10
                        min: 5
                params:
                    gap_mmi: 0.25
                    length_mmi: 12.8
                    length_taper: 10
                    width_mmi: 3.8
                    width_taper: 1.4
                properties:
                    ports: 1x2
        properties: {}
        """
    from ast import literal_eval

    def _mmi1x2(
        length_mmi: float = 12.8,
        width_mmi: float = 3.8,
        gap_mmi: float = 0.25,
        length_taper: float = 10.0,
        width_taper: float = 1.4,
    ) -> gf.Component:
        _args = locals()

        c = gf.Component()
        m = gf.components.mmi1x2(**_args)
        coupler_r = c << m
        c.add_port("o1", port=coupler_r.ports["o1"])
        c.add_port("o2", port=coupler_r.ports["o2"])
        c.add_port("o3", port=coupler_r.ports["o3"])
        c.flatten()
        return c

    print(_mmi1x2)
    res, sparams = run_gpoptimizer(_mmi1x2, yaml_input)   