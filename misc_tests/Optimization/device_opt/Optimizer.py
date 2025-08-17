"""Created on August 20th 2024.

@author: Rishabh Iyer
"""

import math
import os
import pickle
import sys

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import yaml
from FDTD import fdtd
from gdsfactory.technology import LayerStack
from SimulationSettings import (
    SIMULATION_SETTINGS_LUMERICAL_FDTD_DEFAULT,
    SimulationSettingsLumericalFdtd,
)
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_gaussian_process

sys.path.append("C:\\Program Files\\Lumerical\\v232\\api\\python\\")
sys.path.append(os.path.dirname(__file__))  # Current directory
import lumapi

loc = "C:\\Users\\User\\OneDrive - University of Toronto\\Research\\PhotonicsAI\\KnowledgeBase\\DesignLibrary"
sys.path.append(loc)
from _var_bezier_curve import _var_bezier_curve


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
        simulation_settings: SimulationSettingsLumericalFdtd = SIMULATION_SETTINGS_LUMERICAL_FDTD_DEFAULT,
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

        with open(self.filepath + "\\res.PICKLE", "wb") as f:
            pickle.dump(self.res, f)

        self.opt_sparams = self.get_sparams(self.res)

        if self.graph_plot:
            self.get_graphs()

        return self.res, self.opt_sparams

    def fdtd_sparams(self, param_set):
        """Helper function for the optimizer. Runs FDTD simulations on the input parameter set and return the figure of merit.

        Args:
            param_set: optimizable parameter set
        """
        wg_width = self.const_params["wg_width"] or 0.41

        for key, param in zip(self.var_params.keys(), param_set):
            self.var_params[key] = param

        c = self.component(**(self.const_params | self.var_params))

        xs = gf.cross_section.cross_section(width=wg_width, offset=0, layer="WG")
        s = lumapi.FDTD()
        sparams = fdtd(
            session=s,
            component=c,
            cross_section=xs,
            save_filepath=self.filepath + "\\temporary_sparams",
            layer_stack=self.layer_stack,
            simulation_settings=self.ss,
        ).get_sparameters()

        opt = self.fom(sparams, self.fom_args)

        return opt

    def get_graphs(self):
        """Plots convergence graph for all cases. In case of a single optimizable parameter, plots baysian optimizer progress graphs."""
        if self.res.space.n_dims == 1:
            plt.figure()
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

            plt.figure()
            plot_convergence(self.res)
            plt.show(block=True)

        else:
            plt.figure()
            plot_convergence(self.res)
            plt.show(block=True)

    def get_sparams(self, res):
        """Returns the final Sparameters of the optimized geometry.

        Args:
            res: skopt optimized model

        Return:
            sparams: Sparameter dictionary

        """
        wg_width = self.const_params["wg_width"] or 0.5

        param_set = res["x"]

        for key, param in zip(self.var_params.keys(), param_set):
            self.var_params[key] = param

        c = self.component(**(self.const_params | self.var_params))

        xs = gf.cross_section.cross_section(width=wg_width, offset=0, layer="WG")

        s = lumapi.FDTD()
        sparams = fdtd(
            session=s,
            component=c,
            cross_section=xs,
            save_filepath=self.filepath + "\\optimized_sparams",
            layer_stack=self.layer_stack,
            simulation_settings=self.ss,
        ).get_sparameters()

        np.savez(self.filepath + "\\opt_sparams.npz", sparams)

        return sparams


if __name__ == "__main__":
    yaml_str = """
                Name: bezier_curve
                Description: |
                    This is a bend (or generally arc shaped) waveguide.
                    The function maps a radii and and an angle to a 4 control points that implements the Bezier Curve.
                ports: 1x1
                NodeLabels:
                    - passive
                    - 1x1
                Design wavelength: 1450-1650 nm
                Args:
                    cp1:
                        description:
                        optimizable: true
                        opt_range:
                            - 0
                            - 5.25
                    cp2:
                        description:
                        optimizable: true
                        opt_range:
                            - 0
                            - 5.25
                    cp3:
                        description:
                        optimizable: true
                        opt_range:
                            - 0
                            - 5.25
                    cp4:
                        description:
                        optimizable: true
                        opt_range:
                            - 0
                            - 5.25
                Specs:
                    radius: 5
                    angle: 90
                    wg_width: 0.5
                """

    config = yaml.safe_load(yaml_str)
    params_dict = config["Args"]
    specs_dict = config["Specs"]

    pbounds = {
        param: tuple(value["opt_range"])
        for param, value in params_dict.items()
        if value.get("optimizable")
    }
    optparams = {param: 0 for param in pbounds.keys()}

    filepath = "C:\\Users\\User\\OneDrive - University of Toronto\\Research\\Optimizer\\Results"

    def fom1(sparams, params):
        fom1 = np.sum(np.abs(sparams[params[0]]) ** 2) / len(sparams[params[0]])
        # fom2 = np.sum(np.abs(sparams[params[1]])**2)/len(sparams[params[1]])
        print(fom1)
        return fom1

    def fom2(sparams, params):
        fom = np.sum(np.abs(sparams[params[0]]) ** 2) / len(sparams[params[0]])

        print(fom)

        # Model Parameters
        p = 0.08612
        q = 0.9376
        r = 0.00109

        loss = p * math.exp(q * (-5)) + r

        print(loss)

        print(abs(loss - fom))

        return abs(loss - fom)

    ss = SimulationSettingsLumericalFdtd()
    ss.wavelength_start = float(config["Design wavelength"][0:3]) * 10e-3
    ss.wavelength_stop = float(config["Design wavelength"][5:8]) * 10e-3

    gp = GP_BO(
        _var_bezier_curve,
        specs_dict,
        optparams,
        pbounds,  # the bounds on each dimension of x
        fom1,
        ["S11"],
        acq_func="EI",  # the acquisition function
        n_calls=2,  # the number of evaluations of f
        n_random_starts=1,  # the number of random initialization points
        noise=0.1,  # the noise level (optional)
        filepath=filepath,
        graph_plot=True,
        simulation_settings=ss,
    )

    res, sparams = gp.run_opt()  # the random seed

    print(res)

    print(sparams)
