import sys
from datetime import datetime

import gdsfactory as gf
import numpy as np
from gdsfactory.generic_tech import get_generic_pdk
from new_write_sparameters import write_sparameters_lumerical as WL

gf.config.rich_output()
PDK = get_generic_pdk()
PDK.activate()

sys.path.append("C:\\Program Files\\Lumerical\\v211\\api\\python\\")
# sys.path.append(os.path.dirname('__file__')) #Current directory
import lumapi

mySession = lumapi.FDTD(hide=True)


class MyLogger:
    def __init__(self, filepath, terminal=sys.stdout):
        self.terminal = terminal
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Flush after writing to ensure data is written to the file

    def flush(self):
        # This flush method is for compatibility and ensuring that flushing is done right.
        self.terminal.flush()
        self.log.flush()


def timestamp():
    now = datetime.now()
    date_time_string = now.strftime("%Y%m%d_%H%M%S")
    return date_time_string


def parse_opt_params(data):
    settings = data["yamlCode"]["settings"]

    opt_params = []
    x0 = []
    for key, value in settings.items():
        # running eval without any expression checks is a SAFETY concern!
        # using np.round because np.arange is generating numbers like 2.4000000000000004
        opt_params.append([key, list(np.round(eval(value.split("|")[0]), decimals=10))])
        x0.append(eval(value.split("|")[1]))
    return opt_params, x0


def objective_function(params, yamlDefinition):
    if not hasattr(objective_function, "counter"):
        objective_function.counter = 0  # initial value
    # `params` could be a list of parameters.
    # For example, if your model has two parameters, `params` could be [x1, x2].

    param_names = []
    for i in parse_opt_params(yamlDefinition)[0]:
        param_names.append(i[0])

    _component = (
        f"gf.components.{yamlDefinition['yamlCode']['component']}("
        + ", ".join(f"{name}={value}" for name, value in zip(param_names, params))
        + ")"
    )

    # sim_result = sim.write_sparameters_lumerical(
    sim_result = WL(
        # component = gf.components.mmi1x2(length_taper=params[0], length_mmi=params[1], width_mmi=params[2]),
        component=eval(_component),
        session=mySession,
        # run = True,
        # overwrite = False,
        dirpath="d:/vahid/nodes/_LumericalData/",
        # delete_fsp_files = True,
        count=objective_function.counter,
        # xmargin = 1,
        # ymargin = 1,
        # zmargin = 1,
        mesh_accuracy=1,  # (1: coarse, 2: fine, 3: superfine).
        wavelength_start=1.26,  # (um)
        wavelength_stop=1.36,  # (um)
        wavelength_points=50,
        # port_margin = 1.5, #on both sides of the port width (um).
        # port_extension = 5, #port extension (um).
    )

    objective_function.counter += 1
    # print(sim_result.keys())
    # pprint(sim_result)

    _cals = sim_result["S21"]
    _wl = sim_result["wavelengths"]

    _index = min(
        range(len(_wl)), key=lambda i: abs(_wl[i] - 1.31)
    )  # find the index of element closest to wavelength of 1.3 um
    len(_wl) // 2
    # output = np.abs(_cals[_index]**2) # --> all data earlier in the day
    # output = np.mean(np.abs(_cals**2)) # --> 20240410_160808
    # output = np.mean(np.abs(_cals[_index-5:_index+5]**2)) # --> 20240410_170626
    # output = np.mean(np.abs(_cals[_index-1:_index+1]**2)) # --> 20240410_175155
    # output = np.mean(np.abs(_cals**2)) + np.std(np.abs(_cals**2)) * 5 # --> 20240410_183333 [not good]
    # target_value = 0.5  # Example target value
    # objective = abs(output - target_value) # to be minimized

    output_vector = np.abs(_cals**2)  # -->
    target_vector = np.ones(len(output_vector)) * 0.5  # Example target value
    objective = np.linalg.norm(output_vector - target_vector)

    # objective = np.linalg.norm(output_vector[mid_index]-0.5)

    print("=====================")
    print("counter", objective_function.counter)
    print("params", params)
    # print('output', output)
    print("output-target", objective)
    # print('selected index, wl', _index, _wl[_index])
    print("=====================")
    # Assuming `target_value` is the output value you aim to achieve.

    # The objective is to minimize the absolute difference between the output and the target value.
    return objective, sim_result


# Wrapper around your objective function to store additional outputs
sim_results = []


def objective_wrapper(params, yamlDefinition):
    score, sim_res = objective_function(params, yamlDefinition)
    # sim_results.append((params, sim_res))
    sim_results.append(sim_res)
    return score  # Only the score is returned to gp_minimize
