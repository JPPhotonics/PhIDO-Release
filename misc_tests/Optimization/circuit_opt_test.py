"""Created on Sun Jul 21 22:23:26 2024.

@author: ansharma
"""

"""
Name: mzi_arm
Description: This is a Mach-Zehnder interferometer (MZI) arm.
ports: 1x1
NodeLabels:
    - passive
    - 1x1
Bandwidth: 100 nm
"""

import sys

import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np
import sax

sys.path.append(r"C:\Users\ansharma\Desktop")
import yaml
from bayes_opt import BayesianOptimization

from PhotonicsAI.KnowledgeBase.DesignLibrary import _mmi1x2


def mzi2(delta_length: float = 10, length: float = 10) -> gf.Component:
    c = gf.Component()
    # dc = _directional_coupler._directional_coupler()
    mmi = _mmi1x2._mmi1x2()
    ref = c << gf.components.mzi(delta_length, splitter=mmi, combiner=mmi)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    # c.flatten()
    # params = get_params(settings)
    return c


def straight(wl=1.5, length=10.0, neff=2.4) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): np.exp(2j * np.pi * neff * length / wl)})


def interp_model(filepath, xkey: str = "wavelengths", xunits: float = 1):
    sp = np.load(filepath)
    list(sp.keys())
    x = np.asarray(sp[xkey] * xunits)
    np.linspace(1.5, 1.6, 100)

    # make sure x is sorted from low to high
    idxs = np.argsort(x)
    x = x[idxs]
    sp = {k: v[idxs] for k, v in sp.items()}

    def model(wl=1.55):
        S = {}
        zero = np.zeros_like(x)

        for key in sp:
            if not key.startswith("wav"):
                port_mode0, port_mode1 = key.split(",")
                port0, _ = port_mode0.split("@")
                port1, _ = port_mode1.split("@")
                m = np.interp(wl, x, np.abs(sp.get(key, zero)))
                a = np.interp(wl, x, np.angle(sp.get(key, zero)))
                S[(port0, port1)] = m * np.exp(1j * a)
        return S

    return model


def get_model(model="fdtd"):
    m1 = interp_model(
        r"PhotonicsAI/KnowledgeBase/FDTD/cband/mmi1x2/mmi1x2_length12p8_width3p8_gap0p25_taperwidth1p4.npz"
    )
    m2 = straight
    m3 = interp_model(
        r"PhotonicsAI/KnowledgeBase/FDTD/cband/bend_euler/bend_euler_npoints500_radius10.npz"
    )
    interp_model(r"PhotonicsAI/KnowledgeBase/FDTD/cband/coupler/coupler_adiabatic.npz")
    interp_model(
        r"PhotonicsAI/KnowledgeBase/FDTD/cband/mmi2x2/mmi2x2_taper1p3_length36p2_width5p52_gap0p27.npz"
    )
    combined_dict = {
        "_mmi1x2": m1,
        "straight": m2,
        "bend_euler": m3,
    }
    return combined_dict


def target_transfer(wav, **kwargs):
    lambda_0 = kwargs.get("lambda_0")
    fsr = kwargs.get("FSR")
    arg = (2 * np.pi * lambda_0**2) / (wav * fsr)
    T = 0.5 * (1 + np.cos(arg))
    return T


# c = mzi2(delta_length=83.34)
# c.plot()

# print(c.get_netlist())
# recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
# # print('Required Models ==>', sax.get_required_circuit_models(recnet))
# _c, info = sax.circuit(recnet, models=get_model())
# wl = np.linspace(1.51, 1.59, 1000)
# S = _c(wl=wl)
# lambda_0 = 1.55
# fsr = 0.01
# T = target_transfer(wl, lambda_0, fsr)
# plt.plot(wl, abs(S["o1", "o2"]) ** 2)
# plt.plot(wl, T)

yaml_str = """
Name: MZI 2x2 - Thermo-optic
Description: >
    This is a 2x2 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are TiN Heaters.
ports: 2x2
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
aka: amplitude modulator, Mach-Zehnder interferometer, MZI
Technology: Thermo-optic effect (TO)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 200 KHz
Insertion loss: 2 dB
Extinction ratio: 25 dB
Drive voltage/power: 0.75 V
Footprint Estimate: 516.42um x 295.07um

Args:
  length:
    description: straight length heater (um)
  delta_length:
    description: path length difference (um). bottom arm vertical extra length.
    optimizable: true
    opt_range:
      - 0
      - 100

Specs:
  lambda_0: 1.55
  FSR: 0.040
  transmission_fn: target_transfer
"""
config = yaml.safe_load(yaml_str)
params_dict = config["Args"]
specs_dict = config["Specs"]

# Identify optimizable parameters and set up optimization bounds
pbounds = {
    param: tuple(value["opt_range"])
    for param, value in params_dict.items()
    if value.get("optimizable")
}
optparams = {param: 0 for param in pbounds.keys()}


def fom(specs_dict, optparams):
    # get SAX simulation
    c = mzi2(**optparams)
    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    _c, info = sax.circuit(recnet, models=get_model())
    wl = np.linspace(1.51, 1.59, 1000)
    S = _c(wl=wl)

    # Get the transmission function and pass the entire specs_dict
    transmission_fn_name = specs_dict.get("transmission_fn")
    transmission_fn = globals().get(transmission_fn_name)

    T = transmission_fn(wav=wl, **specs_dict)
    error = np.sum(np.abs(abs(S["o1", "o2"]) ** 2 - T) ** 2)
    return -error


def objective(**kwargs):
    for param in kwargs:
        optparams[param] = kwargs[param]
    return fom(specs_dict, optparams)


# print(f"Function 'target_transfer' in globals: {'target_transfer' in globals()}")
# print(f"target_transfer function object: {globals().get('target_transfer')}")

optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    verbose=2,
    random_state=1,
)

optimizer.maximize(
    init_points=5,
    n_iter=150,
)

best_delta_L = optimizer.max["params"]["delta_length"]
c = mzi2(delta_length=best_delta_L)
recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
_c, info = sax.circuit(recnet, models=get_model())
wl = np.linspace(1.51, 1.59, 1000)
S = _c(wl=wl)
lambda_0 = 1.55
fsr = 0.025
T = target_transfer(wl, **specs_dict)
plt.plot(wl, abs(S["o1", "o2"]) ** 2)
plt.plot(wl, T)
print(f"Optimal delta_L: {best_delta_L}")

# target_transfer(np.linspace(1.51, 1.59, 1000), **specs_dict)
