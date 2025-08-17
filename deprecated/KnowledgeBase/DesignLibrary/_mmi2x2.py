"""
Name: mmi2x2
Description: A multimode interferometer with two input and two output ports.
ports: 2x2
NodeLabels:
    - passive
    - 2x2
Bandwidth: 50 nm
Args:
    -width: input and output straight width. Defaults to cross_section width.
    -width_taper: interface between input straights and mmi region.
    -length_taper: into the mmi region.
    -length_mmi: in x direction.
    -width_mmi: in y direction.
    -gap_mmi: gap between tapered wg.
"""

import gdsfactory as gf
import numpy as np
import sax

# from PhotonicsAI.Photon.utils import validate_cell_settings
from gdsfactory.typings import CrossSectionSpec

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz

# args = {
#     'functional': {
#         'wl0': {'default': 1.55, 'range': (1.0, 2.0)},
#         'coupling': {'default': 0.5, 'range': (0, 1)}
#     },
#     'geometrical': {
#         'length_taper': {'default': 10., 'range': (5.0, 15.0)},
#         'length_mmi':   {'default': 5.5, 'range': (5.0, 50.0)},
#         'width_mmi':    {'default': 2.5, 'range': (2.0, 6.0)},
#         'gap_mmi':      {'default': 0.25, 'range': (0.2, 0.3)},
#     }
# }


@gf.cell
def _mmi2x2(
    width: float | None = None,
    width_taper: float = 1.0,
    length_taper: float = 10.0,
    length_mmi: float = 5.5,
    width_mmi: float = 2.5,
    gap_mmi: float = 0.25,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    _args = locals()

    c = gf.Component()
    m = gf.components.mmi2x2(**_args)
    coupler_r = c << m

    c.add_port("o1", port=coupler_r.ports["o1"])
    c.add_port("o2", port=coupler_r.ports["o2"])
    c.add_port("o3", port=coupler_r.ports["o3"])
    c.add_port("o4", port=coupler_r.ports["o4"])

    c.flatten()
    return c


# def get_params(settings={}):
#     """
#     Generates the output configuration based on the settings.

#     Parameters:
#     settings (dict): A dictionary containing settings.

#     Returns:
#     dict: A dictionary containing the mapped geometrical parameters and direct output parameters.
#     """

#     validated_settings = validate_cell_settings(settings, args)

#     def wl_mapper(wl):
#         length_mmi = 20*wl + 2
#         return length_mmi

#     def coupling_mapper(coupling):
#         width_mmi = 2 + 2 * coupling
#         return width_mmi

#     output_params = {}

#     # handle all functional parameters first
#     # output_params['length_mmi'] = wl_mapper(validated_settings['functional']['wl0'])
#     # output_params['width_mmi'] = coupling_mapper(validated_settings['functional']['coupling'])

#     # Add remaining geometrical parameters
#     for arg in validated_settings['geometrical']:
#         if arg not in output_params:
#             output_params[arg] = validated_settings['geometrical'][arg]

#     return output_params


def get_model(model="fdtd"):
    if model == "ana":
        return {"_mmi2x2": get_model_ana}
    if model == "fdtd":
        return {"_mmi2x2": get_model_fdtd}


def get_model_fdtd(wl=1.55):
    file_path = get_file_path(
        "FDTD/cband/mmi2x2/mmi2x2_taper1p3_length36p2_width5p52_gap0p27.npz"
    )
    model_data = model_from_npz(file_path)
    return model_data(wl=wl)


def get_model_ana(wl=1.5, length_mmi=10):
    """a simple coupler model"""
    # wg_factor = np.exp(1j * np.pi * 2.34 * 1 / wl)
    wg_factor = 1
    coupling = length_mmi / 100
    kappa = wg_factor * coupling**0.5
    tau = wg_factor * (1 - coupling) ** 0.5
    sdict = sax.reciprocal(
        {
            ("o1", "o3"): tau,
            ("o1", "o4"): 1j * kappa,
            ("o2", "o3"): 1j * kappa,
            ("o2", "o4"): tau,
        }
    )
    return sdict


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("macosx")

    c = gf.Component()
    ref = c << _mmi2x2(length_mmi=100)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])
    c.add_port("o4", port=ref.ports["o4"])

    print(c.get_netlist())
    print()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model())
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o4"]) ** 2)

    plt.figure()
    wl = np.linspace(1.4, 1.6, 128)
    S31 = _c(wl=wl)["o1", "o3"]
    S41 = _c(wl=wl)["o1", "o4"]
    plt.plot(wl, np.abs(S31) ** 2)
    plt.plot(wl, np.abs(S41) ** 2)

    # c.plot()
    plt.show()
