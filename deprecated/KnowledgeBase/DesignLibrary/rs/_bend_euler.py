"""
Name: bend_euler
Description: This is a bend (or generally arc shaped) waveguide, with an Euler curvature.
ports: 1x1
NodeLabels:
    - passive
    - 1x1
Bandwidth: 100 nm
"""

import gdsfactory as gf
import gplugins.sax as gsax
import numpy as np
import sax

# from PhotonicsAI.Photon.utils import validate_cell_settings

# args = {
#     'functional': {
#     },
#     'geometrical': {
#         'radius':   {'default': 10., 'range': (2.0, 200.0)},
#         'angle':    {'default': 90, 'range': (90, 180)},
#         'p':        {'default': 0.5, 'range': (0, 1)},
#     }
# }


@gf.cell
def _bend_euler(
    radius: float = 10.0,
    npoints: int = 500,
) -> gf.Component:
    # geometrical_params = get_params(settings)
    _args = locals()

    c = gf.Component()
    ref = c << gf.components.bend_euler(**_args)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
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

#     output_params = {}

#     # Add remaining geometrical parameters
#     for arg in validated_settings['geometrical']:
#         if arg not in output_params:
#             output_params[arg] = validated_settings['geometrical'][arg]

#     return output_params


def get_model(model="ana"):
    if model == "ana":
        return {"bend_euler": get_model_ana}
    if model == "fdtd":
        return {"bend_euler": get_model_fdtd_test}


def get_model_fdtd_test(wl=1.55):
    model_data = gsax.read.model_from_npz(
        "../FDTD/c_band/bend_euler/bend_euler_npoints500_radius10.npz"
    )
    return model_data(wl=wl)


def get_model_ana(wl=1.55, radius=10, angle=90):
    neff = 2.34
    ng = 3.4
    # radius = config['radius']
    # angle = config['angle']
    os = lambda x: 0.01 * np.cos(24 * np.pi * x) + 0.01
    # loss=0.0
    loss = os(wl)

    wl0 = 1.55
    length = radius * angle * (np.pi / 180)

    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * np.pi * neff * length / wl
    transmission = 10 ** (-loss * length / 20) * np.exp(1j * phase)
    sdict = sax.reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )
    return sdict


if __name__ == "__main__":
    from pprint import pprint

    import matplotlib.pyplot as plt

    c = _bend_euler({"radius": 100})

    pprint(c.get_netlist())
    print()
    # sys.exit()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model())
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o2"]) ** 2)

    c.plot()
    plt.show()
