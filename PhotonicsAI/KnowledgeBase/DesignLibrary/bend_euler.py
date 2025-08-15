"""This is a bend (or generally arc shaped) waveguide, with an Euler curvature.

---
Name: bend_euler
Description: This is a bend (or generally arc shaped) waveguide, with an Euler curvature.
ports: 1x1
NodeLabels:
    - passive
    - 1x1
Bandwidth: 100 nm
Args:
    -radius: in um. Defaults to cross_section_radius.
    -angle: total angle of the curve.
    -npoints: Number of points used per 360 degrees.
"""

import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz

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
def bend_euler(
    radius: float = 10.0,
    angle: float = 90.0,
    p: float = 0.5,
    npoints: int = 500,
    width: float = 0.5,
    cross_section: gf.typings.CrossSectionSpec = "strip",
) -> gf.Component:
    """This is a bend (or generally arc shaped) waveguide, with an Euler curvature."""
    # geometrical_params = get_params(settings)
    _args = locals()

    c = gf.Component()
    ref = c << gf.components.bend_euler(**_args)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    # c = c.flatten() # this gives an error!
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


def get_model(model="fdtd"):
    """This is a model."""
    if model == "ana":
        return {"bend_euler": get_model_ana}
    if model == "fdtd":
        return {"bend_euler": get_model_fdtd}


def get_model_fdtd(wl=1.55, radius=10, angle=90):
    """The FDTD model."""
    file_path = get_file_path(
        "FDTD/cband/bend_euler/bend_euler_npoints500_radius10.npz"
    )
    model_data = model_from_npz(file_path)
    return model_data(wl=wl)


def get_model_ana(wl=1.55, radius=10, angle=90):
    """The analytical model."""
    neff = 2.34
    ng = 3.4

    # radius = config['radius']
    # angle = config['angle']
    def os(x):
        return 0.01 * np.cos(24 * np.pi * x) + 0.01

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
    c = gf.Component()
    ref = c << bend_euler(radius=100)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    # pprint(c.get_netlist())
    # print()
    # sys.exit()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model(model="fdtd"))
    print(_c(wl=1.55))
    # print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )

    # c.plot()
    # plt.show()
