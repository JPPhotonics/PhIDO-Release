"""This is a bend (or generally arc shaped) waveguide.

---
Name: bezier_curve
Description: |
    This is a bend (or generally arc shaped) waveguide.
    The function maps a radii and and an angle to a 4 control points that implements the Bezier Curve.
ports: 1x1
NodeLabels:
    - passive
    - 1x1
Bandwidth:
"""

import math

import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz, model_from_tidy3d


@gf.cell
def _bezier_curve(radius: float = 10, angle: float = 90) -> gf.Component:
    """This is a bezier curve.

    https://stackoverflow.com/questions/30277646/svg-convert-arcs-to-cubic-bezier
    """
    c = gf.Component()
    rad_angle = ((angle - 180) * math.pi) / 180

    x1 = radius
    y1 = 0

    x2 = radius
    y2 = radius * (4 / 3) * math.tan(rad_angle / 4)

    x3 = radius * (
        math.cos(rad_angle) + (4 / 3 * math.tan(rad_angle / 4) * math.sin(rad_angle))
    )
    y3 = radius * (
        math.sin(rad_angle) - (4 / 3 * math.tan(rad_angle / 4) * math.cos(rad_angle))
    )

    x4 = radius * math.cos(rad_angle)
    y4 = radius * math.sin(rad_angle)

    ref = c << gf.components.bezier(
        control_points=((x1, y1), (x2, y2), (x3, y3), (x4, y4)),
        npoints=201,
        with_manhattan_facing_angles=True,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.flatten()
    return c


def get_model(model="tidy3d"):
    if model == "ana":
        return {"_bezier_curve": get_model_ana}
    if model == "fdtd":
        return {"_bezier_curve": get_model_fdtd}
    if model == "tidy3d":
        return {"_bezier_curve": get_model_tidy3d}


def get_model_tidy3d(wl=1.55):
    try:
        with open('build/modified_netlist.yml', 'r') as file:
            modified_netlist = yaml.safe_load(file)
        if "_bezier_curve" in modified_netlist.split():
            c = gf.read.from_yaml(modified_netlist)
        else:
            c = _bezier_curve()
    except:
        c = _bezier_curve()
    model_data = model_from_tidy3d(c=c)
    return model_data(wl=wl)


def get_model_fdtd(wl=1.55):
    file_path = get_file_path("FDTD/cband/straight/straight_length10um_width500nm.npz")
    model_data = model_from_npz(file_path)
    return model_data(wl=wl)


def get_model_ana(wl=1.55):
    neff = 2.34
    ng = 3.4

    def os(x):
        return 0.01 * np.cos(24 * np.pi * x) + 0.01

    # loss=0.0
    loss = os(wl)

    wl0 = 1.55
    radius = 10
    angle = 90

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


# class bezier_curve:

#     def __init__(self, config=None):
#         default_config = {'radius': 10, 'angle': 90}
#         if config is None:
#             config = default_config
#         else:
#             config = {**default_config, **config}

#         self.config = config
#         self.component = None
#         self.model = None

#         _ = self.config_to_geometry()
#         self.component = self.get_component()
#         self.model = {'bezier_curve': self.get_model_ana}

#     def config_to_geometry(self):
#         # self.wl0 = self.config['wl0']
#         # self.pol = self.config['pol']
#         self.radius = self.config['radius']
#         self.angle = self.config['angle']
#         return None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = _bezier_curve({"radius": 10, "angle": 90})

    print(c.get_netlist())
    print()
    # sys.exit()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model())
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o2"]) ** 2)

    c.plot()
    plt.show()
