"""
Name: bend_s
Description: This is an s-bend using bezier curves.
ports: 1x1
NodeLabels:
    - passive
    - 1x1
Bandwidth: 100 nm
Args:
    -size: in x (length) and y (height) direction.
"""

import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz


@gf.cell
def _bend_s(
    size: tuple[float, float] = (40.0, 26.0),
) -> gf.Component:
    _args = locals()

    c = gf.Component()
    ref = c << gf.components.bend_s(**_args)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.flatten()
    return c


def get_model(model="fdtd"):
    if model == "ana":
        return {"bend_s": get_model_ana}
    if model == "fdtd":
        return {"bend_s": get_model_fdtd}


def get_model_fdtd(wl=1.55):
    file_path = get_file_path("FDTD/cband/bend_s/bend_s_size40__26_npoints99.npz")
    model_data = model_from_npz(file_path)
    return model_data(wl=wl)


def get_model_ana(wl=1.55):
    # TODO: we need to find how long the curve is...
    # for now i approximate this from size-x
    wl0 = 1.55
    size_xy = (100, 4)

    length = size_xy[0]
    loss = 0.001
    neff = 2.34
    ng = 3.4
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


# class bend_s:

#     def __init__(self, config=None):
#         default_config = {'wl0': 1.55,
#                         #   'pol': 'TE',
#                           'size_xy':(10, 4)}
#         if config is None:
#             config = default_config
#         else:
#             config = {**default_config, **config}

#         self.config = config
#         self.component = None
#         self.model = None

#         _ = self.config_to_geometry()
#         self.component = self.get_component()
#         self.model = {'bezier': self.get_model_ana}

#     def config_to_geometry(self):
#         self.wl0 = self.config['wl0']
#         # self.pol = self.config['pol']
#         self.size_xy = self.config['size_xy']
#         return None


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = _bend_s()
    print(c.get_netlist())
    print()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model())
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o2"]) ** 2)

    c.plot()
    plt.show()
