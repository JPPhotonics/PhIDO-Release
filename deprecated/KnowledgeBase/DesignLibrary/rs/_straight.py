"""
Name: straight waveguide
Description: This is a straight single-mode waveguide aka photonic wire.
ports: 1x1
NodeLabels:
    - passive
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
#         'length':   {'default': 10., 'range': (0.1, 20000.0)},
#     }
# }


@gf.cell
def _straight(
    length: float = 10.0,
) -> gf.Component:
    _args = locals()

    c = gf.Component()
    ref = c << gf.components.straight(**_args)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    return c


def get_model(model="ana"):
    if model == "ana":
        return {"straight": get_model_ana}
    if model == "fdtd":
        return {"straight": get_model_fdtd_test}


def get_model_fdtd_test(wl=1.55, length=10):
    model_data = gsax.read.model_from_npz(
        "../FDTD/c_band/straight/straight_length10um_width500nm.npz"
    )

    # with open('../FDTD/straight/straight_20240709.pkl', 'rb') as f:
    # loaded_data = pickle.load(f)

    return model_data(wl=wl)


def get_model_ana(wl=1.55, length=10):
    loss = 0.001
    neff = 2.34
    ng = 3.4
    wl0 = 1.55
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
    print(get_model_fdtd_test(wl=1.31, length=100))
    print()
    print(get_model_ana(wl=1.55, length=10))

    # a = np.load('../FDTD/straight/straight_strip_length10um_width410nm.npz')
    # print(a.shape)
    # print()
    # b = np.load('../FDTD-test/straight_waveguide/straight_fba69bc3_f9e2d120.npz')
    # print(b.shape)

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import sys
#     from pprint import pprint

#     c = _straight(length=100)

#     pprint(c.get_netlist())
#     print()
#     # sys.exit()

#     recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
#     print('Required Models ==>', sax.get_required_circuit_models(recnet))

#     _c, info = sax.circuit(recnet, get_model())
#     print( _c(wl = 1.55) )
#     print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )

#     c.plot()
#     plt.show()
