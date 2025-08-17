"""This is a straight single-mode waveguide aka photonic wire.

---
Name: straight waveguide
Description: This is a straight single-mode waveguide aka photonic wire.
ports: 1x1
NodeLabels:
    - passive
Bandwidth: 100 nm
Args:
    -length: straight length (um)
"""

import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz, model_from_tidy3d

# import pickle

# from PhotonicsAI.Photon.utils import validate_cell_settings

# args = {
#     'functional': {
#     },
#     'geometrical': {
#         'length':   {'default': 10., 'range': (0.1, 20000.0)},
#     }
# }


@gf.cell
def straight(
    length: float = 10.01,
    npoints: int = 2,
    cross_section: gf.typings.CrossSectionSpec = "strip",
) -> gf.Component:
    """The component."""
    _args = locals()

    c = gf.Component()
    ref = c << gf.components.straight(**_args)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    c.flatten()
    return c


def get_model(model="tidy3d"):
    """The model."""
    if model == "ana":
        return {"straight": get_model_ana}
    if model == "fdtd":
        return {"straight": get_model_fdtd}
    if model == "tidy3d":
        return {"straight": get_model_tidy3d}


def get_model_tidy3d(wl=1.55):
    try:
        with open('build/modified_netlist.yml', 'r') as file:
            modified_netlist = yaml.safe_load(file)
        if "straight" in modified_netlist.split():
            c = gf.read.from_yaml(modified_netlist)
        else:
            c = straight()
    except:
        c = straight()
    model_data = model_from_tidy3d(c=c)
    return model_data(wl=wl)


def get_model_fdtd(wl=1.5, length=10.0, neff=3.2) -> sax.SDict:
    """Get FDTD model."""
    return sax.reciprocal({("o1", "o2"): np.exp(2j * np.pi * neff * length / wl)})


def get_model_ana(wl=1.55, length=10):
    """Get analytical model."""
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
    from pprint import pprint

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("macosx")

    c = gf.Component()
    ref = c << straight(length=100)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    pprint(c.get_netlist())
    print()
    # sys.exit()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model(model="fdtd"))
    print(_c(wl=1.55))
    # print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )

    plt.figure()
    wl = np.linspace(1.4, 1.6, 128)
    S21 = _c(wl=wl)["o1", "o2"]
    plt.plot(wl, np.abs(S21) ** 2)
    # gsax.plot_model(get_model_f/dtd)
    plt.show()

    # c.plot()
    # plt.show()
