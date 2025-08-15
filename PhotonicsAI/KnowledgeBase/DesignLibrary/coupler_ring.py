"""This is a directional coupler with an Euler curvature. Typically used to design ring resonators.

---
Name: coupler_ring
Description: This is a directional coupler with an Euler curvature. Typically used to design ring resonators.
ports: 2x2
NodeLabels:
    - passive
    - 2x2
Bandwidth: 100 nm
Args:
    -gap: spacing between parallel coupled straight waveguides
    -radius: radius of the 90 degree bends
    -length_x: length of the parallel coupled straight waveguides 
"""

import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz
from PhotonicsAI.KnowledgeBase.DesignLibrary import bend_euler, straight, _directional_coupler

@gf.cell
def coupler_ring(
    gap: float = 0.2,
    radius: float = 10.0,
    length_x: float = 4,
    bend: gf.typings.ComponentSpec = "bend_euler",
    straight: gf.typings.ComponentSpec = "straight",
    cross_section: gf.typings.CrossSectionSpec = "strip",
) -> gf.Component:
    """This is a directional coupler with an Euler curvature. Typically used to design ring resonators"""
    # geometrical_params = get_params(settings)
    _args = locals()

    c = gf.Component()
    ref = c << gf.components.coupler_ring(**_args)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])
    c.add_port("o4", port=ref.ports["o4"])

    c.flatten() 
    return c


def get_model(model="fdtd"):
    """This is a model."""
    if model == "ana":
        return {"bend_euler": get_model_ana}
    if model == "fdtd":
        return {"coupler_ring": _directional_coupler.get_model_fdtd}


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
    ref = c << coupler_ring(radius=10)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])
    c.add_port("o4", port=ref.ports["o4"])

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
