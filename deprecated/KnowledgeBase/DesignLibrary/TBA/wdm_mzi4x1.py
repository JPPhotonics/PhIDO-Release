"""
Name: Cascaded Mach-Zehnder WDM Filter
paper: https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-10-11652&id=253305
Description: >
    A 4-to-1 wavelength de-multiplexer designed using a binary tree of cascaded Mach-Zehnder-like
    lattice filters. This device is engineered to function as a coarse wavelength division demultiplexing (cWDM)
    for optical data communication.
ports: 4x1
NodeLabels:
    - WDM
    - CWDM
    - active
aka: wavelength filter
Technology: MZI
N of channels: 4
channel spacing: 20 nm
Design wavelength: 1271, 1291, 1311, 1331 nm
Optical Bandwidth: 2 nm
Polarization: TE
Insertion loss: 2.5 dB
Extinction ratio: 23 dB
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import (
    mzi_1x2_pindiode_cband,
    mzi_2x2_heater_tin_cband,
)


@gf.cell
def wdm_mzi4x1(dy: float = 120) -> gf.Component:
    dl = [100.965, 100.965 * 1.5, 100.965 * 1.25]

    c = gf.Component()

    c2 = c << mzi_2x2_heater_tin_cband.mzi_2x2_heater_tin_cband(length=dl[1])
    c2.dmove((0, 50 + c2.dysize / 2))

    c3 = c << mzi_2x2_heater_tin_cband.mzi_2x2_heater_tin_cband(length=dl[2])
    c3.dmove((0, -50 - c3.dysize / 2))

    c1 = c << mzi_1x2_pindiode_cband.mzi_1x2_pindiode_cband(length=dl[0])
    c1.dmirror()
    c1.dmove((2 * c2.dxsize, 0))

    route = gf.routing.route_single(c, port1=c2.ports["o4"], port2=c1.ports["o2"])
    route = gf.routing.route_single(c, port1=c3.ports["o3"], port2=c1.ports["o3"])
    # c2.connect("o1", c1.ports["o2"])
    # c3.connect("o2", c1.ports["o3"])

    c.add_port("o1", port=c3.ports["o1"])
    c.add_port("o2", port=c3.ports["o2"])
    c.add_port("o3", port=c2.ports["o1"])
    c.add_port("o4", port=c2.ports["o2"])
    c.add_port("o5", port=c1.ports["o1"])
    return c


def get_model(model="ana"):
    m1 = mzi_1x2_pindiode_cband.get_model()
    m2 = mzi_2x2_heater_tin_cband.get_model()
    combined_dict = m1 | m2
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = wdm_mzi4x1()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")

    c.plot()
    plt.show()
