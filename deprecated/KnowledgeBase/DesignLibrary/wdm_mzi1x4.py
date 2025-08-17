"""
Name: Cascaded Mach-Zehnder WDM Filter
paper: https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-10-11652&id=253305
Description: >
    A 1-to-4 wavelength (de-)multiplexer designed using a binary tree of cascaded Mach-Zehnder-like
    lattice filters. This device is engineered to function as a coarse wavelength division multiplexing (cWDM)
    filter for optical data communication.
ports: 1x4
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
Inputs: 1
Outputs: 4
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import (
    mzi_1x2_pindiode_cband,
    mzi_2x2_heater_tin_cband,
)


@gf.cell
def wdm_mzi1x4(dy: float = 120) -> gf.Component:
    dl = [100.965, 100.965 * 1.5, 100.965 * 1.25]

    c = gf.Component()

    c1 = c << mzi_1x2_pindiode_cband.mzi_1x2_pindiode_cband(length=dl[0])
    c2 = c << mzi_2x2_heater_tin_cband.mzi_2x2_heater_tin_cband(length=dl[1])
    c2.dmove((50 + c1.dxsize, 0 + c1.dysize / 2))
    c3 = c << mzi_2x2_heater_tin_cband.mzi_2x2_heater_tin_cband(length=dl[2])
    c3.dmove((50 + c1.dxsize, 0 - c1.dysize / 2))

    route = gf.routing.route_single(c, port1=c2.ports["o1"], port2=c1.ports["o2"])
    route = gf.routing.route_single(c, port1=c3.ports["o1"], port2=c1.ports["o3"])
    # c2.connect("o1", c1.ports["o2"])
    # c3.connect("o2", c1.ports["o3"])

    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o3"])
    c.add_port("o3", port=c2.ports["o4"])
    c.add_port("o4", port=c3.ports["o3"])
    c.add_port("o5", port=c3.ports["o4"])
    return c


def get_model(model="ana"):
    m1 = mzi_1x2_pindiode_cband.get_model()
    m2 = mzi_2x2_heater_tin_cband.get_model()
    combined_dict = m1 | m2
    return combined_dict


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import sax

    matplotlib.use("TkAgg")

    c = gf.Component()
    ref1 = c << wdm_mzi1x4()
    c.add_port("o1", port=ref1.ports["o1"])
    c.add_port("o2", port=ref1.ports["o2"])
    c.add_port("o3", port=ref1.ports["o3"])
    c.add_port("o4", port=ref1.ports["o4"])
    c.add_port("o5", port=ref1.ports["o5"])
    # ref1.dmirror()
    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    c.plot()
    plt.show()
    # print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
