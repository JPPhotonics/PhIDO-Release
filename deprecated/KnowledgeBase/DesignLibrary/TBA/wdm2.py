"""
Name: wdm2
Description: >
    This is a four channel WDM (wavelength division multiplexer)
    for the following wavelengths: 1308.5 nm , 1309.5 nm, 1310.5 nm, 1311.5 nm.
    It is based on a tree of MZIs.
ports: 2x4
NodeLabels:
    - active
    - 2x4
Bandwidth: 50 nm
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import mzi1


@gf.cell
def wdm2(settings: dict = {}):
    dy = 120
    dl = [100.055, 100.055 * 1.5, 100.055 * 1.25]

    c = gf.Component()

    c1 = c << mzi1.mzi1({"delta_length": dl[0], "dy": dy})
    c2 = c << mzi1.mzi1({"delta_length": dl[1], "dy": dy})
    c3 = c << mzi1.mzi1({"delta_length": dl[2], "dy": dy})

    c2.connect("o1", c1.ports["o3"])
    c3.connect("o2", c1.ports["o4"])

    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c1.ports["o2"])
    c.add_port("o3", port=c2.ports["o3"])
    c.add_port("o4", port=c2.ports["o4"])
    c.add_port("o5", port=c3.ports["o3"])
    c.add_port("o6", port=c3.ports["o4"])
    return c


def get_model(model="ana"):
    m1 = mzi1.get_model()
    return m1


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = wdm2()
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
