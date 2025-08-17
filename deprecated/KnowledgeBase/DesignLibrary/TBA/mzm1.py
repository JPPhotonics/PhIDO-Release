"""
Name: mzi1
Description: This is a Mach-Zehnder modulator (MZM), with one input and one output port.
ports: 1x1
NodeLabels:
    - active
    - 1x1
Bandwidth: 50 nm
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import _mmi1x2, mzi_arm


@gf.cell
def mzm1(settings: dict = {}):
    delta_length = 50
    c = gf.Component()

    c1 = c << _mmi1x2._mmi1x2()
    a1 = c << mzi_arm.mzi_arm({"length": delta_length + 1})
    a2 = c << mzi_arm.mzi_arm({"length": 1})
    a2.drotate(180)
    c2 = c << _mmi1x2._mmi1x2()

    a1.connect("o1", c1.ports["o2"])
    a2.connect("o2", c1.ports["o3"])
    c2.connect("o2", a2.ports["o1"])
    c2.connect("o3", a1.ports["o2"])

    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c2.ports["o1"])
    return c


def get_model(model="ana"):
    m1 = _mmi1x2.get_model()
    m2 = mzi_arm.get_model()
    combined_dict = m1 | m2
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = mzm1()
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
