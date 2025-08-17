"""
Name: Cascaded Mach-Zehnder WDM Filter
paper: https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-10-11652&id=253305
Description: >
    A 1-to-8 wavelength (de-)multiplexer designed using a binary tree of cascaded Mach-Zehnder-like
    lattice filters. This device is engineered to function as a wavelength division multiplexing (WDM)
    filter for optical data communication.
ports: 1x8
NodeLabels:
    - WDM
    - WDM
    - active
aka: wavelength filter
Technology: MZI
N of channels: 8
channel spacing: 3.2 nm (400 GHz)
Center Wavelength: 1490 nm
Design wavelength: 1280, 1340 nm
Pass-Band Flatness: Flat within 0.7 dB over 2.4 nm
Polarization: TE
Insertion loss: 1.6 dB
Extinction ratio: 18 dB
Inputs: 1
Outputs: 8
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import _directional_coupler, mzi_arm


@gf.cell
def wdm_mzi1x8(settings: dict = {}):
    dy = 80
    coupling1 = 0.5
    coupling2 = 0.5
    delta_length = 50

    c = gf.Component()

    c1 = c << _directional_coupler._directional_coupler(
        {"dy": dy, "coupling": coupling1}
    )
    a1 = c << mzi_arm.mzi_arm({"length": delta_length + 1})
    a2 = c << mzi_arm.mzi_arm({"length": 1})
    a2.drotate(180)
    c2 = c << _directional_coupler._directional_coupler(
        config={"dy": dy, "coupling": coupling2}
    )

    a1.connect("o1", c1.ports["o3"])
    a2.connect("o2", c1.ports["o4"])
    c2.connect("o1", a2.ports["o1"])
    c2.connect("o2", a1.ports["o2"])

    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c1.ports["o2"])
    c.add_port("o3", port=c2.ports["o3"])
    c.add_port("o4", port=c2.ports["o4"])
    return c


def get_model(model="ana"):
    m1 = _directional_coupler.get_model()
    m2 = mzi_arm.get_model()
    combined_dict = m1 | m2
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = wdm_mzi1x8()
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
