"""
Name: Variable Optical Attenuator (VOA) - Thermo-optic
Description: >
    This is a 1x2 Mach-Zehnder interferometer (MZI) with one out of two output ports dumping power.
    Integrated in both arms of the MZI are TiN heaters.
ports: 1x1
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
aka: amplitude modulator, Mach-Zehnder interferometer, MZI
Technology : Thermo-optic effect (TO)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 30 KHz
Insertion loss: 1 dB
Extinction ratio: 25 dB
Drive voltage/power: 25 mW
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import _directional_coupler, mzi_arm


@gf.cell
def voa_to(settings: dict = {}):
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
        {"dy": dy, "coupling": coupling2}
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

    c = voa_to()
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
