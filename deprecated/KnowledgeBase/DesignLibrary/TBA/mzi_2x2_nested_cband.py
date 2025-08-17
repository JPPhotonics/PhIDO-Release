"""
Name: MZI 2x2 - Thermo-optic
Paper: https://www.nature.com/articles/s41598-017-12455-8
Description: >
    This is a 2x2 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are MZI 2x2 which have PIN Phase Shifters on each of its arms.
ports: 2x2
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
aka: amplitude modulator, Mach-Zehnder interferometer, MZI
Technology : Thermo-optic effect (TO)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 200 KHz
Insertion loss: 2 dB
Extinction ratio: 25 dB
Drive voltage/power: 0.75 V
Footprint Estimate: 657.2um x 206.4um
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import dc_mzi_2x2_pindiode_cband


@gf.cell
def mzi_2x2_nested_cband(settings: dict = {}):
    length = 320

    c = gf.Component()

    dc1 = c << gf.components.coupler(gap=0.3, length=9.8, dy=55, dx=36)
    dc2 = c << gf.components.coupler(gap=0.3, length=9.8, dy=55, dx=36)
    mzi1 = c << dc_mzi_2x2_pindiode_cband.dc_mzi_2x2_pindiode_cband({"length": length})
    mzi2 = c << dc_mzi_2x2_pindiode_cband.dc_mzi_2x2_pindiode_cband({"length": length})

    dc1.connect("o4", mzi1.ports["o2"])
    dc2.connect("o1", mzi1.ports["o3"])

    mzi2.connect("o1", dc1.ports["o3"])
    mzi2.connect("o4", dc2.ports["o2"])

    c.add_port("o1", port=dc1.ports["o1"])
    c.add_port("o2", port=dc1.ports["o2"])
    c.add_port("o3", port=dc2.ports["o3"])
    c.add_port("o4", port=dc2.ports["o4"])

    return c


def get_model(model="ana"):
    combined_dict = 0
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = mzi_2x2_nested_cband({"length": 100.4})
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
