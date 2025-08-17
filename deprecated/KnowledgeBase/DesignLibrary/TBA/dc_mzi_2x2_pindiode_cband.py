"""
Name: MZI 2x2 - PIN Diode
Description: >
    This is a 2x2 Mach-Zehnder interferometer (MZI) with 50-50 directional couplers.
    Integrated in both arms of the MZI are PIN Diodes.
ports: 2x2
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
aka: amplitude modulator, Mach-Zehnder interferometer, MZI
Technology : Plasma-Dispersion Effect
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 1 GHz
Insertion loss: 2-20 dB
Extinction ratio: 25 dB
Drive voltage/power: 2 mW
Footprint Estimate: 493.6um x 96.4um
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import pindiode_cband


@gf.cell
def dc_mzi_2x2_pindiode_cband(settings: dict = {}):
    length = 320
    c = gf.Component()

    dc1 = c << gf.components.coupler(gap=0.3, length=9.8, dy=55, dx=36)
    dc2 = c << gf.components.coupler(gap=0.3, length=9.8, dy=55, dx=36)
    pin1 = c << pindiode_cband.pindiode_cband({"length": length})
    pin2 = c << pindiode_cband.pindiode_cband({"length": length})

    dc1.connect("o4", pin2.ports["o1"])
    dc2.connect("o1", pin2.ports["o2"])

    pin1.connect("o1", dc1.ports["o3"])
    pin1.connect("o2", dc2.ports["o2"])

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

    c = dc_mzi_2x2_pindiode_cband()
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
