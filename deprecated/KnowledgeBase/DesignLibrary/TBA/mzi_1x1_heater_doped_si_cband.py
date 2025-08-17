"""
Name: MZI 1x1 - Thermo-optic
Description: >
    This is a 1x1 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are doped silicon waveguides.
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
Modulation bandwidth/Switching speed: 200 KHz
Insertion loss: 2 dB
Extinction ratio: 25 dB
Drive voltage/power: 0.75 V
Footprint Estimate: 435.62um x 90.05um
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import heater_doped_si_cband


@gf.cell
def mzi_1x1_heater_doped_si_cband(settings: dict = {}) -> gf.Component:
    length = 320
    c = gf.Component()

    mmi1x2 = gf.components.mmi1x2(
        width_mmi=3.8, length_mmi=12.8, gap_mmi=0.25, width_taper=1.4, length_taper=10
    )
    heater = heater_doped_si_cband.heater_doped_si_cband()

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")
    # TODO: the following line raises an error @Ankita
    ref = c << gf.components.mzi(
        delta_length=0,
        length_y=2.5,
        length_x=length,
        straight_x_top=heater,
        straight_x_bot=heater,
        splitter=mmi1x2,
        combiner=mmi1x2,
        cross_section=xs_1550,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    return c


def get_model(model="ana"):
    combined_dict = 0
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = mzi_1x1_heater_doped_si_cband()
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
