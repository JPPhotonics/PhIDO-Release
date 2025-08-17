"""
Name: MZI 1x2 - Thermo-optic
Description: >
    This is a 1x2 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are TiN Heaters.
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
Footprint Estimate:  483.02um x 115.15um
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import heater_tin_cband


@gf.cell
def mzi_1x2_heater_tin_cband(settings: dict = {}):
    length = 320
    c = gf.Component()
    tin = gf.Component()

    mmi1x2 = gf.components.mmi1x2(
        width_mmi=3.8, length_mmi=12.8, gap_mmi=0.25, width_taper=1.4, length_taper=10
    )
    mmi2x2 = gf.components.mmi2x2(
        width_mmi=5.52, length_mmi=36.2, gap_mmi=0.27, width_taper=1.3, length_taper=15
    )
    tin = heater_tin_cband.heater_tin_cband({"length": length})

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")
    # ref = c << gf.components.mzi1x2_2x2(delta_length=0, length_y=2.5, length_x=length, straight_x_top=tin, straight_x_bot=tin.dmirror((1,0)), splitter=mmi1x2, combiner=mmi2x2, cross_section=xs_1550)
    ref = c << gf.components.mzi1x2_2x2(
        delta_length=0,
        length_y=2.5,
        length_x=length,
        splitter=mmi1x2,
        combiner=mmi2x2,
        cross_section=xs_1550,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])

    return c


def get_model(model="ana"):
    combined_dict = 0
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = mzi_1x2_heater_tin_cband({"length": 100.4})
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
