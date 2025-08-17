"""
Name: MZI 1x1 - Thermo-optic
Description: >
    This is a 1x1 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are TiN Heaters.
ports: 1x1
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
aka: amplitude modulator, Mach-Zehnder interferometer, MZI
Technology : Thermo-optic effect (TO)
Design wavelength: 1260-1360 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 200 KHz
Insertion loss: 2 dB
Extinction ratio: 25 dB
Drive voltage/power: 0.75 V
Footprint Estimate: 470.02um x 115.3um
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import heater_tin_oband


@gf.cell
def mzi_1x1_heater_tin_oband(settings: dict = {}):
    length = 320
    c = gf.Component()
    tin = gf.Component()

    mmi1x2 = gf.components.mmi1x2(
        width_mmi=3.6,
        length_mmi=13,
        gap_mmi=0.6,
        width_taper=1.2,
        length_taper=15,
        width=0.41,
    )
    tin = heater_tin_oband.heater_tin_oband({"length": length})

    xs_1310 = gf.cross_section.cross_section(width=0.41, offset=0, layer="WG")
    ref = c << gf.components.mzi(
        delta_length=0,
        length_y=2.5,
        length_x=length,
        straight_x_top=tin,
        straight_x_bot=tin.mirror((1, 0)),
        splitter=mmi1x2,
        combiner=mmi1x2,
        cross_section=xs_1310,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    return c


def get_model(model="ana"):
    combined_dict = 0
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = mzi_1x1_heater_tin_oband({"length": 100.4})
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
