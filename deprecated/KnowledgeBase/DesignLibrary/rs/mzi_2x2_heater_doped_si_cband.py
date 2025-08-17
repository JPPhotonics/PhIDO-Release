"""
Name: MZI 2x2 - Thermo-optic
Description: >
    This is a 2x2 Mach-Zehnder interferometer (MZI) with 50-50 directional couplers.
    Integrated in both arms of the MZI are doped heaters.
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
Footprint Estimate: 435.6um x 464.2um
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import heater_doped_si_cband


@gf.cell
def mzi_2x2_heater_doped_si_cband(length: float = 320) -> gf.Component:
    c = gf.Component()

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")

    mmi1x2 = gf.components.mmi1x2(
        width_mmi=3.8, length_mmi=12.8, gap_mmi=0.25, width_taper=1.4, length_taper=10
    )

    ref = c << gf.components.mzi(
        delta_length=0,
        length_y=129.175,
        length_x=length,
        straight_x_top=heater_doped_si_cband.heater_doped_si_cband,
        straight_x_bot=heater_doped_si_cband.heater_doped_si_cband,
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
    c = mzi_2x2_heater_doped_si_cband()
    print(c.ports)
    c.show()
    print(
        "Footprint Estimate: "
        + str(round(c.xsize, 2))
        + "um x "
        + str(round(c.ysize, 2))
        + "um"
    )
