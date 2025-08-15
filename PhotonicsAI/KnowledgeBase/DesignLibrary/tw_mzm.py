"""This is a Mach-Zehnder modulator that uses a pn-diode phase shifter and traveling-wave electrode for high-frequency operation.

---
Name: Traveling-wave Mach Zehnder Modulator (TW-MZM)
Description: >
    This is a Mach-Zehnder modulator that uses a pn-diode phase shifter
    and traveling-wave electrode for high-frequency operation.
ports: 1x1
NodeLabels:
    - modulator
    - active
    - phase modulation (AM)
aka: amplitude modulator, PN Junction
Technology : Plasma Dispersion Effect (PD)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 20 GHz
Insertion loss: 6.5 dB
Extinction ratio: 4.4 dB
Drive voltage/power: 2 mW
Footprint Estimate: 532.4um x 468.5um
Args:
    -length: straight length heater (um)
    -delta_length: path length difference (um). bottom arm vertical extra length.
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import (
    _mmi1x2,
    pndiode,
    straight,
)


@gf.cell
def tw_mzm(
    dy: float = 80,
    coupling1: float = 0.5,
    coupling2: float = 0.5,
    delta_length: float = 50,
    length: float = 320,
) -> gf.Component:
    """The component."""
    c = gf.Component()

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")

    mmi1x2 = _mmi1x2._mmi1x2()

    m2_xs = gf.cross_section.cross_section(width=68.5, offset=0, layer=(45, 0))
    _m2_pad = c << gf.path.straight(length=75).dmovex((length - 75) / 2 + 91.2).extrude(
        m2_xs
    )

    m2_xs = gf.cross_section.cross_section(width=420, offset=0, layer=(45, 0))
    _m2_straight = c << gf.path.straight(length=20).dmovex(
        (length - 75) / 2 + 62.2
    ).extrude(m2_xs)

    m2_xs = gf.cross_section.cross_section(width=20, offset=0, layer=(45, 0))
    _m2_top = c << gf.path.straight(length=30).dmovex((length - 75) / 2 + 62.2).dmovey(
        200
    ).extrude(m2_xs)
    _m2_mid = c << gf.path.straight(length=30).dmovex((length - 75) / 2 + 62.2).extrude(
        m2_xs
    )
    _m2_bot = c << gf.path.straight(length=30).dmovex((length - 75) / 2 + 62.2).dmovey(
        -200
    ).extrude(m2_xs)

    m2_xs = gf.cross_section.cross_section(width=68.5, offset=0, layer=(45, 0))
    _m2_pad = c << gf.path.straight(length=75).dmovex((length - 75) / 2 + 91.2).extrude(
        m2_xs
    )

    pad_xs = gf.cross_section.cross_section(width=64.5, offset=0, layer=(46, 0))
    _pad = c << gf.path.straight(length=71).dmovex((length - 71) / 2 + 91.2).extrude(
        pad_xs
    )

    ref = c << gf.components.mzi(
        delta_length=0,
        length_y=129.215,
        length_x=length,
        straight_x_top=pndiode.pndiode,
        straight_x_bot=pndiode.pndiode,
        splitter=mmi1x2,
        combiner=mmi1x2,
        cross_section=xs_1550,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.flatten()

    return c


def get_model(model="fdtd"):
    """The model."""
    return {"tw_mzm": straight.get_model_fdtd}


if __name__ == "__main__":
    c = tw_mzm()
    c.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
