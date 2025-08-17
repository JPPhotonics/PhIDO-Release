"""This is a 1x1 Mach-Zehnder interferometer (MZI).

---
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
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 200 KHz
Insertion loss: 2 dB
Extinction ratio: 25 dB
Drive voltage/power: 0.75 V
Footprint Estimate: 449.62um x 295.15um
Args:
    -length: straight length heater (um)
    -delta_length: path length difference (um). bottom arm vertical extra length.
"""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from PhotonicsAI.KnowledgeBase.DesignLibrary import (
    _mmi1x2,
    bend_euler,
    heater_tin_cband,
    straight,
)

# from PhotonicsAI.KnowledgeBase.DesignLibrary import straight


@gf.cell
def mzi_1x1_heater_tin_cband(
    delta_length: float = 10, length: float = 320
) -> gf.Component:
    """The component."""
    c = gf.Component()
    _tin = gf.Component()

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")

    mmi1x2 = _mmi1x2._mmi1x2()

    ref = c << gf.components.mzi(
        delta_length=delta_length,
        length_y=2.5,
        length_x=length,
        straight_x_top=heater_tin_cband.heater_tin_cband,
        straight_x_bot=heater_tin_cband_flipped,
        splitter=mmi1x2,
        combiner=mmi1x2,
        cross_section=xs_1550,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    return c


def get_model(model="tidy3d"):
    """Get the model for the edge coupler."""
    m1 = _mmi1x2.get_model(model=model)
    m2 = straight.get_model(model=model)
    m3 = bend_euler.get_model(model=model)
    combined_dict = m1 | m2 | m3
    return combined_dict


@gf.cell
def heater_tin_cband_flipped(
    length: float = 300,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    """Heater with TiN layer."""
    c = gf.Component()

    wg_xs = gf.cross_section.cross_section(width=0.5, offset=0, layer=(1, 0))
    wg = c << gf.path.straight(length=length + 24).extrude(wg_xs)
    wg.dmove((-12, 0))

    mh_xs = gf.cross_section.cross_section(width=3.5, offset=0, layer=(47, 0))
    mh = c << gf.path.straight(length=length).extrude(mh_xs)
    mh.dmirror_y()

    mhsq_xs = gf.cross_section.cross_section(width=12, offset=0, layer=(47, 0))

    mh1 = c << gf.path.straight(length=12).extrude(mhsq_xs)
    mh1.dmove((-12, 4.25))
    mh1.dmirror_y()

    mh2 = c << gf.path.straight(length=12).extrude(mhsq_xs)
    mh2.dmove((length, 4.25))
    mh2.dmirror_y()

    via1_xs = gf.cross_section.cross_section(width=5, offset=0, layer=(44, 0))

    via11 = c << gf.path.straight(length=5).extrude(via1_xs)
    via11.dmove((-8.5, 4.25))
    via11.dmirror_y()

    via12 = c << gf.path.straight(length=5).extrude(via1_xs)
    via12.dmove((length + 3.5, 4.25))
    via12.dmirror_y()

    m21_xs = gf.cross_section.cross_section(width=40, offset=0, layer=(45, 0))

    m21 = c << gf.path.straight(length=15).extrude(m21_xs)
    m21.dmove((-13.5, 14.25))
    m21.dmirror_y()

    m22 = c << gf.path.straight(length=15).extrude(m21_xs)
    m22.dmove((length - 1.5, 14.25))
    m22.dmirror_y()

    m22_xs = gf.cross_section.cross_section(width=100, offset=0, layer=(45, 0))

    m23 = c << gf.path.straight(length=100).extrude(m22_xs)
    m23.dmove((-48.5, 74.25))
    m23.dmirror_y()

    m24 = c << gf.path.straight(length=100).extrude(m22_xs)
    m24.dmove((length - 51.5, 74.25))
    m24.dmirror_y()

    pad_xs = gf.cross_section.cross_section(width=96, offset=0, layer=(46, 0))

    pad1 = c << gf.path.straight(length=96).extrude(pad_xs)
    pad1.dmove((-46.5, 74.25))
    pad1.dmirror_y()

    pad2 = c << gf.path.straight(length=96).extrude(pad_xs)
    pad2.dmove((length - 49.5, 74.25))
    pad2.dmirror_y()

    c.add_port("o1", port=wg.ports["o1"])
    c.add_port("o2", port=wg.ports["o2"])

    return c


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = mzi_1x1_heater_tin_cband()
    # c.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")

    c.plot()
    plt.show()
