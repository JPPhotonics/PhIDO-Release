"""This module defines a heater formed using doped silicon waveguides (n++, n).

---
Name: Doped silicon heater
Description: This is a heater formed using doped silicon waveguides (n++, n).
ports: 1x1
NodeLabels:
    - modulator
    - active
    - phase modulation (AM)
aka: phase modulator
Technology : Thermo-optic effect (TO)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 200 KHz
Insertion loss: 2 dB
Extinction ratio: N/A
Drive voltage/power: 0.75 V
Footprint Estimate: 110.4um x 164.2um
"""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from PhotonicsAI.KnowledgeBase.DesignLibrary import straight


@gf.cell
def heater_doped_si_cband(
    length: float = 100.4, cross_section: CrossSectionSpec = "strip"
) -> gf.Component:
    """The component."""
    c = gf.Component()

    wg_xs = gf.cross_section.cross_section(width=0.5, offset=0, layer=(1, 0))
    wg = c << gf.path.straight(length=length + 10).extrude(wg_xs)

    slab_xs = gf.cross_section.cross_section(width=24.5, offset=0, layer=(3, 0))
    _slab = c << gf.path.straight(length=length).dmovex(5).extrude(slab_xs)

    slab_taper1 = c << gf.components.taper(
        length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
    )
    slab_taper1.dmovex(-5).drotate(180)

    slab_taper2 = c << gf.components.taper(
        length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
    )
    slab_taper2.dmovex(length + 5)

    npp_top_xs = gf.Section(width=10.45, offset=-6.575, layer=(24, 0))
    npp_bottom_xs = gf.Section(width=10.45, offset=6.575, layer=(24, 0))
    n_xs = gf.Section(width=23.6, offset=0, layer=(20, 0))
    doping_xs = gf.CrossSection(sections=[npp_top_xs, n_xs, npp_bottom_xs])
    _doping = c << gf.path.straight(length=length - 0.4).dmovex(5.2).extrude(doping_xs)

    viac_top_xs = gf.Section(width=7.1, offset=-7.75, layer=(40, 0))
    viac_bottom_xs = gf.Section(width=7.1, offset=7.75, layer=(40, 0))
    viac_xs = gf.CrossSection(sections=[viac_top_xs, viac_bottom_xs])
    _viac = c << gf.path.straight(length=length - 1).dmovex(5.5).extrude(viac_xs)

    m1_top_xs = gf.Section(width=20.5, offset=-11.45, layer=(41, 0))
    m1_bottom_xs = gf.Section(width=20.5, offset=11.45, layer=(41, 0))
    m1_xs = gf.CrossSection(sections=[m1_top_xs, m1_bottom_xs])
    _m1 = c << gf.path.straight(length=length + 0.4).dmovex(4.8).extrude(m1_xs)

    via1_top_xs = gf.Section(width=4, offset=17.7, layer=(44, 0))
    via1_bottom_xs = gf.Section(width=4, offset=-17.7, layer=(44, 0))
    via1_xs = gf.CrossSection(sections=[via1_top_xs, via1_bottom_xs])
    _via1 = c << gf.path.straight(length=71).dmovex((length - 71) / 2 + 5).extrude(
        via1_xs
    )

    m2_top_xs = gf.Section(width=68.5, offset=47.85, layer=(45, 0))
    m2_bottom_xs = gf.Section(width=68.5, offset=-47.85, layer=(45, 0))
    m2_xs = gf.CrossSection(sections=[m2_top_xs, m2_bottom_xs])
    _m2 = c << gf.path.straight(length=75).dmovex((length - 75) / 2 + 5).extrude(m2_xs)

    pad_top_xs = gf.Section(width=64.5, offset=47.85, layer=(46, 0))
    pad_bottom_xs = gf.Section(width=64.5, offset=-47.85, layer=(46, 0))
    pad_xs = gf.CrossSection(sections=[pad_top_xs, pad_bottom_xs])
    _pad = c << gf.path.straight(length=71).dmovex((length - 71) / 2 + 5).extrude(
        pad_xs
    )

    c.add_port("o1", port=wg.ports["o1"])
    c.add_port("o2", port=wg.ports["o2"])

    c.flatten()
    return c


def get_model(model="fdtd"):
    """The model."""
    return {"heater_doped_si_cband": straight.get_model_fdtd}


if __name__ == "__main__":
    c = heater_doped_si_cband()
    c.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
