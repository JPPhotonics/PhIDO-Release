"""
Name: PIN Diode
Description: >
    This is a pin diode.
    The device is operated in forward bias which causes carrier injectio in the waveguide core.
ports: 1x1
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
    - Variable Optical Attenuator (VOA)
aka: amplitude modulator, PIN diode, Variable Optical Attenuator (VOA)
Technology : Plasma Dispersion Effect
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 1 GHz
Insertion loss: 2-20 dB
Extinction ratio: N/A
Drive voltage/power: 2 mW
Footprint Estimate: 510.0um x 162.2um
Args:
    -length: straight length pin diode (um)
"""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from PhotonicsAI.KnowledgeBase.DesignLibrary import straight


@gf.cell
def pindiode_cband(
    length=500,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    c = gf.Component()

    wg_xs = gf.cross_section.cross_section(width=0.5, offset=0, layer=(1, 0))
    wg = c << gf.path.straight(length=length + 10).extrude(wg_xs)

    slab_xs = gf.cross_section.cross_section(width=25.1, offset=0, layer=(3, 0))
    slab = c << gf.path.straight(length=length).dmovex(5).extrude(slab_xs)

    slab_taper1 = c << gf.components.taper(
        length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
    )
    slab_taper1.dmovex(-5).drotate(180)

    slab_taper2 = c << gf.components.taper(
        length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
    )
    slab_taper2.dmovex(length + 5)

    npp_xs = gf.Section(width=11, offset=6.55, layer=(24, 0))
    ppp_xs = gf.Section(width=11, offset=-6.55, layer=(25, 0))
    doping_xs = gf.CrossSection(sections=[npp_xs, ppp_xs])
    doping = c << gf.path.straight(length=length - 2).dmovex(6).extrude(doping_xs)

    viac_top_xs = gf.Section(width=7.3, offset=-7.85, layer=(40, 0))
    viac_bottom_xs = gf.Section(width=7.3, offset=7.85, layer=(40, 0))
    viac_xs = gf.CrossSection(sections=[viac_top_xs, viac_bottom_xs])
    viac = c << gf.path.straight(length=length - 3).dmovex(6.5).extrude(viac_xs)

    m1_top_xs = gf.Section(width=17, offset=-12.2, layer=(41, 0))
    m1_bottom_xs = gf.Section(width=17, offset=12.2, layer=(41, 0))
    m1_xs = gf.CrossSection(sections=[m1_top_xs, m1_bottom_xs])
    m1 = c << gf.path.straight(length=length).dmovex(5).extrude(m1_xs)

    via1_top_xs = gf.Section(width=4, offset=16.7, layer=(44, 0))
    via1_bottom_xs = gf.Section(width=4, offset=-16.7, layer=(44, 0))
    via1_xs = gf.CrossSection(sections=[via1_top_xs, via1_bottom_xs])
    via1 = c << gf.path.straight(length=71).dmovex((length - 71) / 2).extrude(via1_xs)

    m2_top_xs = gf.Section(width=68.5, offset=46.85, layer=(45, 0))
    m2_bottom_xs = gf.Section(width=68.5, offset=-46.85, layer=(45, 0))
    m2_xs = gf.CrossSection(sections=[m2_top_xs, m2_bottom_xs])
    m2 = c << gf.path.straight(length=75).dmovex((length - 75) / 2).extrude(m2_xs)

    pad_top_xs = gf.Section(width=64.5, offset=46.85, layer=(46, 0))
    pad_bottom_xs = gf.Section(width=64.5, offset=-46.85, layer=(46, 0))
    pad_xs = gf.CrossSection(sections=[pad_top_xs, pad_bottom_xs])
    pad = c << gf.path.straight(length=71).dmovex((length - 71) / 2).extrude(pad_xs)

    c.add_port("o1", port=wg.ports["o1"])
    c.add_port("o2", port=wg.ports["o2"])
    c.flatten()
    return c


def get_model(model="fdtd"):
    return {"pindiode_cband": straight.get_model_fdtd}


if __name__ == "__main__":
    c = pindiode_cband()
    c.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
