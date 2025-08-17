"""
Name: Titanium Nitride (TiN) Heater
Description: Thermo-optic phase shifter with a Titanium nitride heating element.
ports: 1x1
NodeLabels:
    - modulator
    - active
    - phase modulation (PM)
aka: phase modulator, thermo-optic phase shifter, heater
Technology : Thermo-optic effect (TO)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 30 KHz
Insertion loss: 1 dB
Extinction ratio: 25 dB
Drive voltage/power: 25 mW
Footprint Estimate: 397.0um x 130.0um
Args:
    -length: straight length heater (um)
"""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from PhotonicsAI.KnowledgeBase.DesignLibrary import straight


@gf.cell
def heater_tin_cband(
    length: float = 300,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    c = gf.Component()

    wg_xs = gf.cross_section.cross_section(width=0.5, offset=0, layer=(1, 0))
    wg = c << gf.path.straight(length=length + 24).extrude(wg_xs)
    wg.dmove((-12, 0))

    mh_xs = gf.cross_section.cross_section(width=3.5, offset=0, layer=(47, 0))
    mh = c << gf.path.straight(length=length).extrude(mh_xs)

    mhsq_xs = gf.cross_section.cross_section(width=12, offset=0, layer=(47, 0))

    mh1 = c << gf.path.straight(length=12).extrude(mhsq_xs)
    mh1.dmove((-12, 4.25))

    mh2 = c << gf.path.straight(length=12).extrude(mhsq_xs)
    mh2.dmove((length, 4.25))

    via1_xs = gf.cross_section.cross_section(width=5, offset=0, layer=(44, 0))

    via11 = c << gf.path.straight(length=5).extrude(via1_xs)
    via11.dmove((-8.5, 4.25))

    via12 = c << gf.path.straight(length=5).extrude(via1_xs)
    via12.dmove((length + 3.5, 4.25))

    m21_xs = gf.cross_section.cross_section(width=40, offset=0, layer=(45, 0))

    m21 = c << gf.path.straight(length=15).extrude(m21_xs)
    m21.dmove((-13.5, 14.25))

    m22 = c << gf.path.straight(length=15).extrude(m21_xs)
    m22.dmove((length - 1.5, 14.25))

    m22_xs = gf.cross_section.cross_section(width=100, offset=0, layer=(45, 0))

    m23 = c << gf.path.straight(length=100).extrude(m22_xs)
    m23.dmove((-48.5, 74.25))

    m24 = c << gf.path.straight(length=100).extrude(m22_xs)
    m24.dmove((length - 51.5, 74.25))

    pad_xs = gf.cross_section.cross_section(width=96, offset=0, layer=(46, 0))

    pad1 = c << gf.path.straight(length=96).extrude(pad_xs)
    pad1.dmove((-46.5, 74.25))

    pad2 = c << gf.path.straight(length=96).extrude(pad_xs)
    pad2.dmove((length - 49.5, 74.25))

    c.add_port("o1", port=wg.ports["o1"])
    c.add_port("o2", port=wg.ports["o2"])

    c.flatten()
    return c


def get_model(model="fdtd"):
    return {"heater_tin_cband": straight.get_model_fdtd}


# electrical model once we had eports to netlist
# def get_model(
# wl: float = 1.55,
# neff: float = 2.45,
# voltage: float = 0,
# length: float = 10,
# loss: float = 0.0,
# ) -> sax.SDict:
# """Returns simple phase shifter model"""
# deltaphi = voltage * jnp.pi
# phase = 2 * jnp.pi * neff * length / wl + deltaphi
# amplitude = jnp.asarray(10 ** (-loss * length / 20), dtype=complex)
# transmission = amplitude * jnp.exp(1j * phase)
# return sax.reciprocal(
# {
# ("o1", "o2"): transmission,
# ("l_e1", "r_e1"): 0.0,
# ("l_e2", "r_e2"): 0.0,
# ("l_e3", "r_e3"): 0.0,
# ("l_e4", "r_e4"): 0.0,
# }
# )

if __name__ == "__main__":
    c = heater_tin_cband()
    c.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
