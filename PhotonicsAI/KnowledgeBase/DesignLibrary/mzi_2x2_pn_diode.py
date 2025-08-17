"""This is a 2x2 Mach-Zehnder Interferometer (MZI) with a pn diode.

---
Name: MZI 2x2 PN Diode
Description: >
    This is a pn diode.
    The device is operated in reverse bias which causes carrier injectio in the waveguide core.
ports: 2x2
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
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
    _mmi2x2,
    bend_euler,
    pndiode,
    straight,
)


@gf.cell
def mzi_2x2_pn_diode(
    dy: float = 80,
    coupling1: float = 0.5,
    coupling2: float = 0.5,
    delta_length: float = 50,
    length: float = 320,
) -> gf.Component:
    """The component."""
    c = gf.Component()

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")

    mmi2x2 = _mmi2x2._mmi2x2()

  

    ref = c << gf.components.mzi2x2_2x2(
        delta_length=delta_length,
        length_y=129.215,
        length_x=length,
        straight_x_top=pndiode.pndiode,
        straight_x_bot=pndiode.pndiode,
        splitter=mmi2x2,
        combiner=mmi2x2,
        cross_section=xs_1550,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])
    c.add_port("o4", port=ref.ports["o4"])

    return c


def get_model(model="fdtd"):
    """The model."""
    m1 = _mmi2x2.get_model(model=model)
    m2 = straight.get_model(model=model)
    m3 = bend_euler.get_model(model=model)
    m4 = pndiode.get_model(model=model)
    combined_dict = m1 | m2 | m3 | m4
    return combined_dict


if __name__ == "__main__":
    c = mzi_2x2_pn_diode()
    c.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
