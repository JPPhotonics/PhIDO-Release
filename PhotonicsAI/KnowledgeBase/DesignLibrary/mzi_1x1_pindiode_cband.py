"""This is a 1x1 Mach-Zehnder interferometer (MZI).

---
Name: MZI 1x1 - PIN Diode
Description: >
    This is a 1x1 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are PIN Diode.
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
Modulation bandwidth/Switching speed: 1 GHz
Insertion loss: 2-20 dB
Extinction ratio: 25 dB
Drive voltage/power: 2 mW
Footprint Estimate: 435.62um x 462.2um
Args:
    -length: straight length heater (um)
    -delta_length: path length difference (um). bottom arm vertical extra length.
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import (
    _mmi1x2,
    bend_euler,
    pindiode_cband,
    straight,
)


@gf.cell
def mzi_1x1_pindiode_cband(
    delta_length: float = 10, length: float = 320
) -> gf.Component:
    """The component."""
    c = gf.Component()

    mmi1x2 = _mmi1x2._mmi1x2()

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")
    ref = c << gf.components.mzi(
        delta_length=delta_length,
        length_y=129.175,
        length_x=length,
        straight_x_top=pindiode_cband.pindiode_cband,
        straight_x_bot=pindiode_cband.pindiode_cband,
        splitter=mmi1x2,
        combiner=mmi1x2,
        cross_section=xs_1550,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    return c


def get_model(model="fdtd"):
    """The model."""
    m1 = _mmi1x2.get_model(model=model)
    m2 = straight.get_model(model=model)
    m3 = bend_euler.get_model(model=model)
    combined_dict = m1 | m2 | m3
    return combined_dict


if __name__ == "__main__":
    c = mzi_1x1_pindiode_cband()
    c.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
