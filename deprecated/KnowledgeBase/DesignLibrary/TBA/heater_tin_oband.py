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
Design wavelength: 1260-1360 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 30 KHz
Insertion loss: 1 dB
Extinction ratio: 25 dB
Drive voltage/power: 25 mW
Footprint Estimate: 327.0um x 40.0um
"""

import gdsfactory as gf


@gf.cell
def heater_tin_oband(settings: dict = {}):
    length = 300
    c = gf.Component()

    wg_xs = gf.cross_section.cross_section(width=0.41, offset=0, layer=(1, 0))
    wg = c << gf.path.straight(length=length + 24).extrude(wg_xs)
    wg.move((-12, 0))

    mh_xs = gf.cross_section.cross_section(width=3.5, offset=0, layer=(47, 0))
    mh = c << gf.path.straight(length=length).extrude(mh_xs)

    mhsq_xs = gf.cross_section.cross_section(width=12, offset=0, layer=(47, 0))

    mh1 = c << gf.path.straight(length=12).extrude(mhsq_xs)
    mh1.move((-12, 4.25))

    mh2 = c << gf.path.straight(length=12).extrude(mhsq_xs)
    mh2.move((length, 4.25))

    via1_xs = gf.cross_section.cross_section(width=5, offset=0, layer=(44, 0))

    via11 = c << gf.path.straight(length=5).extrude(via1_xs)
    via11.move((-8.5, 4.25))

    via12 = c << gf.path.straight(length=5).extrude(via1_xs)
    via12.move((length + 3.5, 4.25))

    m2_xs = gf.cross_section.cross_section(width=40, offset=0, layer=(45, 0))

    m21 = c << gf.path.straight(length=15).extrude(m2_xs)
    m21.move((-13.5, 14.25))

    m22 = c << gf.path.straight(length=15).extrude(m2_xs)
    m22.move((length - 1.5, 14.25))

    c.add_port("o1", port=wg.ports["o1"])
    c.add_port("o2", port=wg.ports["o2"])

    return c


def get_model(model="ana"):
    combined_dict = 0
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = heater_tin_oband({"length": 100.4})
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
