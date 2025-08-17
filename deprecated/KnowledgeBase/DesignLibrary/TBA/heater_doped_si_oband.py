"""
Name: Doped silicon heater
Description: This is a heater formed using doped silicon waveguides (n++, n).
ports: 1x1
NodeLabels:
    - modulator
    - active
    - phase modulation (AM)
aka: phase modulator
Technology : Thermo-optic effect (TO)
Design wavelength: 1260-1360 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 200 KHz
Insertion loss: 2 dB
Extinction ratio: N/A
Drive voltage/power: 0.75 V
Footprint Estimate: 110.4um x 43.4um
"""

import gdsfactory as gf


@gf.cell
def heater_doped_si_oband(settings: dict = {}) -> gf.Component:
    length = 100.4
    c = gf.Component()

    wg_xs = gf.cross_section.cross_section(width=0.41, offset=0, layer=(1, 0))
    wg = c << gf.path.straight(length=length + 10).extrude(wg_xs)

    slab_xs = gf.cross_section.cross_section(width=24.5, offset=0, layer=(3, 0))
    slab = c << gf.path.straight(length=length).movex(5).extrude(slab_xs)

    slab_taper1 = c << gf.components.taper(
        length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
    )
    slab_taper1.movex(-5).rotate(180)

    slab_taper2 = c << gf.components.taper(
        length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
    )
    slab_taper2.movex(length + 5)

    npp_top_xs = gf.Section(width=10.45, offset=-6.575, layer=(24, 0))
    npp_bottom_xs = gf.Section(width=10.45, offset=6.575, layer=(24, 0))
    n_xs = gf.Section(width=23.6, offset=0, layer=(20, 0))
    doping_xs = gf.CrossSection(sections=[npp_top_xs, n_xs, npp_bottom_xs])
    doping = c << gf.path.straight(length=length - 0.4).movex(5.2).extrude(doping_xs)

    metal_top_xs = gf.Section(width=20.5, offset=-11.45, layer=(41, 0))
    metal_bottom_xs = gf.Section(width=20.5, offset=11.45, layer=(41, 0))
    metal_xs = gf.CrossSection(sections=[metal_top_xs, metal_bottom_xs])
    metal = c << gf.path.straight(length=length + 0.4).movex(4.8).extrude(metal_xs)

    via_top_xs = gf.Section(width=7.1, offset=-7.75, layer=(40, 0))
    via_bottom_xs = gf.Section(width=7.1, offset=7.75, layer=(40, 0))
    metal_xs = gf.CrossSection(sections=[via_top_xs, via_bottom_xs])
    metal = c << gf.path.straight(length=length - 1).movex(5.5).extrude(metal_xs)

    c.add_port("o1", port=wg.ports["o1"])
    c.add_port("o2", port=wg.ports["o2"])

    return c


def get_model(model="ana"):
    combined_dict = 0
    return combined_dict


if __name__ == "__main__":
    c = heater_doped_si_oband({"length": 100.4})
    c.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
