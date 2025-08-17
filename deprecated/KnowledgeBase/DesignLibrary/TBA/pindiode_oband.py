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
aka: amplitude modulator, PIN diode, Variable Optical Attenuator (VOA)
Technology : Plasma Dispersion Effect
Design wavelength: 1260-1360 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 1 GHz
Insertion loss: 2-20 dB
Extinction ratio: N/A
Drive voltage/power: 2 mW
Footprint Estimate: 510.0um x 41.4um
"""

import gdsfactory as gf


@gf.cell
def pindiode_oband(settings: dict = {}):
    length = 500
    c = gf.Component()

    wg_xs = gf.cross_section.cross_section(width=0.41, offset=0, layer=(1, 0))
    wg = c << gf.path.straight(length=length + 10).extrude(wg_xs)

    slab_xs = gf.cross_section.cross_section(width=25.1, offset=0, layer=(3, 0))
    slab = c << gf.path.straight(length=length).movex(5).extrude(slab_xs)

    slab_taper1 = c << gf.components.taper(
        length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
    )
    slab_taper1.movex(-5).rotate(180)

    slab_taper2 = c << gf.components.taper(
        length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
    )
    slab_taper2.movex(length + 5)

    npp_xs = gf.Section(width=11, offset=6.55, layer=(24, 0))
    ppp_xs = gf.Section(width=11, offset=-6.55, layer=(25, 0))
    doping_xs = gf.CrossSection(sections=[npp_xs, ppp_xs])
    doping = c << gf.path.straight(length=length - 2).movex(6).extrude(doping_xs)

    metal_top_xs = gf.Section(width=17, offset=-12.2, layer=(41, 0))
    metal_bottom_xs = gf.Section(width=17, offset=12.2, layer=(41, 0))
    metal_xs = gf.CrossSection(sections=[metal_top_xs, metal_bottom_xs])
    metal = c << gf.path.straight(length=length).movex(5).extrude(metal_xs)

    via_top_xs = gf.Section(width=7.3, offset=-7.85, layer=(40, 0))
    via_bottom_xs = gf.Section(width=7.3, offset=7.85, layer=(40, 0))
    metal_xs = gf.CrossSection(sections=[via_top_xs, via_bottom_xs])
    metal = c << gf.path.straight(length=length - 3).movex(6.5).extrude(metal_xs)

    c.add_port("o1", port=wg.ports["o1"])
    c.add_port("o2", port=wg.ports["o2"])
    return c


def get_model(model="ana"):
    combined_dict = 0
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = pindiode_oband()
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
