"""This is a PN diode.

---
Name: PN Diode
Description: >
    This is a pn diode.
    The device is operated in reverse bias which causes carrier injectio in the waveguide core.
ports: 1x1
NodeLabels:
    - modulator
    - active
    - phase modulation (PM)
aka: phase modulator, PN Junction, high-speed phase modulator
Technology : Plasma Dispersion Effect (PD)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 20 GHz
Insertion loss: 6.5 dB
Extinction ratio: N/A
Drive voltage/power: 2 mW
Inputs: 1
Outputs: 1
Estimated Footprint: 640.0um x 168.5um
Args:
    -straight length pn diode (um)
"""

import gdsfactory as gf
from gdsfactory.typings import CrossSectionSpec

from PhotonicsAI.KnowledgeBase.DesignLibrary import straight


@gf.cell
def pndiode(
    length: float = 600, cross_section: CrossSectionSpec = "strip"
) -> gf.Component:
    """The component."""
    c = gf.Component()

    wg_prop_xs = gf.cross_section.cross_section(width=0.7, offset=0, layer=(1, 0))
    _wg_prop = c << gf.path.straight(length=length).extrude(wg_prop_xs)

    wg_top_xs = gf.Section(width=9, offset=6.25, layer=(1, 0))
    wg_bottom_xs = gf.Section(width=9, offset=-6.25, layer=(1, 0))
    wg_xs = gf.CrossSection(sections=[wg_top_xs, wg_bottom_xs])
    _wg = c << gf.path.straight(length=length - 1).dmovex(0.5).extrude(wg_xs)

    wg1_taper1 = c << gf.components.taper(
        length=20, width1=0.7, width2=0.5, port=None, layer=(1, 0)
    )
    _wg1_taper1 = wg1_taper1.drotate(180)

    wg1_taper2 = c << gf.components.taper(
        length=20, width1=0.7, width2=0.5, port=None, layer=(1, 0)
    )
    wg1_taper2.dmovex(length)

    slab_xs = gf.cross_section.cross_section(width=22.7, offset=0, layer=(3, 0))
    _slab = c << gf.path.straight(length=length).extrude(slab_xs)

    slab_taper1 = c << gf.components.taper(
        length=20, width1=5, width2=0.3, port=None, layer=(3, 0)
    )
    slab_taper1.drotate(180)

    slab_taper2 = c << gf.components.taper(
        length=20, width1=5, width2=0.3, port=None, layer=(3, 0)
    )
    slab_taper2.dmovex(length)

    np_xs = gf.Section(width=12.2, offset=5.84, layer=(20, 0))
    pp_xs = gf.Section(width=12.2, offset=-5.84, layer=(21, 0))
    doping1_xs = gf.CrossSection(sections=[np_xs, pp_xs])
    _doping1 = c << gf.path.straight(length=length + 1.5).dmovex(-1).extrude(doping1_xs)

    npp_xs = gf.Section(width=11, offset=6.35, layer=(24, 0))
    ppp_xs = gf.Section(width=11, offset=-6.35, layer=(25, 0))
    doping2_xs = gf.CrossSection(sections=[npp_xs, ppp_xs])
    _doping2 = c << gf.path.straight(length=length + 1.5).dmovex(-1).extrude(doping2_xs)

    viac_top_xs = gf.Section(width=6, offset=-6.35, layer=(40, 0))
    viac_bottom_xs = gf.Section(width=6, offset=6.35, layer=(40, 0))
    viac_xs = gf.CrossSection(sections=[viac_top_xs, viac_bottom_xs])
    _viac = c << gf.path.straight(length=length - 4).dmovex(2).extrude(viac_xs)

    m1_top_xs = gf.Section(width=22.5, offset=-12.6, layer=(41, 0))
    m1_bottom_xs = gf.Section(width=22.5, offset=12.6, layer=(41, 0))
    m1_xs = gf.CrossSection(sections=[m1_top_xs, m1_bottom_xs])
    _m1 = c << gf.path.straight(length=length + 3).dmovex(-1.5).extrude(m1_xs)

    via1_top_xs = gf.Section(width=4, offset=19.85, layer=(44, 0))
    via1_bottom_xs = gf.Section(width=4, offset=-19.85, layer=(44, 0))
    via1_xs = gf.CrossSection(sections=[via1_top_xs, via1_bottom_xs])
    _via1 = c << gf.path.straight(length=71).dmovex((length - 71) / 2).extrude(via1_xs)

    m2_top_xs = gf.Section(width=68.5, offset=50, layer=(45, 0))
    m2_bottom_xs = gf.Section(width=68.5, offset=-50, layer=(45, 0))
    m2_xs = gf.CrossSection(sections=[m2_top_xs, m2_bottom_xs])
    _m2 = c << gf.path.straight(length=75).dmovex((length - 75) / 2).extrude(m2_xs)

    pad_top_xs = gf.Section(width=64.5, offset=50, layer=(46, 0))
    pad_bottom_xs = gf.Section(width=64.5, offset=-50, layer=(46, 0))
    pad_xs = gf.CrossSection(sections=[pad_top_xs, pad_bottom_xs])
    _pad = c << gf.path.straight(length=71).dmovex((length - 71) / 2).extrude(pad_xs)

    c.add_port("o1", port=wg1_taper1.ports["o2"])
    c.add_port("o2", port=wg1_taper2.ports["o2"])
    c.flatten()
    return c


def get_model(model="tidy3d"):
    """The model."""
    return {"pndiode": straight.get_model_fdtd}


# class pn_diode:
#     def __init__(self, config=None):
#         default_config = {'fsr': 0.0010, # in micrometers
#                           'dy': 80,
#                           'coupling1': 0.5,
#                           'coupling2': 0.5}
#         if config is None:
#             config = default_config
#         else:
#             config = {**default_config, **config}

#         self.config = config
#         self.component = None
#         self.model = None

#         _ = self.config_to_geometry()
#         self.component = self.get_component()
#         self.model = self.get_circuit_model() # TODO

#     def config_to_geometry(self):
#         # self.delta_length = self.config['delta_length']
#         self.dy = self.config['dy']

#         self.fsr = self.config['fsr']
#         self.delta_length = int(0.25/self.fsr)

#         self.coupling1 = self.config['coupling1']
#         self.coupling2 = self.config['coupling2']
#         return None


if __name__ == "__main__":
    c = pndiode()
    c.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
