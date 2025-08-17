"""This is a grating coupler used for I/O to the PIC.

---
Name: _gc
Description: This is a grating coupler used for I/O to the PIC.
ports: 1x0
NodeLabels:
    - passive
    - 1x1
Bandwidth: 100 nm
"""

import gdsfactory as gf
import numpy as np
import sax
from gdsfactory.typings import CrossSectionSpec, LayerSpec

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz, model_from_tidy3d

@gf.cell
def _gc(
    polarization: str = "te",
    taper_length: float = 16.6,
    taper_angle: float = 40.01,
    wavelength: float = 1.554,
    fiber_angle: float = 15.01,
    grating_line_width: float = 0.343,
    neff: float = 2.638,
    nclad: float = 1.443,
    n_periods: int = 30,
    big_last_tooth: bool = False,
    layer_slab: LayerSpec | None = "SLAB90",
    slab_xmin: float = -1.01,
    slab_offset: float = 2.01,
    spiked: bool = True,
    cross_section: CrossSectionSpec = "strip",
) -> gf.Component:
    c = gf.Component()

    xs_wg = gf.cross_section.cross_section(width=0.4, offset=0, layer=(1, 0))
    coupler = c << gf.components.grating_coupler_elliptical_uniform(
        n_periods=32,
        period=0.63,
        fill_factor=0.5,
        taper_length=18.427,
        taper_angle=52,
        spiked=False,
        cross_section=xs_wg,
        layer_slab=False,
    )
    # coupler.drotate(90)

    taper_wg = c << gf.components.taper(length=12, width1=0.5, width2=0.4)

    slab90 = c << gf.components.rectangle(size=(33.587, 43.15), layer=(3, 0))
    slab90.dmove((10.158, -21.575))
    # slab90.drotate(90)

    taper_wg.connect("o2", coupler.ports["o1"])

    c.add_port("o1", port=taper_wg.ports["o1"])

    c.flatten()
    return c


def get_model(model="tidy3d"):
    if model == "ana":
        return {"_gc": get_model_ana}
    if model == "fdtd":
        return {"_gc": get_model_fdtd}
    if model == "tidy3d":
        return {"_gc": get_model_tidy3d}


def get_model_tidy3d(wl=1.55):
    try:
        with open('build/modified_netlist.yml', 'r') as file:
            modified_netlist = yaml.safe_load(file)
        if "_gc" in modified_netlist.split():
            c = gf.read.from_yaml(modified_netlist)
        else:
            c = _gc()
    except:
        c = _gc()
    model_data = model_from_tidy3d(c=c)
    return model_data(wl=wl)


def get_model_fdtd(wl=1.5, length=10.0, neff=3.2) -> sax.SDict:
    return sax.reciprocal({("o1", "o2"): np.exp(2j * np.pi * neff * length / wl)})


def get_model_ana(wl=1.55):
    # TODO: we need to find how long the curve is...
    # for now i approximate this from size-x
    wl0 = 1.55
    size_xy = (100, 4)

    length = size_xy[0]
    loss = 0.001
    neff = 2.34
    ng = 3.4
    dwl = wl - wl0
    dneff_dwl = (ng - neff) / wl0
    neff = neff - dwl * dneff_dwl
    phase = 2 * np.pi * neff * length / wl
    transmission = 10 ** (-loss * length / 20) * np.exp(1j * phase)
    sdict = sax.reciprocal(
        {
            ("o1", "o2"): transmission,
        }
    )
    return sdict


if __name__ == "__main__":
    from pprint import pprint

    import matplotlib.pyplot as plt

    c = gf.Component()

    ref1 = c << _gc()
    # ref1.dmirror()

    pprint(c.get_netlist())
    # print()

    # recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    # print('Required Models ==>', sax.get_required_circuit_models(recnet))

    # _c, info = sax.circuit(recnet, get_model())
    # print( _c(wl = 1.55) )
    # print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )

    c.plot()
    plt.show()
