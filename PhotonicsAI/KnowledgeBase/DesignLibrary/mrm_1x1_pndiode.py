"""This is a microring modulator with PN Diodes for high speed modulation.

---
Name: Microring Modulator with PN Diode
Description: This is a microring modulator with PN Diodes for high speed modulation
ports: 1x1
NodeLabels:
    - active
    - 1x1
Bandwidth: 50 nm
Args:
    -length: straight length heater (um)
    -delta_length: path length difference (um). bottom arm vertical extra length.
"""

import gdsfactory as gf
import numpy as np
import sax

# from PhotonicsAI.Photon.utils import validate_cell_settings
from PhotonicsAI.KnowledgeBase.DesignLibrary import _mmi1x2, bend_euler, straight

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz, model_from_tidy3d

# args = {
#     'functional': {
#     },
#     'geometrical': {
#         'coupling1':    {'default': 0.5, 'range': (0, 1)},
#         'coupling2':    {'default': 0.5, 'range': (0, 1)},
#         'length':       {'default': 10.0, 'range': (0.1, 1000.0)},
#         'delta_length': {'default': 2.0, 'range': (0.1, 1000.0)},
#         'dy':           {'default': 4.0, 'range': (1, 1000.0)},
#     }
# }


@gf.cell
def mrm_1x1_pndiode(gap: float = 0.3, radius: float = 5) -> gf.Component:
    """The component."""
    c = gf.Component()
    ref = c << gf.components.ring_single_pn(gap=0.3, radius=5)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.flatten()
    # params = get_params(settings)
    return c

def get_model(model="tidy3d"):
    """Get the model."""
    if model == "ana":
        return {"mrm_1x1_pndiode": get_model_ana}
    if model == "fdtd":
        return {"mrm_1x1_pndiode": get_model_fdtd}
    if model == "tidy3d":
        return {"mrm_1x1_pndiode": get_model_tidy3d}


def get_model_tidy3d(wl=1.55):
    try:
        with open('build/modified_netlist.yml', 'r') as file:
            modified_netlist = yaml.safe_load(file)
        if "mrm_1x1_pndiode" in modified_netlist.split():
            c = gf.read.from_yaml(modified_netlist)
        else:
            c = mrm_1x1_pndiode()
    except:
        c = mrm_1x1_pndiode()
    model_data = model_from_tidy3d(c=c)
    return model_data(wl=wl)

def get_model_fdtd(wl=1.5, length=10.0, neff=3.2) -> sax.SDict:
    """Get FDTD model."""
    return sax.reciprocal({("o1", "o2"): np.exp(2j * np.pi * neff * length / wl)})

if __name__ == "__main__":
    from pprint import pprint

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    matplotlib.use("macosx")

    # c = get_component({'delta_length':100, 'coupling1':0.1, 'coupling2':0.1})
    c = gf.Component()
    ref = c << mrm_1x1_pndiode()
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    pprint(c.get_netlist())
    print()
    # sys.exit()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model())
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o2"]) ** 2)

    c.plot()

    plt.figure()
    wl = np.linspace(1.5, 1.6, 500)
    S21 = _c(wl=wl)["o1", "o2"]
    plt.plot(wl, np.abs(S21) ** 2)
    plt.show()
