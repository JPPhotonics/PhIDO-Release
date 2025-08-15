"""This is a microring resonator with TiN heater for modulation.

---
Name: Microring Resonator with TiN Heater
Description: This is a microring resonator with TiN heater for modulation
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

# from PhotonicsAI.Photon.utils import validate_cell_settings
from PhotonicsAI.KnowledgeBase.DesignLibrary import _mmi1x2, bend_euler, straight

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
def mrr_1x1_heater_tin(gap: float = 0.2, radius: float = 10) -> gf.Component:
    """The component."""
    c = gf.Component()
    ref = c << gf.components.ring_single_heater(gap=0.2, radius=10)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    # params = get_params(settings)
    return c


def get_model(model="fdtd"):
    """Get the model for the edge coupler."""
    m1 = _mmi1x2.get_model(model=model)
    m2 = straight.get_model(model=model)
    m3 = bend_euler.get_model(model=model)
    combined_dict = m1 | m2 | m3
    return combined_dict


if __name__ == "__main__":
    from pprint import pprint

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    matplotlib.use("macosx")

    # c = get_component({'delta_length':100, 'coupling1':0.1, 'coupling2':0.1})
    c = gf.Component()
    ref = c << mrr_1x1_heater_tin()
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
