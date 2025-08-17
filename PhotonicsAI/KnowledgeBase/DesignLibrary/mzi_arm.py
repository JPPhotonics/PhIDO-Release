"""This is a Mach-Zehnder interferometer (MZI) arm.

---
Name: mzi_arm
Description: This is a Mach-Zehnder interferometer (MZI) arm.
ports: 1x1
NodeLabels:
    - passive
    - 1x1
Bandwidth: 100 nm
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import bend_euler, straight

# from PhotonicsAI.Photon.utils import validate_cell_settings

# args = {
#     'functional': {
#     },
#     'geometrical': {
#         'length':   {'default': 20.0, 'range': (0.1, 1000.0)},
#     }
# }


@gf.cell
def mzi_arm(
    length_y: float = 1,
    # length_x: float = 0.1,
) -> gf.Component:
    """The component."""
    c = gf.Component()
    a1 = c << bend_euler.bend_euler()
    a2 = c << straight.straight(length=length_y)
    a3 = c << bend_euler.bend_euler()
    a4 = c << bend_euler.bend_euler()
    a5 = c << straight.straight(length=length_y)
    a6 = c << bend_euler.bend_euler()

    a2.connect("o1", a1.ports["o2"])
    a3.connect("o2", a2.ports["o2"])
    a4.connect("o2", a3.ports["o1"])
    a5.connect("o1", a4.ports["o1"])
    a6.connect("o1", a5.ports["o2"])

    c.add_port("o1", port=a1.ports["o1"])
    c.add_port("o2", port=a6.ports["o2"])

    # params = get_params(settings)

    return c


def get_model(model="tidy3d"):
    """The model."""
    m1 = bend_euler.get_model(model=model)
    m2 = straight.get_model(model=model)
    combined_dict = m1 | m2
    return combined_dict


if __name__ == "__main__":
    from pprint import pprint

    import sax

    c = gf.Component()
    ref = c << mzi_arm(length_y=100)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])

    pprint(c.get_netlist())
    print()
    # sys.exit()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    # _c, info = sax.circuit(recnet, get_model())
    # print( _c(wl = 1.55) )
    # print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )

    # c.plot()
    # plt.show()
