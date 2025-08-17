import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.KnowledgeBase.Circuits.mzi_arm import mzi_arm
from PhotonicsAI.KnowledgeBase.Components.mmi1x2 import mmi1x2


class mzm1:
    """Name: mzi1
    Description: This is a Mach-Zehnder modulator (MZM), with one input and one output port.
    ports: 1x1
    NodeLabels:
        - active
        - 1x1
    Bandwidth: 50 nm.
    """

    def __init__(self, config=None):
        default_config = {
            "fsr": 0.005,
        }  # in micrometers

        if config is None:
            config = default_config
        else:
            config = {**default_config, **config}

        self.config = config
        self.component = None
        self.model = None

        _ = self.config_to_geometry()
        self.component = self.get_component()
        self.model = self.get_circuit_model()

    def config_to_geometry(self):
        self.fsr = self.config["fsr"]

        self.delta_length = int(0.7 / self.fsr)
        return None

    def get_component(self):
        c = gf.Component()

        c1 = c << mmi1x2().component
        a1 = c << mzi_arm(config={"length": self.delta_length + 1}).component
        a2 = c << mzi_arm(config={"length": 1}).component
        a2.rotate(180)
        c2 = c << mmi1x2().component

        a1.connect("o1", c1.ports["o2"])
        a2.connect("o2", c1.ports["o3"])
        c2.connect("o2", a2.ports["o1"])
        c2.connect("o3", a1.ports["o2"])

        c.add_port("o1", port=c1.ports["o1"])
        c.add_port("o2", port=c2.ports["o1"])
        return c

    def get_circuit_model(self):
        m1 = mmi1x2().model
        m2 = mzi_arm().model
        combined_dict = m1 | m2
        return combined_dict


if __name__ == "__main__":
    # Example usage

    from pprint import pprint

    import matplotlib.pyplot as plt
    from gdsfactory.quickplotter import quickplot

    c = mzm1(config={"fsr": 0.005})
    print(c.component)
    print(c.config)
    print(c.model)
    quickplot(c.component)
    plt.show()

    recnet = sax.RecursiveNetlist.parse_obj(c.component.get_netlist_recursive())
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    pprint(_c(wl=1.55))
    # print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )
    wl = np.linspace(1.5, 1.6, 500)
    S21 = _c(wl=wl)["o1", "o2"]
    plt.plot(wl, np.abs(S21) ** 2)
    plt.show()
