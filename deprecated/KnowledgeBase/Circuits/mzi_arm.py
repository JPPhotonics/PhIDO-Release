import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.KnowledgeBase.Components.bend_euler import bend_euler
from PhotonicsAI.KnowledgeBase.Components.straight import straight


class mzi_arm:
    """Name: mzi_arm
    Description: This is a Mach-Zehnder interferometer (MZI) arm.
    ports: 1x1
    NodeLabels:
        - passive
        - 1x1
    Bandwidth: 100 nm.
    """

    def __init__(self, config=None):
        default_config = {"length": 20}
        if config is None:
            config = default_config
        else:
            config = {**default_config, **config}

        self.config = config
        self.component = None
        self.model = None

        _ = self.config_to_geometry()
        self.component = self.get_component()
        self.model = self.get_circuit_model()  # TODO

    def config_to_geometry(self):
        self.length = self.config["length"]
        return None

    def get_component(self):
        c = gf.Component()
        a1 = c << bend_euler().component
        a2 = c << straight(config={"length": self.length / 2}).component
        a3 = c << bend_euler().component
        a4 = c << bend_euler().component
        a5 = c << straight(config={"length": self.length / 2}).component
        a6 = c << bend_euler().component

        a2.connect("o1", a1.ports["o2"])
        a3.connect("o2", a2.ports["o2"])
        a4.connect("o2", a3.ports["o1"])
        a5.connect("o1", a4.ports["o1"])
        a6.connect("o1", a5.ports["o2"])

        c.add_port("o1", port=a1.ports["o1"])
        c.add_port("o2", port=a6.ports["o2"])
        return c

    def get_circuit_model(self, wl=1.55):
        m1 = bend_euler().model
        m2 = straight().model
        combined_dict = m1 | m2
        return combined_dict


if __name__ == "__main__":
    # Example usage

    import matplotlib.pyplot as plt
    # from gdsfactory.quickplotter import quickplot

    c = mzi_arm(config={"length": 200})
    print(c.component)
    print(c.config)
    print("type of c.model", type(c.model))
    print(c.model)
    c.plot()
    # quickplot(c.component)
    plt.show()

    recnet = sax.RecursiveNetlist.model_validate(
        c.component.get_netlist(recursive=True)
    )
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o2"]))

    wl = np.linspace(1.5, 1.6, 500)
    S21 = _c(wl=wl)["o1", "o2"]
    # plt.plot(wl, np.abs(S21)**2)
    plt.plot(wl, np.angle(S21))
    plt.show()
