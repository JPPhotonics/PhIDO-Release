import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.KnowledgeBase.Circuits.mzi1 import mzi1


class wdm1:
    """Name: wdm1
    Description: >
        This is a four channel WDM (wavelength division multiplexer)
        for the following wavelengths: 1308 nm , 1309 nm, 1310 nm, 1311 nm.
        It is based on a tree of MZIs.
    ports: 2x4
    NodeLabels:
        - active
        - 2x4
    Bandwidth: 50 nm.
    """

    def __init__(self, config=None):
        default_config = {
            "dl": [100.965, 100.965 * 1.5, 100.965 * 1.25],
            "coupler_dy": 120,
        }
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
        self.dl = self.config["dl"]
        self.coupler_dy = self.config["coupler_dy"]
        return None

    def get_component(self):
        c = gf.Component()

        c1 = (
            c
            << mzi1(
                config={"delta_length": self.dl[0], "dy": self.coupler_dy}
            ).component
        )
        c2 = (
            c
            << mzi1(
                config={"delta_length": self.dl[1], "dy": self.coupler_dy}
            ).component
        )
        c3 = (
            c
            << mzi1(
                config={"delta_length": self.dl[2], "dy": self.coupler_dy}
            ).component
        )

        c2.connect("o1", c1.ports["o3"])
        c3.connect("o2", c1.ports["o4"])

        c.add_port("o1", port=c1.ports["o1"])
        c.add_port("o2", port=c1.ports["o2"])
        c.add_port("o3", port=c2.ports["o3"])
        c.add_port("o4", port=c2.ports["o4"])
        c.add_port("o5", port=c3.ports["o3"])
        c.add_port("o6", port=c3.ports["o4"])
        return c

    def get_circuit_model(self):
        m1 = mzi1().model
        return m1


if __name__ == "__main__":
    # Example usage

    from pprint import pprint

    import matplotlib.pyplot as plt
    from gdsfactory.quickplotter import quickplot

    c = wdm1()
    print(c.component)
    print(c.config)
    print(c.model)
    quickplot(c.component)
    plt.show()

    recnet = sax.RecursiveNetlist.parse_obj(c.component.get_netlist_recursive())
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    pprint(_c(wl=1.55))
    # # print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )
    wl = np.linspace(1.29, 1.33, 500)

    S32 = _c(wl=wl)["o2", "o3"]
    S42 = _c(wl=wl)["o2", "o4"]
    S52 = _c(wl=wl)["o2", "o5"]
    S62 = _c(wl=wl)["o2", "o6"]

    plt.plot(wl, np.abs(S32) ** 2)
    plt.plot(wl, np.abs(S42) ** 2)
    plt.plot(wl, np.abs(S52) ** 2)
    plt.plot(wl, np.abs(S62) ** 2)
    plt.show()
