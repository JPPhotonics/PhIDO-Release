import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.KnowledgeBase.Circuits.mzi_arm import mzi_arm
from PhotonicsAI.KnowledgeBase.Components.directional_coupler import directional_coupler


class mzi_2x2_pn_diode:
    """Name: MZI 2x2 PN Diode
    Description: >
        This is a pn diode.
        The device is operated in reverse bias which causes carrier injectio in the waveguide core.
    NodeLabels:
        - modulator
        - active
        - phase modulation (AM)
    aka: amplitude modulator, PN Junction
    Technology : Plasma Dispersion Effect (PD)
    Design wavelength: 1450-1650 nm
    Optical Bandwidth: 200 nm
    Polarization: TE/TM
    Modulation bandwidth/Switching speed: 20 GHz
    Insertion loss: 6.5 dB
    Extinction ratio: 4.4 dB
    Drive voltage/power: 2 mW
    Inputs: 1
    Outputs: 1.
    """

    def __init__(self, config=None):
        default_config = {
            "fsr": 0.0010,  # in micrometers
            "dy": 80,
            "coupling1": 0.5,
            "coupling2": 0.5,
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
        # self.delta_length = self.config['delta_length']
        self.dy = self.config["dy"]

        self.fsr = self.config["fsr"]
        self.delta_length = int(0.25 / self.fsr)

        self.coupling1 = self.config["coupling1"]
        self.coupling2 = self.config["coupling2"]
        return None

    def get_component(self):
        c = gf.Component()

        c1 = (
            c
            << directional_coupler(
                config={"dy": self.dy, "coupling": self.coupling1}
            ).component
        )
        a1 = c << mzi_arm(config={"length": self.delta_length + 1}).component
        a2 = c << mzi_arm(config={"length": 1}).component
        a2.rotate(180)
        c2 = (
            c
            << directional_coupler(
                config={"dy": self.dy, "coupling": self.coupling2}
            ).component
        )

        a1.connect("o1", c1.ports["o3"])
        a2.connect("o2", c1.ports["o4"])
        c2.connect("o1", a2.ports["o1"])
        c2.connect("o2", a1.ports["o2"])

        c.add_port("o1", port=c1.ports["o1"])
        c.add_port("o2", port=c1.ports["o2"])
        c.add_port("o3", port=c2.ports["o3"])
        c.add_port("o4", port=c2.ports["o4"])
        return c

    def get_circuit_model(self):
        m1 = directional_coupler().model
        m2 = mzi_arm().model
        combined_dict = m1 | m2
        return combined_dict


if __name__ == "__main__":
    # Example usage

    from pprint import pprint

    import matplotlib.pyplot as plt
    from gdsfactory.quickplotter import quickplot

    c = mzi1(config={"delta_length": 50, "coupling1": 0.1, "coupling2": 0.1})
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
    S41 = _c(wl=wl)["o1", "o4"]
    S31 = _c(wl=wl)["o1", "o3"]
    S42 = _c(wl=wl)["o2", "o4"]
    S32 = _c(wl=wl)["o2", "o3"]
    plt.plot(wl, np.abs(S41) ** 2)
    plt.plot(wl, np.abs(S31) ** 2)
    plt.plot(wl, np.abs(S42) ** 2)
    plt.plot(wl, np.abs(S32) ** 2)
    plt.show()
