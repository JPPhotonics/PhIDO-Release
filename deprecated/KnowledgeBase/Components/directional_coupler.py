import gdsfactory as gf
import gplugins.sax as gsax
import numpy as np
import sax


class directional_coupler:
    """Name: directional_coupler
    Description: >
        A directional coupler with two input and two output ports.
        Can be used for power splitting.
    ports: 2x2
    NodeLabels:
        - passive
    Bandwidth: 50 nm.
    """

    def __init__(self, config=None):
        default_config = {
            "wl0": 1.55,
            #   'pol':'TE',
            "coupling": 0.5,
            "dy": 5,
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
        self.model = {"coupler": self.get_model_ana}

    def config_to_geometry(self):
        """Provides mapping from design config to geometric settings of gdsfacotory component."""
        self.wl0 = self.config["wl0"]
        self.dy = self.config["dy"]
        # self.pol = config['pol']
        self.coupling = self.config["coupling"]

        def x_to_y(x):
            return 20 * x + 2  # dummy mapping

        self.length = x_to_y(self.coupling)
        return None

    # @gf.cell
    def get_component(self):
        c = gf.Component()
        # wg = straight(config={'length':self.length}).get_component()
        # bend = bend_s().get_component()

        # wg_top = c << wg
        # wg_top.movey(+self.gap/2)
        # bend_top_left = c << bend.mirror()
        # bend_top_right = c << bend
        # bend_top_left.connect("o1", destination=wg_top.ports["o1"])
        # bend_top_right.connect("o1", destination=wg_top.ports["o2"])

        # wg_bot = c << wg
        # wg_bot.movey(-self.gap/2)
        # bend_bot_left = c << bend
        # bend_bot_right = c << bend.mirror()
        # bend_bot_left.connect("o2", destination=wg_bot.ports["o1"])
        # bend_bot_right.connect("o2", destination=wg_bot.ports["o2"])

        # c.add_port("o1", port=bend_bot_left.ports["o1"])
        # c.add_port("o2", port=bend_top_left.ports["o2"])
        # c.add_port("o3", port=bend_top_right.ports["o2"])
        # c.add_port("o4", port=bend_bot_right.ports["o1"])

        coupler = gf.components.coupler(length=self.length, dy=self.dy)
        coupler_r = c << coupler
        c.add_port("o1", port=coupler_r.ports["o1"])
        c.add_port("o2", port=coupler_r.ports["o2"])
        c.add_port("o3", port=coupler_r.ports["o3"])
        c.add_port("o4", port=coupler_r.ports["o4"])

        self.component = c
        return self.component

    def get_model_fdtd_test(self, wl=1.55):
        model_data = gsax.read.model_from_npz(
            "../FDTD-test/coupler/coupler_2e5eb039_1ce252ac.npz"
        )
        return model_data(wl=wl)

    def get_model_ana(self, wl=1.5, coupling: float = 0.5, length=12):
        """A simple coupler model."""
        # wg_factor = np.exp(1j * np.pi * 2.34 * 1 / wl)
        wg_factor = 1
        coupling = (np.sin(np.pi * length / 6) + 1) / 2
        print("====COUPLING===>", coupling)
        kappa = wg_factor * coupling**0.5
        tau = wg_factor * (1 - coupling) ** 0.5
        sdict = sax.reciprocal(
            {
                ("o1", "o3"): tau,
                ("o1", "o4"): 1j * kappa,
                ("o2", "o3"): 1j * kappa,
                ("o2", "o4"): tau,
            }
        )
        return sdict


if __name__ == "__main__":
    # Example usage
    print("running main")

    import matplotlib.pyplot as plt
    from gdsfactory.quickplotter import quickplot

    c = directional_coupler(config={"coupling": 0.1, "dy": 5})
    print(c.component)
    print(c.config)

    recnet = sax.RecursiveNetlist.parse_obj(c.component.get_netlist_recursive())
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o4"]) ** 2)

    quickplot(c.component)
    plt.show()
