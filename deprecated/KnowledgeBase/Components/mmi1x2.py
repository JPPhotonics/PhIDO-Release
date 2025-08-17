import gdsfactory as gf
import gplugins.sax as gsax
import numpy as np
import sax


class mmi1x2:
    """Name: mmi1x2
    Description: >
        This multimode interferometer has one input and two output ports.
        It functions as a beamsplitter or power splitter, dividing the input equally between the two outputs.
        Each output receives half of the input power, ensuring balanced splitting.
    ports: 1x2
    NodeLabels:
        - passive
        - 1x2
    Bandwidth: 50 nm.
    """

    def __init__(self, config=None):
        default_config = {
            "wl0": 1.55,
            #   'pol':'TE',
            "coupling": 0.5,
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
        self.model = {"mmi1x2": self.get_model_ana}

    def config_to_geometry(self):
        """Provides mapping from design config to geometric settings of gdsfacotory component."""
        self.wl0 = self.config["wl0"]
        # self.pol = config['pol']
        self.coupling = self.config["coupling"]

        def x_to_y(x):
            return 20 * x + 2  # dummy mapping

        self.length = x_to_y(self.coupling)
        return None

    # @gf.cell
    def get_component(self):
        c = gf.Component()

        m = gf.components.mmi1x2(length_mmi=self.length, length_taper=5, gap_mmi=0.4)
        coupler_r = c << m
        c.add_port("o1", port=coupler_r.ports["o1"])
        c.add_port("o2", port=coupler_r.ports["o2"])
        c.add_port("o3", port=coupler_r.ports["o3"])

        self.component = c
        return self.component

    def get_model_fdtd_test(self, wl=1.55):
        model_data = gsax.read.model_from_npz(
            "../FDTD-test/coupler/coupler_2e5eb039_1ce252ac.npz"
        )
        return model_data(wl=wl)

    def get_model_ana(self, wl=1.5, coupling: float = 0.5):
        """A simple coupler model."""
        # wg_factor = np.exp(1j * np.pi * 2.34 * 1 / wl)
        wg_factor = 1
        kappa = wg_factor * coupling**0.5
        tau = wg_factor * (1 - coupling) ** 0.5
        sdict = sax.reciprocal(
            {
                ("o1", "o2"): tau,
                ("o1", "o3"): 1j * kappa,
            }
        )
        return sdict


if __name__ == "__main__":
    # Example usage
    print("running main")

    import matplotlib.pyplot as plt
    from gdsfactory.quickplotter import quickplot

    c = mmi1x2(config={"coupling": 0.5})
    print(c.component)
    print(c.config)

    recnet = sax.RecursiveNetlist.parse_obj(c.component.get_netlist_recursive())
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o2"]) ** 2)

    quickplot(c.component)
    plt.show()
