import gdsfactory as gf
import gplugins.sax as gsax
import numpy as np
import sax


class mmi2x2:
    """Name: mmi2x2
    Description: A multimode interferometer with two input and two output ports.
    ports: 2x2
    NodeLabels:
        - passive
        - 2x2
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
        self.model = {"mmi2x2": self.get_model_ana}

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

        m = gf.components.mmi2x2(length_mmi=self.length, length_taper=5, gap_mmi=0.4)
        coupler_r = c << m
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

    def get_model_ana(self, wl=1.5, coupling: float = 0.5):
        """A simple coupler model."""
        # wg_factor = np.exp(1j * np.pi * 2.34 * 1 / wl)
        wg_factor = 1
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
    # from gdsfactory.quickplotter import quickplot

    c = mmi2x2(config={"coupling": 0.5})
    print(c.component)
    print(c.config)

    recnet = sax.RecursiveNetlist.model_validate(
        c.component.get_netlist(recursive=True)
    )
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o4"]) ** 2)

    c.component.plot()
    plt.show()
