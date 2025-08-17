import gdsfactory as gf
import gplugins.sax as gsax
import numpy as np
import sax


class straight:
    """Name: straight
    Description: This is a straight single-mode waveguide aka photonic wire.
    ports: 1x1
    NodeLabels:
        - passive
        - 1x1
    Bandwidth: 100 nm.
    """

    def __init__(self, config=None):
        default_config = {
            "wl0": 1.55,
            #   'pol': 'TE',
            "length": 10,
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
        self.model = {"straight": self.get_model_ana}

    def config_to_geometry(self):
        self.wl0 = self.config["wl0"]
        # self.pol = config['pol']
        self.length = self.config["length"]
        return None

    # @gf.cell
    def get_component(self):
        self.component = gf.components.straight(length=self.length)
        return self.component

    def get_model_fdtd_test(self, wl=1.55):
        model_data = gsax.read.model_from_npz(
            "../nodes_dummy/straight_waveguide/straight_fba69bc3_f9e2d120.npz"
        )
        return model_data(wl=wl)

    def get_model_ana(self, wl=1.55, length=10):
        loss = 0.0
        neff = 2.34
        ng = 3.4
        dwl = wl - self.wl0
        dneff_dwl = (ng - neff) / self.wl0
        neff = neff - dwl * dneff_dwl
        phase = 2 * np.pi * neff * length / wl
        transmission = 10 ** (-loss * length / 20) * np.exp(1j * phase)
        sdict = sax.reciprocal(
            {
                ("o1", "o2"): transmission,
            }
        )
        return sdict

    # def update_config(self, new_config):
    #     """
    #     Update the configuration and re-calculate component
    #     """
    #     self.config.update(new_config)
    #     self.component = self.comp()
