import gdsfactory as gf
import gplugins.sax as gsax
import numpy as np
import sax


class bend_s:
    """Name: bend_s
    Description: This is an s-bend using bezier curves.
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
            "size_xy": (10, 4),
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
        self.model = {"bezier": self.get_model_ana}

    def config_to_geometry(self):
        self.wl0 = self.config["wl0"]
        # self.pol = self.config['pol']
        self.size_xy = self.config["size_xy"]
        return None

    # @gf.cell
    def get_component(self):
        c = gf.Component()
        ref = c << gf.components.bend_s(size=self.size_xy)
        c.add_port("o1", port=ref.ports["o1"])
        c.add_port("o2", port=ref.ports["o2"])
        return c

    def get_model_fdtd_test(self, wl=1.55):
        model_data = gsax.read.model_from_npz(
            "../nodes_dummy/straight_waveguide/straight_fba69bc3_f9e2d120.npz"
        )
        return model_data(wl=wl)

    def get_model_ana(self, wl=1.55):
        # TODO: we need to find how long the curve is...
        # for now i approximate this from size-x

        length = self.size_xy[0]
        loss = 0.001
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


if __name__ == "__main__":
    # Example usage

    c = bend_s(config={"size_xy": (100, 4)})
    print(c.component)
    # print(c.component.get_info)
    print(c.config)
    print(type(c.model))

    recnet = sax.RecursiveNetlist.parse_obj(c.component.get_netlist_recursive())
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o2"]))

    # quickplot(c.component)
    # plt.show()
