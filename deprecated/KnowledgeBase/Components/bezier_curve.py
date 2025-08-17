import math

import gdsfactory as gf
import gplugins.sax as gsax
import numpy as np
import sax


class bezier_curve:
    """Name: bezier_curve
    Description: This is a bend (or generally arc shaped) waveguide. The function maps a radii and and an angle to a 4 control points that implements the Bezier Curve.
    ports: 1x1
    NodeLabels:
        - passive
        - 1x1
    Bandwidth:
    """

    def __init__(self, config=None):
        default_config = {"radius": 10, "angle": 90}
        if config is None:
            config = default_config
        else:
            config = {**default_config, **config}

        self.config = config
        self.component = None
        self.model = None

        _ = self.config_to_geometry()
        self.component = self.get_component()
        self.model = {"bezier_curve": self.get_model_ana}

    def config_to_geometry(self):
        # self.wl0 = self.config['wl0']
        # self.pol = self.config['pol']
        self.radius = self.config["radius"]
        self.angle = self.config["angle"]
        return None

    def get_component(self):
        """https://stackoverflow.com/questions/30277646/svg-convert-arcs-to-cubic-bezier."""
        c = gf.Component()

        rad_angle = ((self.angle - 180) * math.pi) / 180

        x1 = self.radius
        y1 = 0

        x2 = self.radius
        y2 = self.radius * (4 / 3) * math.tan(rad_angle / 4)

        x3 = self.radius * (
            math.cos(rad_angle)
            + (4 / 3 * math.tan(rad_angle / 4) * math.sin(rad_angle))
        )
        y3 = self.radius * (
            math.sin(rad_angle)
            - (4 / 3 * math.tan(rad_angle / 4) * math.cos(rad_angle))
        )

        x4 = self.radius * math.cos(rad_angle)
        y4 = self.radius * math.sin(rad_angle)

        ref = c << gf.components.bezier(
            control_points=((x1, y1), (x2, y2), (x3, y3), (x4, y4)),
            npoints=201,
            with_manhattan_facing_angles=True,
        )

        c.add_port("o1", port=ref.ports["o1"])
        c.add_port("o2", port=ref.ports["o2"])

        return c

    def get_model_fdtd_test(self, wl=1.55):
        model_data = gsax.read.model_from_npz(
            "../nodes_dummy/straight_waveguide/straight_fba69bc3_f9e2d120.npz"
        )
        return model_data(wl=wl)

    def get_model_ana(self, wl=1.55):
        neff = 2.34
        ng = 3.4

        def os(x):
            return 0.01 * np.cos(24 * np.pi * x) + 0.01

        # loss=0.0
        loss = os(wl)

        wl0 = 1.55
        length = self.radius * self.angle * (np.pi / 180)

        dwl = wl - wl0
        dneff_dwl = (ng - neff) / wl0
        neff = neff - dwl * dneff_dwl
        phase = 2 * np.pi * neff * length / wl
        transmission = 10 ** (-loss * length / 20) * np.exp(1j * phase)
        sdict = sax.reciprocal(
            {
                ("o1", "o2"): transmission,
            }
        )
        return sdict


if __name__ == "__main__":
    # Example usage

    c = bezier_curve({"radius": 10, "angle": 90})
    print(c.component)
    # print(c.component.get_info)
    print(c.config)
    print("type of c.model", type(c.model))

    c.component.show()

    """ recnet = sax.RecursiveNetlist.parse_obj(c.component.get_netlist_recursive())
    print('Required Models ==>', sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    print( _c(wl = 1.55) )
    print( np.abs(_c(wl = 1.35)['o1','o2']) )

    # quickplot(c.component)
    # plt.show()

    wl = np.linspace(1.5, 1.6, 500)
    S21 = np.abs(_c(wl = wl)['o1','o2'])**2
    plt.plot(wl, S21)
    plt.show() """
