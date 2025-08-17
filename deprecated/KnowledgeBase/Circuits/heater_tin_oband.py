import gdsfactory as gf


class heater_tin_oband:
    """Name: Titanium Nitride (TiN) Heater
    Description: Thermo-optic phase shifter with a Titanium nitride heating element.
    ports: 1x1
    NodeLabels:
        - modulator
        - active
        - phase modulation (PM)
    aka: phase modulator, thermo-optic phase shifter, heater
    Technology : Thermo-optic effect (TO)
    Design wavelength: 1260-1360 nm
    Optical Bandwidth: 200 nm
    Polarization: TE/TM
    Modulation bandwidth/Switching speed: 30 KHz
    Insertion loss: 1 dB
    Extinction ratio: 25 dB
    Drive voltage/power: 25 mW
    Footprint Estimate: 327.0um x 40.0um.
    """

    def __init__(self, config=None):
        default_config = {
            "length": 324  # in micrometers
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
        self.length = self.config["length"]

        return None

    def get_component(self):
        c = gf.Component()

        wg_xs = gf.cross_section.cross_section(width=0.41, offset=0, layer=(1, 0))
        wg = c << gf.path.straight(length=self.length + 24).extrude(wg_xs)
        wg.move((-12, 0))

        mh_xs = gf.cross_section.cross_section(width=3.5, offset=0, layer=(47, 0))
        c << gf.path.straight(length=self.length).extrude(mh_xs)

        mhsq_xs = gf.cross_section.cross_section(width=12, offset=0, layer=(47, 0))

        mh1 = c << gf.path.straight(length=12).extrude(mhsq_xs)
        mh1.move((-12, 4.25))

        mh2 = c << gf.path.straight(length=12).extrude(mhsq_xs)
        mh2.move((self.length, 4.25))

        via1_xs = gf.cross_section.cross_section(width=5, offset=0, layer=(44, 0))

        via11 = c << gf.path.straight(length=5).extrude(via1_xs)
        via11.move((-8.5, 4.25))

        via12 = c << gf.path.straight(length=5).extrude(via1_xs)
        via12.move((self.length + 3.5, 4.25))

        m2_xs = gf.cross_section.cross_section(width=40, offset=0, layer=(45, 0))

        m21 = c << gf.path.straight(length=15).extrude(m2_xs)
        m21.move((-13.5, 14.25))

        m22 = c << gf.path.straight(length=15).extrude(m2_xs)
        m22.move((self.length - 1.5, 14.25))

        c.add_port("o1", port=wg.ports["o1"])
        c.add_port("o2", port=wg.ports["o2"])

        return c

    def get_circuit_model(self):
        return None


if __name__ == "__main__":
    # Example usage

    c = heater_tin_oband({"length": 300})
    c.component.show()
    print(
        "Footprint Estimate: "
        + str(c.component.xsize)
        + "um x "
        + str(c.component.ysize)
        + "um"
    )

    """ recnet = sax.RecursiveNetlist.parse_obj(c.component.get_netlist_recursive())
    print('Required Models ==>', sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, c.model)
    pprint( _c(wl = 1.55) )
    # print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )
    wl = np.linspace(1.5, 1.6, 500)
    S41 = (_c(wl = wl)['o1','o4'])
    S31 = (_c(wl = wl)['o1','o3'])
    S42 = (_c(wl = wl)['o2','o4'])
    S32 = (_c(wl = wl)['o2','o3'])
    plt.plot(wl, np.abs(S41)**2)
    plt.plot(wl, np.abs(S31)**2)
    plt.plot(wl, np.abs(S42)**2)
    plt.plot(wl, np.abs(S32)**2)
    plt.show() """
