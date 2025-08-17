import gdsfactory as gf
from heater_doped_si_cband import heater_doped_si_cband


class dc_mzi_2x2_heater_doped_si_cband:
    """Name: MZI 2x2 - Thermo-optic
    Description: >
        This is a 2x2 Mach-Zehnder interferometer (MZI) with 50-50 directional couplers.
        Integrated in both arms of the MZI are doped heaters.
    ports: 2x2
    NodeLabels:
        - modulator
        - active
        - amplitude modulation (AM)
    aka: amplitude modulator, Mach-Zehnder interferometer, MZI
    Technology : Thermo-optic effect (TO)
    Design wavelength: 1450-1650 nm
    Optical Bandwidth: 200 nm
    Polarization: TE/TM
    Modulation bandwidth/Switching speed: 200 KHz
    Insertion loss: 2 dB
    Extinction ratio: 25 dB
    Drive voltage/power: 0.75 V
    Footprint Estimate: 493.6um x 98.4um.
    """

    def __init__(self, config=None):
        default_config = {
            "length": 320,  # in micrometers
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
        self.length = self.config["length"]
        return None

    def get_component(self):
        c = gf.Component()

        dc1 = c << gf.components.coupler(gap=0.3, length=9.8, dy=55, dx=36)
        dc2 = c << gf.components.coupler(gap=0.3, length=9.8, dy=55, dx=36)
        heater1 = c << heater_doped_si_cband(config={"length": self.length}).component
        heater2 = c << heater_doped_si_cband(config={"length": self.length}).component

        dc1.connect("o4", heater2.ports["o1"])
        dc2.connect("o1", heater2.ports["o2"])

        heater1.connect("o1", dc1.ports["o3"])
        heater1.connect("o2", dc2.ports["o2"])

        c.add_port("o1", port=dc1.ports["o1"])
        c.add_port("o2", port=dc1.ports["o2"])
        c.add_port("o3", port=dc2.ports["o3"])
        c.add_port("o4", port=dc2.ports["o4"])

        return c

    def get_circuit_model(self):
        """m1 = directional_coupler().model
        m2 = mzi_arm().model.
        """
        combined_dict = 0
        return combined_dict


if __name__ == "__main__":
    # Example usage

    c = dc_mzi_2x2_heater_doped_si_cband(config={"length": 320})
    print(c.component.ports)
    c.component.show()
    print(
        "Footprint Estimate: "
        + str(round(c.component.xsize, 2))
        + "um x "
        + str(round(c.component.ysize, 2))
        + "um"
    )

    """ print(c.component)
    print(c.config)
    print(c.model)
    quickplot(c.component)
    plt.show() """

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
