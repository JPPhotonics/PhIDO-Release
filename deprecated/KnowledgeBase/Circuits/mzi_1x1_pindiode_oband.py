import gdsfactory as gf
from pindiode_oband import pindiode_oband


class mzi_1x1_pindiode_oband:
    """Name: MZI 1x1 - PIN Diode
    Description: >
        This is a 1x1 Mach-Zehnder interferometer (MZI).
        Integrated in both arms of the MZI are PIN Diode.
    ports: 1x1
    NodeLabels:
        - modulator
        - active
        - amplitude modulation (AM)
    aka: amplitude modulator, Mach-Zehnder interferometer, MZI
    Technology : Thermo-optic effect (TO)
    Design wavelength: 1260-1360 nm
    Optical Bandwidth: 200 nm
    Polarization: TE/TM
    Modulation bandwidth/Switching speed: 1 GHz
    Insertion loss: 2-20 dB
    Extinction ratio: 25 dB
    Drive voltage/power: 2 mW
    Footprint Estimate: 456.02um x 88.12um.
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

        mmi1x2 = gf.components.mmi1x2(
            width_mmi=3.6,
            length_mmi=13,
            gap_mmi=0.6,
            width_taper=1.2,
            length_taper=15,
            width=0.41,
        )
        pin = pindiode_oband(config={"length": self.length}).component

        xs_1310 = gf.cross_section.cross_section(width=0.41, offset=0, layer="WG")
        ref = c << gf.components.mzi(
            delta_length=0,
            length_y=2.5,
            length_x=self.length,
            straight_x_top=pin,
            straight_x_bot=pin,
            splitter=mmi1x2,
            combiner=mmi1x2,
            cross_section=xs_1310,
        )

        c.add_port("o1", port=ref.ports["o1"])
        c.add_port("o2", port=ref.ports["o2"])

        return c

    def get_circuit_model(self):
        """m1 = directional_coupler().model
        m2 = mzi_arm().model.
        """
        combined_dict = 0
        return combined_dict


if __name__ == "__main__":
    # Example usage

    c = mzi_1x1_pindiode_oband()
    print(c.component.ports)
    c.component.show()
    print(
        "Footprint Estimate: "
        + str(c.component.xsize)
        + "um x "
        + str(c.component.ysize)
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
