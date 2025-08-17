import gdsfactory as gf

# from PhotonicsAI.KnowledgeBase.Components.directional_coupler import directional_coupler
# from PhotonicsAI.KnowledgeBase.Circuits.mzi_arm import mzi_arm


class heater_doped_si_cband:
    """Name: Doped silicon heater
    Description: This is a heater formed using doped silicon waveguides (n++, n).
    ports: 1x1
    NodeLabels:
        - modulator
        - active
        - phase modulation (AM)
    aka: phase modulator
    Technology : Thermo-optic effect (TO)
    Design wavelength: 1450-1650 nm
    Optical Bandwidth: 200 nm
    Polarization: TE/TM
    Modulation bandwidth/Switching speed: 200 KHz
    Insertion loss: 2 dB
    Extinction ratio: N/A
    Drive voltage/power: 0.75 V
    Footprint Estimate: 110.4um x 43.4um.
    """

    def __init__(self, config=None):
        default_config = {
            "length": 100.4,  # in micrometers
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

        wg_xs = gf.cross_section.cross_section(width=0.5, offset=0, layer=(1, 0))
        wg = c << gf.path.straight(length=self.length + 10).extrude(wg_xs)

        slab_xs = gf.cross_section.cross_section(width=24.5, offset=0, layer=(3, 0))
        c << gf.path.straight(length=self.length).movex(5).extrude(slab_xs)

        slab_taper1 = c << gf.components.taper(
            length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
        )
        slab_taper1.movex(-5).rotate(180)

        slab_taper2 = c << gf.components.taper(
            length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
        )
        slab_taper2.movex(self.length + 5)

        npp_top_xs = gf.Section(width=10.45, offset=-6.575, layer=(24, 0))
        npp_bottom_xs = gf.Section(width=10.45, offset=6.575, layer=(24, 0))
        n_xs = gf.Section(width=23.6, offset=0, layer=(20, 0))
        doping_xs = gf.CrossSection(sections=[npp_top_xs, n_xs, npp_bottom_xs])
        c << gf.path.straight(length=self.length - 0.4).movex(5.2).extrude(doping_xs)

        metal_top_xs = gf.Section(width=20.5, offset=-11.45, layer=(41, 0))
        metal_bottom_xs = gf.Section(width=20.5, offset=11.45, layer=(41, 0))
        metal_xs = gf.CrossSection(sections=[metal_top_xs, metal_bottom_xs])
        c << gf.path.straight(length=self.length + 0.4).movex(4.8).extrude(metal_xs)

        via_top_xs = gf.Section(width=7.1, offset=-7.75, layer=(40, 0))
        via_bottom_xs = gf.Section(width=7.1, offset=7.75, layer=(40, 0))
        metal_xs = gf.CrossSection(sections=[via_top_xs, via_bottom_xs])
        c << gf.path.straight(length=self.length - 1).movex(5.5).extrude(metal_xs)

        c.add_port("o1", port=wg.ports["o1"])
        c.add_port("o2", port=wg.ports["o2"])

        return c

    def get_circuit_model(self):
        """m1 = directional_coupler().model
        m2 = mzi_arm().model.
        """
        combined_dict = 0
        return combined_dict


if __name__ == "__main__":
    # Example usage

    c = heater_doped_si_cband(config={"length": 100.4})
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
