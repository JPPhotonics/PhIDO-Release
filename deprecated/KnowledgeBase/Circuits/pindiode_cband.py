import gdsfactory as gf

# from PhotonicsAI.KnowledgeBase.Components.directional_coupler import directional_coupler
# from PhotonicsAI.KnowledgeBase.Circuits.mzi_arm import mzi_arm


class pindiode_cband:
    """Name: PIN Diode
    Description: >
        This is a pin diode.
        The device is operated in forward bias which causes carrier injectio in the waveguide core.
    ports: 1x1
    NodeLabels:
        - modulator
        - active
        - amplitude modulation (AM)
    aka: amplitude modulator, PIN diode, Variable Optical Attenuator (VOA)
    Technology : Plasma Dispersion Effect
    Design wavelength: 1450-1650 nm
    Optical Bandwidth: 200 nm
    Polarization: TE/TM
    Modulation bandwidth/Switching speed: 1 GHz
    Insertion loss: 2-20 dB
    Extinction ratio: N/A
    Drive voltage/power: 2 mW
    Footprint Estimate: 510.0um x 41.4um.
    """

    def __init__(self, config=None):
        default_config = {
            "length": 500  # in micrometers
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

        slab_xs = gf.cross_section.cross_section(width=25.1, offset=0, layer=(3, 0))
        c << gf.path.straight(length=self.length).movex(5).extrude(slab_xs)

        slab_taper1 = c << gf.components.taper(
            length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
        )
        slab_taper1.movex(-5).rotate(180)

        slab_taper2 = c << gf.components.taper(
            length=5, width1=4.5, width2=0.45, port=None, layer=(3, 0)
        )
        slab_taper2.movex(self.length + 5)

        npp_xs = gf.Section(width=11, offset=6.55, layer=(24, 0))
        ppp_xs = gf.Section(width=11, offset=-6.55, layer=(25, 0))
        doping_xs = gf.CrossSection(sections=[npp_xs, ppp_xs])
        c << gf.path.straight(length=self.length - 2).movex(6).extrude(doping_xs)

        metal_top_xs = gf.Section(width=17, offset=-12.2, layer=(41, 0))
        metal_bottom_xs = gf.Section(width=17, offset=12.2, layer=(41, 0))
        metal_xs = gf.CrossSection(sections=[metal_top_xs, metal_bottom_xs])
        c << gf.path.straight(length=self.length).movex(5).extrude(metal_xs)

        via_top_xs = gf.Section(width=7.3, offset=-7.85, layer=(40, 0))
        via_bottom_xs = gf.Section(width=7.3, offset=7.85, layer=(40, 0))
        metal_xs = gf.CrossSection(sections=[via_top_xs, via_bottom_xs])
        c << gf.path.straight(length=self.length - 3).movex(6.5).extrude(metal_xs)

        c.add_port("o1", port=wg.ports["o1"])
        c.add_port("o2", port=wg.ports["o2"])
        return c

    def get_circuit_model(self):
        # m1 = directional_coupler().model
        # m2 = mzi_arm().model
        combined_dict = 0
        return combined_dict


if __name__ == "__main__":
    # Example usage

    c = pindiode_cband(config={"length": 500})
    """ print(c.component)
    print(c.config)
    print(c.model)
    quickplot(c.component) """
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
