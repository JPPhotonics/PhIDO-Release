import gdsfactory as gf
from heater_doped_si_cband import heater_doped_si_cband


class mzi_2x2_heater_doped_si_cband:
    """Name: MZI 2x2 with heater -  Doped silicon heater
    Description: >
        This is a 2x2 Mach-Zehnder interferometer (MZI). The MZI uses 2x2 MMIs.
        Integrated in both arms of the MZI are doped silicon waveguides.
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
    Extinction ratio: 20 dB
    Drive voltage/power: 0.75 V.
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
        mmi2x2 = gf.components.mmi2x2(
            width_mmi=5.52,
            length_mmi=36.2,
            gap_mmi=0.27,
            width_taper=1.3,
            length_taper=15,
        )
        heater = heater_doped_si_cband(config={"length": self.length}).component

        ref = c << gf.components.mzi2x2_2x2(
            delta_length=0,
            length_y=2.5,
            length_x=self.length,
            straight_x_top=heater,
            straight_x_bot=heater,
            splitter=mmi2x2,
            combiner=mmi2x2,
        )

        c.add_port("o1", port=ref.ports["o1"])
        c.add_port("o2", port=ref.ports["o2"])
        c.add_port("o3", port=ref.ports["o3"])
        c.add_port("o4", port=ref.ports["o4"])

        return c

    def get_circuit_model(self):
        """m1 = directional_coupler().model
        m2 = mzi_arm().model.
        """
        combined_dict = 0
        return combined_dict


if __name__ == "__main__":
    # Example usage

    c = mzi_2x2_heater_doped_si_cband(config={"length": 100.4})
    c.component.show()
