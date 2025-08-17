"""
Name: Titanium nitride heater with undercut and multipass
Description: This is a TiN heater with substrate undercut. The waveguide passes through the heater three times.
ports: 1x1
NodeLabels:
    - modulator
    - active
    - phase modulation (AM)
aka: phase modulator, heater, thermo-optic phase shifter, low-power heater
Technology : Thermo-optic effect (TO)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 6 KHz
Insertion loss: 1 dB
Extinction ratio: N/A
Drive voltage/power: 3 mW
Inputs: 1
Outputs: 1
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import _directional_coupler, mzi_arm


@gf.cell
def heater_tin_undercut_multipass(settings: dict = {}) -> gf.Component:
    dy = 80
    coupling1 = 0.5
    coupling2 = 0.5
    delta_length = 30

    c = gf.Component()

    c1 = c << _directional_coupler._directional_coupler(
        {"dy": dy, "coupling": coupling1}
    )
    a1 = c << mzi_arm.mzi_arm({"length": delta_length + 1})
    a2 = c << mzi_arm.mzi_arm({"length": 1})
    a2.rotate(180)
    c2 = c << _directional_coupler._directional_coupler(
        {"dy": dy, "coupling": coupling2}
    )

    a1.connect("o1", c1.ports["o3"])
    a2.connect("o2", c1.ports["o4"])
    c2.connect("o1", a2.ports["o1"])
    c2.connect("o2", a1.ports["o2"])

    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c1.ports["o2"])
    c.add_port("o3", port=c2.ports["o3"])
    c.add_port("o4", port=c2.ports["o4"])
    return c


def get_model():
    m1 = _directional_coupler.get_model()
    m2 = mzi_arm.mzi_arm.get_model()
    combined_dict = m1 | m2
    return combined_dict


# class heater_tin_undercut_multipass:

#     def __init__(self, config=None):
#         default_config = {'fsr': 0.0010, # in micrometers
#                           'dy': 80,
#                           'coupling1': 0.5,
#                           'coupling2': 0.5}
#         if config is None:
#             config = default_config
#         else:
#             config = {**default_config, **config}

#         self.config = config
#         self.component = None
#         self.model = None

#         _ = self.config_to_geometry()
#         self.component = self.get_component()
#         self.model = self.get_circuit_model() # TODO

#     def config_to_geometry(self):
#         # self.delta_length = self.config['delta_length']
#         self.dy = self.config['dy']

#         self.fsr = self.config['fsr']
#         self.delta_length = int(0.25/self.fsr)

#         self.coupling1 = self.config['coupling1']
#         self.coupling2 = self.config['coupling2']
#         return None

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = heater_tin_undercut_multipass(
        {"delta_length": 50, "coupling1": 0.1, "coupling2": 0.1}
    )

    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")
