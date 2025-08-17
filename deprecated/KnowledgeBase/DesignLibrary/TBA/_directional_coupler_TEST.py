"""
Name: directional_coupler
Description: >
    A directional coupler with two input and two output ports.
    Can be used for power splitting.
ports: 2x2
NodeLabels:
    - passive
Bandwidth: 50 nm
"""

import gdsfactory as gf
import gplugins.sax as gsax
import numpy as np
import sax

# from PhotonicsAI.Photon.utils import validate_cell_settings


# args = {
#     'functional': {
#         'coupling': {'default': 0.5, 'range': (0, 1)}
#     },
#     'geometrical': {
#         'gap':      {'default': 0.236, 'range': (0.1, 5.0)},
#         'length':   {'default': 20.0, 'range': (0.1, 1000.0)},
#         'dy':       {'default': 4.0, 'range': (1, 1000)},
#         'dx':       {'default': 10.0, 'range': (0.1, 1000)},
#     }
# }

# {'cross_section': 'strip', 'dx': 10, 'dy': 4, 'gap': 0.236, 'length': 20}


@gf.cell
def _directional_coupler(settings: dict = {}) -> gf.Component:
    c = gf.Component()
    coupler = gf.components.coupler()
    coupler_r = c << coupler
    c.add_port("o1", port=coupler_r.ports["o1"])
    c.add_port("o2", port=coupler_r.ports["o2"])
    c.add_port("o3", port=coupler_r.ports["o3"])
    c.add_port("o4", port=coupler_r.ports["o4"])

    # geometrical_params = get_params(settings)

    return c


from ..mzi_arm import mzi_arm


@gf.cell
def mzi1(settings: dict = {}) -> gf.Component:
    c = gf.Component()

    c1 = c << _directional_coupler()
    a1 = c << mzi_arm.mzi_arm()
    a2 = c << mzi_arm.mzi_arm()
    a2.drotate(180)
    c2 = c << _directional_coupler()

    a1.connect("o1", c1.ports["o3"])
    a2.connect("o2", c1.ports["o4"])
    c2.connect("o1", a2.ports["o1"])
    c2.connect("o2", a1.ports["o2"])

    c.add_port("o1", port=c1.ports["o1"])
    c.add_port("o2", port=c1.ports["o2"])
    c.add_port("o3", port=c2.ports["o3"])
    c.add_port("o4", port=c2.ports["o4"])

    # params = get_params(settings)

    return c


def get_params(settings: dict = {}) -> dict:
    """
    Generates the output configuration based on the settings.

    Parameters:
    settings (dict): A dictionary containing settings.

    Returns:
    dict: A dictionary containing the mapped geometrical parameters and direct output parameters.
    """

    validated_settings = validate_cell_settings(settings, args)

    def coupling_mapper(coupling):
        length = 2 + 10 * coupling
        return length

    output_params = {}

    # handle all functional parameters first
    # output_params['length'] = coupling_mapper(validated_settings['functional']['coupling'])

    # Add remaining geometrical parameters
    for arg in validated_settings["geometrical"]:
        if arg not in output_params:
            output_params[arg] = validated_settings["geometrical"][arg]

    return output_params


def get_model(model: str = "ana") -> dict:
    if model == "ana":
        return {"coupler": get_model_ana}
    if model == "fdtd":
        return {"coupler": get_model_fdtd_test}


def get_model_fdtd_test(wl=1.55):
    model_data = gsax.read.model_from_npz(
        "../FDTD-test/coupler/coupler_2e5eb039_1ce252ac.npz"
    )
    return model_data(wl=wl)


def get_model_ana(wl=1.5, length=12):
    """a simple coupler model"""
    # wg_factor = np.exp(1j * np.pi * 2.34 * 1 / wl)
    wg_factor = 1
    coupling = (np.sin(np.pi * length / 6) + 1) / 2
    # print('====COUPLING===>', coupling, length)
    kappa = wg_factor * coupling**0.5
    tau = wg_factor * (1 - coupling) ** 0.5
    sdict = sax.reciprocal(
        {
            ("o1", "o3"): tau,
            ("o1", "o4"): 1j * kappa,
            ("o2", "o3"): 1j * kappa,
            ("o2", "o4"): tau,
        }
    )
    return sdict


# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import sys

#     # c = get_component({'coupling':0.8, 'dy': 10})
#     # c = get_component({'length':1, 'dy': 4})
#     c = get_component()

#     print(c.get_netlist())
#     print()

#     recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
#     print('Required Models ==>', sax.get_required_circuit_models(recnet))

#     _c, info = sax.circuit(recnet, get_model())
#     print( _c(wl = 1.55) )
#     print( np.abs(_c(wl = 1.35)['o1','o4'])**2 )

#     c.plot()
#     plt.show()


class NetlistParser:
    def __init__(self, netlist):
        self.netlist = netlist
        self.settings_parameters = self.parse_settings()

    def parse_settings(self):
        parameters = []
        for component in self.netlist.values():
            for instance_name, instance in component.get("instances", {}).items():
                settings = instance.get("settings", {})
                parameters.extend(settings.keys())
                self.create_setter(instance_name, settings)
        return list(
            set(parameters)
        )  # Remove duplicates by converting to a set and back to a list

    def create_setter(self, instance_name, settings):
        # Dynamically create setter methods for each setting parameter
        for param in settings:
            method_name = f"{instance_name}_{param}"
            setattr(
                self, method_name, self.create_setter_function(instance_name, param)
            )

    def create_setter_function(self, instance_name, param):
        def setter_function(value):
            for component in self.netlist.values():
                if instance_name in component.get("instances", {}):
                    component["instances"][instance_name]["settings"][param] = value

        return setter_function

    def get_netlist(self):
        return self.netlist


if __name__ == "__main__":
    from pprint import pprint

    # c = get_component({'coupling':0.8, 'dy': 10})
    # c = get_component({'length':1, 'dy': 4})
    # c = mzi1()
    c = _directional_coupler()

    recnet = c.get_netlist(recursive=True)
    pprint(recnet)
    parser = NetlistParser(recnet)
    pprint(parser.settings_parameters)
    # _p = parser.extract_settings()
    # params = list(_p.values())[0]
    # pprint( params )

    # parser._directional_coupler_S.instances.coupler_G0p236_L20_D4_D_e8f9c7a8_10000_368.length = 30

    # parser.coupler_G0p236_L20_D4_D_e8f9c7a8_10000_368.length = 30
    # pprint(recnet)
    # print(parser._bend_euler_S.instances.bend_euler_R10_A90_P0p5_2f1f5c6d_5125_4875.radius)
    # parser._bend_euler_S.instances.bend_euler_R10_A90_P0p5_2f1f5c6d_5125_4875.radius = 20
    # parser.update_netlist()
    # print(parser._bend_euler_S.instances.bend_euler_R10_A90_P0p5_2f1f5c6d_5125_4875.list_settings())

    # pprint(parser.instances)
    # pprint(  )
    # recnet = c.get_netlist()
    # pprint(recnet)
    # c.plot()
    # plt.show()
