"""This is a multimode interferometer with one input and two output ports.

---
Name: mmi1x2
Description: >
    This multimode interferometer has one input and two output ports.
    It functions as a beamsplitter or power splitter, dividing the input equally between the two outputs.
    Each output receives half of the input power, ensuring balanced splitting.
ports: 1x2
NodeLabels:
    - passive
    - 1x2
Bandwidth: 50 nm
Args:
    -width: input and output straight width. Defaults to cross_section width.
    -width_taper: interface between input straights and mmi region.
    -length_taper: into the mmi region.
    -length_mmi: in x direction.
    -width_mmi: in y direction.
    -gap_mmi:  gap between tapered wg.
"""

import gdsfactory as gf
import numpy as np
import sax
import yaml

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz, model_from_tidy3d

@gf.cell
def _mmi1x2(
    length_mmi: float = 12.8,
    width_mmi: float = 3.8,
    gap_mmi: float = 0.25,
    length_taper: float = 10.01,
    width_taper: float = 1.4,
) -> gf.Component:
    _args = locals()

    c = gf.Component()
    m = gf.components.mmi1x2(**_args)
    coupler_r = c << m
    c.add_port("o1", port=coupler_r.ports["o1"])
    c.add_port("o2", port=coupler_r.ports["o2"])
    c.add_port("o3", port=coupler_r.ports["o3"])
    c.flatten()
    return c


def get_model(model="tidy3d"):
    if model == "ana":
        return {"_mmi1x2": get_model_ana}
    if model == "fdtd":
        return {"_mmi1x2": get_model_fdtd}
    if model == "tidy3d":
        return {"_mmi1x2": get_model_tidy3d}


def get_model_tidy3d(wl=1.55):

    # Define the path
    yaml_path = 'build/modified_netlist.yaml'
    print("succ load st1")
    # Read and parse the YAML file
    with open(yaml_path, "r") as f:
        print("hera 101")
        modified_netlist = yaml.safe_load(f)
        print("succ load")
        print(modified_netlist)
    
    c = gf.read.from_yaml(modified_netlist)
    print("here")
    """ with open('build/modified_netlist.yaml', 'r') as file:
        modified_netlist = yaml.safe_load(file)
        print(modified_netlist)
    if "_mmi1x2" in modified_netlist.split():
        c = gf.read.from_yaml(modified_netlist)
    else:
        c = _mmi1x2() """
    
    """ except:
        print("here12")
        c = _mmi1x2() """
    
    model_data = model_from_tidy3d(c=c)
    return model_data(wl=wl)


def get_model_fdtd(wl=1.55):
    file_path = get_file_path(
        "FDTD/cband/mmi1x2/mmi1x2_length12p8_width3p8_gap0p25_taperwidth1p4.npz"
    )
    model_data = model_from_npz(file_path)
    return model_data(wl=wl)


def get_model_ana(wl=1.5, coupling: float = 0.5):
    """A simple coupler model."""
    # wg_factor = np.exp(1j * np.pi * 2.34 * 1 / wl)
    wg_factor = 1
    kappa = wg_factor * coupling**0.5
    tau = wg_factor * (1 - coupling) ** 0.5
    sdict = sax.reciprocal(
        {
            ("o1", "o2"): tau,
            ("o1", "o3"): 1j * kappa,
        }
    )
    return sdict


# class mmi1x2:

#     def __init__(self, config=None):
#         default_config = {'wl0': 1.55,
#                         #   'pol':'TE',
#                           'coupling': 0.5}
#         if config is None:
#             config = default_config
#         else:
#             config = {**default_config, **config}

#         self.config = config
#         self.component = None
#         self.model = None

#         _ = self.config_to_geometry()
#         self.component = self.get_component()
#         self.model = {'mmi1x2': self.get_model_ana}

#     def config_to_geometry(self):
#         """
#         Provides mapping from design config to geometric settings of gdsfacotory component.
#         """
#         self.wl0 = self.config['wl0']
#         # self.pol = config['pol']
#         self.coupling = self.config['coupling']
#         x_to_y = lambda x: 20*x + 2 # dummy mapping
#         self.length = x_to_y(self.coupling)
#         return None

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    #matplotlib.use("macosx")

    c = gf.Component()
    ref = c << _mmi1x2(width_mmi=10)
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])

    filepath = "./build/" + "/opt_sparams.npz"

    sp = np.load(filepath, allow_pickle=True)
    keys = list(sp.keys())

    xkey = "wavelengths"

    if xkey not in keys:
        try:
            sp = sp['arr_0'].tolist()
        except Exception as e:
            raise ValueError(f"{xkey!r} not in {keys}")
    wl = sp["wavelengths"]

    print(sp)

    # Example: Extract wavelength/frequency

    #wl = sp["wavelengths"]  # e.g., in nanometers or Hz

    # Loop through all S-parameters
    for key in sp:
        print(key)
        sparam = sp[key]
        magnitude = 20 * np.log10(np.abs(sparam))
        phase = np.angle(sparam, deg=True)  # in degrees

        """ # Label like "S(port1, port2)"
        port_mode0, port_mode1 = key.split(",")
        port0, _ = port_mode0.split("@")
        port1, _ = port_mode1.split("@")
        label = f"S({port0}, {port1})" """

        # Plot magnitude
        plt.figure(1)
        plt.plot(wl, magnitude, label=key)


    # Configure magnitude plot
    plt.figure(1)
    plt.title("S-Parameter Magnitude")
    plt.xlabel("Wavelength (or Frequency)")
    plt.ylabel("|S|")
    plt.grid(True)
    plt.legend()


    plt.savefig("sparameters_plot.png", dpi=300, bbox_inches='tight')
    plt.close()  # Ensure the figure is closed and no backend tries to show it

    # c.plot()

    """ print(c.get_netlist())
    print()
    # sys.exit()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model())
    print(_c(wl=1.55))
    # print( np.abs(_c(wl = 1.35)['o1','o2'])**2 )

    plt.figure()
    wl = np.linspace(1.4, 1.6, 128)
    S21 = _c(wl=wl)["o1", "o2"]
    S31 = _c(wl=wl)["o1", "o3"]
    plt.plot(wl, np.abs(S31) ** 2)
    plt.plot(wl, np.abs(S21) ** 2)
    # gsax.plot_model(get_model_f/dtd)
    plt.show() """
