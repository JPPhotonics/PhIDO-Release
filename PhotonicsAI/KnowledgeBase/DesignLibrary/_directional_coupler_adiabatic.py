"""This is an adiabatic directional coupler with two input and two output ports.

---
Name: directional_coupler
Description: >
    This is an adiabatic directional coupler with two input and two output ports.
    Can be used for 50:50 power splitting. The device is optimized for 1480 nm.
ports: 2x2
NodeLabels:
    - passive
Bandwidth: 50 nm
"""

import gdsfactory as gf
import numpy as np
import sax

from PhotonicsAI.Photon.utils import get_file_path, model_from_npz

# from PhotonicsAI.Photon.utils import validate_cell_settings


# {'cross_section': 'strip', 'dx': 10, 'dy': 4, 'gap': 0.236, 'length': 20}


@gf.cell
def _directional_coupler_adiabatic() -> gf.Component:
    _args = locals()

    c = gf.Component()
    coupler = gf.components.coupler_adiabatic()
    coupler_r = c << coupler
    c.add_port("o1", port=coupler_r.ports["o1"])
    c.add_port("o2", port=coupler_r.ports["o2"])
    c.add_port("o3", port=coupler_r.ports["o3"])
    c.add_port("o4", port=coupler_r.ports["o4"])
    c.flatten()
    return c


def get_model(model: str = "fdtd") -> dict:
    if model == "ana":
        return {"_directional_coupler_adiabatic": get_model_ana}
    if model == "fdtd":
        return {"_directional_coupler_adiabatic": get_model_fdtd}


def get_model_fdtd(wl=1.55):
    file_path = get_file_path("FDTD/cband/coupler/coupler_adiabatic.npz")
    model_data = model_from_npz(file_path)
    return model_data(wl=wl)


def get_model_ana(wl=1.5, length=12):
    """A simple coupler model."""
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = _directional_coupler_adiabatic()

    print(c.get_netlist())
    print()

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    _c, info = sax.circuit(recnet, get_model())
    print(_c(wl=1.55))
    print(np.abs(_c(wl=1.35)["o1", "o4"]) ** 2)

    c.plot()
    plt.show()
