"""
Name: MZI 1x2 - PIN Diode
Description: >
    This is a 1x2 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are PIN Diode.
ports: 1x2
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
aka: amplitude modulator, Mach-Zehnder interferometer, MZI
Technology : Plasma Dispersion effect (TO)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 30 KHz
Insertion loss: 2-20 dB
Extinction ratio: 25 dB
Drive voltage/power: 2 mW
Footprint Estimate: 428um x 512um
Args:
    -length: straight length heater (um)
    -delta_length: path length difference (um). bottom arm vertical extra length.

args:
  length:
    description: straight length heater (um)
  delta_length:
    description: path length difference (um). bottom arm vertical extra length.
    optimizable: true
    opt_range:
      - 0
      - 100

specs:
  lambda_0: 1.55
  fsr: 0.025
  transmission_fn: target_transfer

"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import (
    _mmi1x2,
    _mmi2x2,
    bend_euler,
    pindiode_cband,
    straight,
)


@gf.cell
def mzi_1x2_pindiode_cband(delta_length=100, length: float = 320):
    c = gf.Component()

    mmi1x2 = _mmi1x2._mmi1x2()
    mmi2x2 = _mmi2x2._mmi2x2()

    ref = c << gf.components.mzi1x2_2x2(
        delta_length=delta_length,
        length_y=129.175,
        length_x=length,
        straight_x_top=pindiode_cband.pindiode_cband,
        straight_x_bot=pindiode_cband.pindiode_cband,
        splitter=mmi1x2,
        combiner=mmi2x2,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])

    return c


def get_model(model="fdtd"):
    m1 = _mmi1x2.get_model(model=model)
    m2 = straight.get_model(model=model)
    m3 = bend_euler.get_model(model=model)
    m4 = _mmi2x2.get_model(model=model)
    m5 = pindiode_cband.get_model(model=model)
    combined_dict = m1 | m2 | m3 | m4 | m5
    return combined_dict


if __name__ == "__main__":
    import sax

    c = gf.Component()
    ref = c << mzi_1x2_pindiode_cband()
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])

    # c.show()
    print("Footprint Estimate: " + str(c.xsize) + "um x " + str(c.ysize) + "um")

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))
