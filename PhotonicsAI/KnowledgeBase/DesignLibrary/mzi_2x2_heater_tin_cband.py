"""This is a 2x2 Mach-Zehnder interferometer (MZI).

---
Name: MZI 2x2 - Thermo-optic
Description: >
    This is a 2x2 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are TiN Heaters.
ports: 2x2
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
aka: amplitude modulator, Mach-Zehnder interferometer, MZI
Technology: Thermo-optic effect (TO)
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 200 KHz
Insertion loss: 2 dB
Extinction ratio: 25 dB
Drive voltage/power: 0.75 V
Footprint Estimate: 516.42um x 295.07um

Args:
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
    _mmi2x2,
    bend_euler,
    heater_tin_cband,
    straight,
)


@gf.cell
def mzi_2x2_heater_tin_cband(
    delta_length: float = 40, length: float = 320
) -> gf.Component:
    """The component."""
    c = gf.Component()

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")
    mmi2x2 = _mmi2x2._mmi2x2()
    ref = c << gf.components.mzi2x2_2x2(
        delta_length=delta_length,
        length_y=2.5,
        length_x=length,
        straight_x_top=heater_tin_cband.heater_tin_cband,
        straight_x_bot=heater_tin_cband.heater_tin_cband,
        mirror_bot=True,
        splitter=mmi2x2,
        combiner=mmi2x2,
        cross_section=xs_1550,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])
    c.add_port("o4", port=ref.ports["o4"])

    # this is another version:
    # r1 = c << _mmi2x2._mmi2x2()
    # r2 = c << bend_euler.bend_euler()
    # r3 = c << straight.straight()
    # r4 = c << bend_euler.bend_euler()
    # r5 = c << heater_tin_cband.heater_tin_cband()
    # r6 = c << bend_euler.bend_euler()
    # r7 = c << straight.straight()
    # r8 = c << bend_euler.bend_euler()
    # r9 = c << _mmi2x2._mmi2x2()
    # r10 = c << bend_euler.bend_euler()
    # r11 = c << straight.straight()
    # r12 = c << bend_euler.bend_euler()
    # r13 = c << heater_tin_cband.heater_tin_cband()
    # r14 = c << bend_euler.bend_euler()
    # r15 = c << straight.straight()
    # r16 = c << bend_euler.bend_euler()

    # r2.connect("o1", r1.ports["o3"])
    # r3.connect("o1", r2.ports["o2"])
    # r4.connect("o2", r3.ports["o2"])
    # r5.connect("o1", r4.ports["o1"])
    # r6.connect("o2", r5.ports["o2"])
    # r7.connect("o1", r6.ports["o1"])
    # r8.connect("o1", r7.ports["o2"])
    # r9.connect("o2", r8.ports["o2"])
    # r10.connect("o2", r1.ports["o4"])
    # r11.connect("o1", r10.ports["o1"])
    # r12.connect("o1", r11.ports["o2"])
    # r13.connect("o2", r12.ports["o2"])
    # r14.connect("o1", r13.ports["o1"])
    # r15.connect("o1", r14.ports["o2"])
    # r16.connect("o2", r15.ports["o2"])

    # r9.connect("o2", r16.ports["o1"])

    # c.add_port("o1", port=r1.ports["o1"])
    # c.add_port("o2", port=r1.ports["o2"])
    # c.add_port("o3", port=r9.ports["o3"])
    # c.add_port("o4", port=r9.ports["o4"])

    return c


def get_model(model="tidy3d"):
    """The model."""
    m1 = _mmi2x2.get_model(model=model)
    m2 = straight.get_model(model=model)
    m3 = bend_euler.get_model(model=model)
    m4 = heater_tin_cband.get_model(model=model)
    combined_dict = m1 | m2 | m3 | m4
    return combined_dict


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import sax

    matplotlib.use("macosx")

    c = gf.Component()
    ref = c << mzi_2x2_heater_tin_cband()
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])
    c.add_port("o4", port=ref.ports["o4"])

    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    print("Required Models ==>", sax.get_required_circuit_models(recnet))

    # print(recnet)
    c.plot()
    # plt.show()

    _c, info = sax.circuit(recnet, get_model())
    # print( _c(wl = 1.55) )

    plt.figure()
    wl = np.linspace(1.1, 1.7, 200)
    S = _c(wl=wl)["o1", "o3"]
    SS = _c(wl=wl)["o1", "o4"]
    plt.plot(wl, np.abs(S) ** 2)
    plt.plot(wl, np.abs(SS) ** 2)
    plt.show()
