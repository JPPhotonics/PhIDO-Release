"""
Name: MZI 2x2 - PIN Diode
Description: >
    This is a 2x2 Mach-Zehnder interferometer (MZI).
    Integrated in both arms of the MZI are PIN Diode. The MZI uses 2x2 MMIs.
ports: 2x2
NodeLabels:
    - modulator
    - active
    - amplitude modulation (AM)
aka: amplitude modulator, Mach-Zehnder interferometer, MZI
Technology : Plasma-Dispersion effect
Design wavelength: 1450-1650 nm
Optical Bandwidth: 200 nm
Polarization: TE/TM
Modulation bandwidth/Switching speed: 1 GHz
Insertion loss: 2-20 dB
Extinction ratio: 25 dB
Drive voltage/power: 2 mW
Footprint Estimate: 502.42um x 462.2um
Args:
    -length: straight length heater (um)
    -delta_length: path length difference (um). bottom arm vertical extra length.
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import (
    _mmi2x2,
    bend_euler,
    pindiode_cband,
    straight,
)


@gf.cell
def mzi_2x2_pindiode_cband(
    delta_length: float = 40, length: float = 320
) -> gf.Component:
    c = gf.Component()

    xs_1550 = gf.cross_section.cross_section(width=0.5, offset=0, layer="WG")

    mmi2x2 = _mmi2x2._mmi2x2()

    ref = c << gf.components.mzi2x2_2x2(
        delta_length=delta_length,
        length_y=129.215,
        length_x=length,
        straight_x_top=pindiode_cband.pindiode_cband,
        straight_x_bot=pindiode_cband.pindiode_cband,
        splitter=mmi2x2,
        combiner=mmi2x2,
        cross_section=xs_1550,
    )

    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])
    c.add_port("o4", port=ref.ports["o4"])
    return c


def get_model(model="fdtd"):
    m1 = _mmi2x2.get_model(model=model)
    m2 = straight.get_model(model=model)
    m3 = bend_euler.get_model(model=model)
    combined_dict = m1 | m2 | m3
    return combined_dict


if __name__ == "__main__":
    import matplotlib
    import sax

    matplotlib.use("macosx")

    c = gf.Component()
    ref = c << mzi_2x2_pindiode_cband()
    c.add_port("o1", port=ref.ports["o1"])
    c.add_port("o2", port=ref.ports["o2"])
    c.add_port("o3", port=ref.ports["o3"])
    c.add_port("o4", port=ref.ports["o4"])

    # pprint(c.get_netlist())
    # c.plot()
    # plt.show()
    # c.show()
    # print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")

    recnet = sax.RecursiveNetlist.model_validate(c.get_netlist(recursive=True))
    # print('Required Models ==>', sax.get_required_circuit_models(recnet))
