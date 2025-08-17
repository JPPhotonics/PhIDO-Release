"""
Name: Wavelength Division Multiplexers (WDMs) based on Multi-Ring Resonators (MRR) with heaters
paper: https://opg.optica.org/ol/fulltext.cfm?uri=ol-39-21-6304&id=303411
Description: >
    These devices leverage properties of microring resonators (MRRs) to achieve efficient
    wavelength filtering and channel separation, which are essential for coarse WDM (CWDM) applications.
    The MRR-based filters are designed to provide a box-like spectral response,
    making them robust against wavelength shifts caused by environmental variations.
    This robustness is achieved through the cascading of multiple microrings with
    optimal coupling ratios, ensuring low excess loss and minimal crosstalk.
ports: 1x2
NodeLabels:
    - WDM
    - CWDM
    - active
aka: wavelength filter
Technology: ring resonators
N of channels: 2
channel spacing: 2 THz
Design wavelength: 1280, 1340 nm
Optical Bandwidth: 2 nm
Polarization: TE
Insertion loss: 2 dB
Extinction ratio: 30 dB
"""

import gdsfactory as gf

from PhotonicsAI.KnowledgeBase.DesignLibrary import _directional_coupler, mzi_arm


@gf.cell
def wdm_mrr1(settings: dict = {}):
    dy = 80
    coupling1 = 0.5
    coupling2 = 0.5
    delta_length = 50

    c = gf.Component()

    c1 = c << _directional_coupler._directional_coupler(
        {"dy": dy, "coupling": coupling1}
    )
    a1 = c << mzi_arm.mzi_arm({"length": delta_length + 1})
    a2 = c << mzi_arm.mzi_arm({"length": 1})
    a2.drotate(180)
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


def get_model(model="ana"):
    m1 = _directional_coupler.get_model()
    m2 = mzi_arm.get_model()
    combined_dict = m1 | m2
    return combined_dict


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    c = wdm_mrr1()
    c.plot()
    plt.show()
    print("Footprint Estimate: " + str(c.dxsize) + "um x " + str(c.dysize) + "um")
