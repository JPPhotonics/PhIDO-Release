"""
Name: bezier_curve
Description: |
    This is a bend (or generally arc shaped) waveguide.
    The function maps a radii and and an angle to a 4 control points that implements the Bezier Curve.
ports: 1x1
NodeLabels:
    - passive
    - 1x1
Design wavelength: 1450-1650 nm
Args:
    cp1:
        description:
        optimizable: true
        opt_range:
            - 0
            - 5.25
    cp2:
        description:
        optimizable: true
        opt_range:
            - 0
            - 5.25
    cp3:
        description:
        optimizable: true
        opt_range:
            - 0
            - 5.25
    cp4:
        description:
        optimizable: true
        opt_range:
            - 0
            - 5.25
Specs:
    radius: 5
    angle: 90
    wg_width: 0.5
"""

import math

import gdsfactory as gf
import numpy as np


@gf.cell
def _var_bezier_curve(
    radius: float = 5,
    angle: float = 90,
    wg_width: float = 0.5,
    cp1: float = 0,
    cp2: float = 0,
    cp3: float = 0,
    cp4: float = 0,
) -> gf.Component:
    control_points_outter = (
        (-wg_width / 2, 0),
        (-wg_width / 2, cp1),
        (-wg_width / 2 + cp2, radius + wg_width),
        (radius, radius + wg_width / 2),
    )

    control_points_inner = (
        (wg_width / 2, 0),
        (wg_width / 2, cp3),
        (wg_width / 2 + cp4, radius - wg_width / 2),
        (radius, radius - wg_width / 2),
    )

    control_points_center = (
        (
            (control_points_outter[0][0] + control_points_inner[0][0]) / 2,
            (control_points_outter[0][1] + control_points_inner[0][1]) / 2,
        ),
        (
            (control_points_outter[1][0] + control_points_inner[1][0]) / 2,
            (control_points_outter[1][1] + control_points_inner[1][1]) / 2,
        ),
        (
            (control_points_outter[2][0] + control_points_inner[2][0]) / 2,
            (control_points_outter[2][1] + control_points_inner[2][1]) / 2,
        ),
        (
            (control_points_outter[3][0] + control_points_inner[3][0]) / 2,
            (control_points_outter[3][1] + control_points_inner[3][1]) / 2,
        ),
    )

    def bezier_curve_width(points):
        xs = []
        ys = []
        for t in points:
            xs.append(
                np.power((1 - t), 3) * control_points_outter[0][0]
                + 3 * np.power((1 - t), 2) * t * control_points_outter[1][0]
                + 3 * (1 - t) * np.power(t, 2) * control_points_outter[2][0]
                + np.power(t, 3) * control_points_outter[3][0]
            )
            ys.append(
                np.power((1 - t), 3) * control_points_outter[0][1]
                + 3 * np.power((1 - t), 2) * t * control_points_outter[1][1]
                + 3 * (1 - t) * np.power(t, 2) * control_points_outter[2][1]
                + np.power(t, 3) * control_points_outter[3][1]
            )

        path_points_outter = np.column_stack([xs, ys])

        xs = []
        ys = []
        for t in points:
            xs.append(
                np.power((1 - t), 3) * control_points_inner[0][0]
                + 3 * np.power((1 - t), 2) * t * control_points_inner[1][0]
                + 3 * (1 - t) * np.power(t, 2) * control_points_inner[2][0]
                + np.power(t, 3) * control_points_inner[3][0]
            )
            ys.append(
                np.power((1 - t), 3) * control_points_inner[0][1]
                + 3 * np.power((1 - t), 2) * t * control_points_inner[1][1]
                + 3 * (1 - t) * np.power(t, 2) * control_points_inner[2][1]
                + np.power(t, 3) * control_points_inner[3][1]
            )

        path_points_inner = np.column_stack([xs, ys])

        temp = path_points_outter - path_points_inner

        return np.linalg.norm(temp, axis=1)

    P = gf.path.arc(radius=radius, angle=angle, start_angle=0, npoints=201)

    s1 = gf.Section(
        width=0,
        width_function=bezier_curve_width,
        offset=0,
        layer=(1, 0),
        port_names=("in", "out"),
    )
    X = gf.CrossSection(sections=[s1])

    c = gf.Component()

    c = gf.path.extrude(P, cross_section=X)

    return c


# def get_model_ana(radius):
def get_model(model="fdtd", radius=5):
    # Model Parameters
    p = -0.01948
    q = 0.927
    r = 0.9998

    loss = p * math.exp(q * (-radius)) + r

    return loss


if __name__ == "__main__":
    c = _var_bezier_curve()
    c.show()
