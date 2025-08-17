from __future__ import annotations
from pydantic import BaseModel
from pydantic import NonNegativeFloat

from gdsfactory.technology import LayerStack
from gdsfactory.generic_tech import LAYER_STACK
from gplugins.tidy3d.types import (
    Tidy3DMedium,
)

import tidy3d as td
from tidy3d.components.types import Symmetry

material_name_to_medium = {
    "si": td.Medium(name="Si", permittivity=3.47**2),
    "sio2": td.Medium(name="SiO2", permittivity=1.47**2),
    "sin": td.Medium(name="SiN", permittivity=2.0**2),
}

class SimulationSettingsTiny3DFdtd(BaseModel):

    # PDK
    layer_stack: LayerStack = LAYER_STACK
    material_mapping: dict[str, Tidy3DMedium] = material_name_to_medium

    # Adaptive Meshing
    min_steps_per_wvl: int = 10

    # 2D vs 3D
    sim_size_z: float = 1.0
    center_z: float | str | None = None
    
    # Source Settings
    wavelength: float = 1.5
    bandwidth: float = 0.1
    n_eff: float = 1

    # Simulation Dimentions
    extend_ports: float = 0
    port_offset: float = 3*wavelength/n_eff 
    pad_xy_inner: NonNegativeFloat = 3*wavelength/n_eff 
    pad_xy_outer: NonNegativeFloat = 3*wavelength/n_eff 
    pad_z: float = 3*wavelength/n_eff

    # Scale port size for the simulation
    port_size_mult: float | tuple[float, float] = (4, 6)
    
    # Number of Modes for Mode Solver
    num_modes: int = 1

    # Simulation Parameters
    grid_spec: td.GridSpec = td.GridSpec.auto(wavelength=wavelength/n_eff, min_steps_per_wvl=min_steps_per_wvl)
    boundary_spec: td.BoundarySpec = td.BoundarySpec.all_sides(boundary=td.PML()) 
    
    symmetry: tuple[Symmetry, Symmetry, Symmetry] = (0, 0, 0) 
    run_time: float = 1e-11
    verbose: bool = True
    shutoff: float = 1e-5 
    run_only: tuple[tuple[str, int], ...] | None = None


SIMULATION_SETTINGS_LUMERICAL_TINY3D_DEFAULT = SimulationSettingsTiny3DFdtd()