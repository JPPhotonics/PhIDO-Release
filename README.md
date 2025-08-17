# PhIDO - Photonic Intelligent Design & Optimization

## Tidy3D Integration Branch

This branch integrates Tidy3D FDTD (Finite-Difference Time-Domain) simulation capabilities with the PhIDO workflow for iterative device optimization. The integration enables automated optimization of photonic devices using full-wave electromagnetic simulation.

## ⚠️ Important Notice

**This branch is not stable and does not ensure proper functioning in normal workflow mode (vs optimization workflow mode).** The Tidy3D integration is specifically designed for optimization workflows and may cause issues when used in standard design mode.

## Dependencies

This branch requires specific versions of key dependencies:

- `tidy3d==2.8.4` - Tidy3D FDTD simulation engine
- `gdsfactory==9.8.0` - GDS layout generation and manipulation
- `gplugins==1.3.8` - GDSFactory plugins including Tidy3D integration

## Tidy3D FDTD Optimization Integration

### Workflow Overview

The Tidy3D integration adds a new optimization workflow mode that can be enabled via the "Enable optimizer mode" toggle in the web interface. When enabled, the system follows this workflow:

1. **Circuit Design Phase (P100-P300)**: Standard circuit interpretation and schematic generation
2. **Optimization Phase (P600)**: Tidy3D FDTD-based device optimization instead of standard layout generation

### Figure of Merit (FoM) Optimization

The system iteratively optimizes a user-defined Figure of Merit (FoM) through the following process:

#### FoM Definition
- FoM is defined in the circuit DSL under `nodes.N1.FoM`
- Contains three key fields:
  - `equation`: Python expression using S-parameters (S11, S12, etc.)
  - `direction`: Either "maximize" or "minimize"
  - `wavelength`: Target wavelength in micrometers or "none" for broadband

#### S-Parameter Mapping
The system maps standard S-parameter notation to Tidy3D output format:
- S11 → "o1@0,o1@0"
- S12 → "o2@0,o1@0"
- S21 → "o1@0,o2@0"
- etc.

#### Optimization Process
1. **Parameter Extraction**: Identifies variable parameters from `opt_settings` and static parameters from `params`
2. **FDTD Simulation**: Runs Tidy3D simulations for each parameter set
3. **FoM Evaluation**: Computes FoM using the defined equation and S-parameter results
4. **Bayesian Optimization**: Uses scikit-optimize's Gaussian Process optimization to find optimal parameters
5. **Convergence Tracking**: Monitors optimization progress and generates convergence plots

### Key Components

#### `device_optimizer()` Function
Located in `PhotonicsAI/Photon/DemoPDK.py`, this function:
- Extracts device specifications from the circuit DSL
- Sets up the FoM evaluation function
- Configures Tidy3D simulation settings
- Runs the optimization loop
- Generates optimization plots

#### `GP_BO` Class
Located in `misc_tests/Optimization/device_opt/tidy3d_simulator.py`, this class:
- Implements Bayesian optimization using Gaussian Processes
- Manages FDTD simulation calls
- Handles parameter bounds and constraints
- Generates optimization visualizations

#### Simulation Settings
The `SimulationSettingsTiny3DFdtd` class in `misc_tests/Optimization/device_opt/SimulationSettings.py` configures:
- Material properties (Si, SiO2, SiN)
- Mesh settings and grid specifications
- Boundary conditions (PML)
- Source parameters and wavelength range
- Port configurations

### Usage Example

1. Enable "Optimizer mode" toggle in the web interface
2. Describe a photonic device with optimization requirements
3. The system will automatically:
   - Generate a suitable FoM based on the device type
   - Define optimization parameters and bounds
   - Run iterative FDTD simulations
   - Display convergence plots and optimized parameters

### Output Visualization

The optimization process generates three key visualizations:
- **Convergence Plot**: Shows FoM improvement over iterations
- **Parameter Optimization**: Displays parameter evolution during optimization
- **Final S-Parameters**: Shows the optimized device's scattering parameters

## Installation

Before installing the python package, you need to install graphviz:
```bash
apt-get install graphviz graphviz-dev
```

```bash
make install
```

## Configuration

Create an `.env` file in the current working directory or HOME folder:
```
OPENAI_API_KEY='your-api-key'
```

## Running

```bash
make run
```

## Branch Stability

This branch is experimental and intended for optimization workflows only. For stable design workflows, use the main branch or other stable branches.
