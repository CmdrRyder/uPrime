# uPrime

[![DOI](https://zenodo.org/badge/1197739784.svg)](https://doi.org/10.5281/zenodo.19376184)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
![Version](https://img.shields.io/badge/version-0.3%20alpha-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

### *Because u′ matters*

**uPrime** is an open-source desktop application for post-processing planar and stereo PIV data. It reads Tecplot `.dat` files exported from LaVision DaVis and provides a clean GUI for turbulence analysis — no MATLAB or Python scripting required.

Developed at the **Transient Fluid Mechanics Laboratory, Technion — Israel Institute of Technology**.

---

## Interface

<!-- Add screenshot here -->
![uPrime GUI](docs/screenshot_main.png)

> The interface is organized around a central visualization panel with modular analysis windows that can be opened and compared simultaneously.

---

## Features

| Module | Description | TR required |
|---|---|---|
| **Mean field viewer** | Contour maps of U, V, W, speed, vorticity, Std(U/V) with vector overlay | — |
| **Reynolds stresses** | All $R_{ij}$ components, 2D maps and line profiles | — |
| **TKE budget** | Production, convection, turbulent diffusion, residual, ∂k/∂t | ∂k/∂t only |
| **Spectral analysis** | Spatial E(k), temporal E(f) via Welch, space–time E(k,f) | E(f), E(k,f) |
| **Correlation analysis** | Two-point spatial/temporal correlations, integral length and time scales, four scale methods | Temporal tab |
| **Anisotropy invariants** | Lumley triangle and barycentric RGB maps | — (stereo only) |
| **POD** | Energy spectrum, spatial modes, temporal coefficients, flow reconstruction | Coefficients tab |
| **Coordinate transform** | Rotation (±10°) and origin shift, linear or cubic interpolation | — |

All analyses open in separate windows so you can compare results side by side. Every result can be exported to Tecplot `.dat` or `.csv`.

---

## Installation

### Option 1 — Windows standalone `.exe`

Download the latest `uPrime_v0.3.exe` from the [Releases](https://github.com/CmdrRyder/uPrime/releases) page. No Python installation needed.

> **Note:** Windows Defender may flag the `.exe` on first run. Click **More info → Run anyway**. This is normal for unsigned PyInstaller executables.

### Option 2 — Run from source

Requires **Python 3.10 or later**.

```bash
git clone https://github.com/CmdrRyder/uPrime.git
cd uPrime
pip install -r requirements.txt
python main.py
```

---

## Quick start

1. Launch uPrime.  
2. Click **Select .dat Files** and choose your snapshot files.  
3. Set **Acquisition Type** — Time-Resolved (TR) or Non-TR — and enter $f_s$ if TR.  
4. Use the field viewer to inspect mean fields and verify the coordinate system.  
5. Apply **Transform / Align** if needed (rotation, origin shift).  
6. Open any analysis module from the **Analysis** panel.  

---

## Documentation

A full user manual is available:

📄 [uPrime User Manual](docs/manual.pdf)

> The manual provides detailed explanations of all modules, workflows, and implementation notes. It is updated alongside the software.

---

## Input format

uPrime reads **Tecplot ASCII `.dat` files** in the standard DaVis export format:

```
TITLE = "filename"
VARIABLES = "x [mm]", "y [mm]", "Velocity u [m/s]", "Velocity v [m/s]", ...
ZONE T="Frame 0", I=NX, J=NY, F=POINT
...data...
```

- One snapshot per file; select multiple files at once to load a time series.  
- Column names are detected automatically — extra columns (vorticity, acceleration, etc.) are handled gracefully.  
- Both 2D (u, v) and stereo (u, v, w) exports are supported.  

Compatible with **LaVision DaVis** and any CFD post-processor that writes Tecplot ASCII format.

---

## Code Structure

```
uPrime/
│── main.py              # application entry point
│── requirements.txt
│
├── gui/                 # user interface (PyQt)
├── core/                # numerical routines and analysis modules
├── docs/                # manual, screenshots, documentation assets
```

---

## Roadmap

- [ ] FTLE (Finite-Time Lyapunov Exponents)  
- [ ] DMD (Dynamic Mode Decomposition)  
- [ ] Spectral POD (SPOD)  
- [ ] Phase averaging for cyclic/oscillatory data  
- [ ] macOS and Linux packaged releases  

---

## Citation

If uPrime contributes to published research, please cite:

> Jibu Tom Jose. *uPrime: Open-source PIV post-processing toolkit*. Transient Fluid Mechanics Laboratory, Technion, 2026. https://doi.org/10.5281/zenodo.19376184

---

## Contributing

Contributions are welcome. If you plan to add new analysis modules or extend existing functionality:

- Follow the existing structure (`core/` for computations, `gui/` for interface)  
- Keep modules self-contained  
- Update documentation where relevant  

For major changes, please open an issue first to discuss the proposed addition.

---

## License

**CC BY-NC-ND 4.0** — Free for personal, academic, and research use with attribution. Commercial use and redistribution require explicit written permission.  
Full license: https://creativecommons.org/licenses/by-nc-nd/4.0/

---

## Author

**Jibu Tom Jose**  
Postdoctoral Research Fellow  
Department of Mechanical Engineering  
Technion — Israel Institute of Technology, Haifa, Israel  

Built with assistance from [Claude](https://www.anthropic.com) (Anthropic).
