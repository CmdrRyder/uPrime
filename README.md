# uPrime

[![DOI](https://zenodo.org/badge/1197739784.svg)](https://doi.org/10.5281/zenodo.19376184)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)
![Version](https://img.shields.io/badge/version-0.3%20alpha-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

### *Because u′ matters*

![GUI](docs/images/gui.png)

**uPrime** is a standalone software for post-processing and analysis of velocity field data from both experimental and numerical sources. Distributed as a ready-to-use executable, it provides a unified environment for analyzing planar and stereo velocity datasets without requiring programming or external dependencies.

The software is designed to streamline common workflows in fluid mechanics by bringing multiple analysis tools into a single, intuitive graphical interface. While originally developed for Particle Image Velocimetry (PIV), uPrime is equally applicable to CFD datasets and other velocity field data, provided they are supplied in the appropriate format.

uPrime is capable of handling large datasets efficiently, making it suitable for high-resolution and time-resolved measurements commonly encountered in modern experiments and simulations.

Currently, uPrime supports planar (2D2C) and stereo (2D3C) velocity fields.

---

## Key Features

- **Standalone executable**  
  No installation of Python or external libraries required. Runs as a ready-to-use application.

- **User-friendly graphical interface**  
  Clean and intuitive GUI designed for rapid data loading, visualization, and analysis without coding.

- **Unified analysis environment**  
  Multiple turbulence and flow diagnostics available within a single platform.

- **Supports experimental and numerical data**  
  Compatible with PIV and CFD velocity fields, as well as any structured velocity dataset in the required format.

- **Handles large datasets**  
  Designed to process high-resolution and time-resolved velocity fields efficiently.

- **Time-resolved and non-time-resolved support**  
  Handles both instantaneous datasets and time-resolved sequences for advanced analysis.

- **Alignment and transformation tools**  
  Built-in utilities for shifting, rotating, and aligning datasets, particularly useful for correcting calibration offsets and geometric misalignment in PIV measurements.

---

## Available Analysis Modules

- Reynolds stress tensor and profiles  
- Turbulent kinetic energy (TKE) and budget terms  
- Space, time, and space–time spectral analysis  
- Anisotropy invariants (Lumley triangle, barycentric map)  
- Two-point correlation analysis  
- Proper Orthogonal Decomposition (POD)  

---

## Workflow Highlights

- Load velocity data directly from `.dat` files  
- Interactively select regions, lines, and domains  
- Apply alignment and transformation corrections when needed  
- Perform analysis with minimal setup  
- Visualize results instantly with built-in plotting tools  
- Export processed data for further use  

---

## Scope and Limitations

- Currently supports **planar and stereo velocity data only**  
- Input data must follow the required structured format  
- Designed primarily for structured grid datasets  

---

## Example Results

### Reynolds Stress Analysis
![Reynolds](docs/images/reynolds.png)

### Space–Time Spectral Analysis
![Spectra](docs/images/spectra.png)

--

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

> Jose, J. T. (2026). uPrime: Open-source velocity field analysis toolkit for PIV and CFD. Transient Fluid Mechanics Laboratory (TFML), Technion – Israel Institute of Technology. https://doi.org/10.5281/zenodo.19376184
---

## Contributing

Contributions are welcome. If you plan to add new analysis modules or extend existing functionality:

- Follow the existing structure (`core/` for computations, `gui/` for interface)  
- Keep modules self-contained  
- Update documentation where relevant  

For major changes, please open an issue first to discuss the proposed addition.

---

## Development Philosophy

uPrime is developed and maintained by the Transient Fluid Mechanics Laboratory (TFML), Technion.

To ensure consistency and reliability, we encourage new features and extensions to be integrated within the main repository rather than maintained as separate forks.

---

## License

This project is licensed under the GNU General Public License v3.0 (GPLv3).

This ensures that any modifications or derivative works remain open-source and properly attribute the original work.

Full license text: https://www.gnu.org/licenses/gpl-3.0.en.html

---

## Author

**Jibu Tom Jose**  
Postdoctoral Research Fellow  
Department of Mechanical Engineering  
Technion — Israel Institute of Technology, Haifa, Israel  

Built with assistance from [Claude](https://www.anthropic.com) (Anthropic).
