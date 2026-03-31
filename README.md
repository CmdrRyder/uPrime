# uPrime

### *Because u' matters*

**uPrime** is an open-source fluid velocity field analysis toolkit. It reads Tecplot `.dat` format files exported from DaVis (LaVision) or CFD solvers and provides a clean GUI for turbulence post-processing — no MATLAB required.

Developed at the **Transient Fluid Mechanics Laboratory, Technion — Israel Institute of Technology**.

---

## Current Features (v0.1 Alpha)

- **Data loading** — 2D planar and stereo PIV auto-detected, up to 10,000+ snapshots
- **Mean flow display** — contour maps of U, V, W, speed, vorticity, Std(U), Std(V) with interactive vector overlay
- **Reynolds stresses** — all Rij components, 2D contour maps and line profiles, optional normalization by Um²
- **Turbulent kinetic energy (TKE)** — 2D and full 3C formulations, contour maps and line profiles
- **Temporal spectral analysis** — Welch PSD for u, v, w with Kolmogorov -5/3 reference line, point or rectangle selection
- **Anisotropy invariant analysis** — Lumley triangle (-I₂ vs I₃) and barycentric RGB maps
- **Convergence warnings** — automatic alerts for under-sampled datasets

---

## Coming Soon

- Spatial spectra
- POD (Proper Orthogonal Decomposition)
- FTLE (Finite-Time Lyapunov Exponents)
- TKE budget (production, dissipation, diffusion, convection)
- DMD (Dynamic Mode Decomposition)
- Export results to CSV / HDF5
- Cross-correlation and two-point statistics
- Phase averaging for cyclic data
- Linux and macOS support

---

## Input Format

uPrime reads **Tecplot ASCII `.dat` files** with the following structure:

```
TITLE = "filename"
VARIABLES = "x [mm]", "y [mm]", "Velocity u [m/s]", "Velocity v [m/s]", ...
ZONE T="Frame 0", I=NX, J=NY, F=POINT
...data...
```

- One snapshot per file
- Columns are auto-detected by name — extra columns (acceleration, pressure, vorticity, etc.) are handled gracefully
- Both 2D (u, v) and stereo (u, v, w) exports are supported

Compatible with **LaVision DaVis** exports and any CFD post-processor that writes Tecplot ASCII format.

---

## Installation

### Option 1 — Windows standalone `.exe` (recommended)

Download `uPrime_v0.1.exe` from the [Releases](https://github.com/CmdrRyder/uPrime/releases) page. No Python installation needed. Just download and run.

> **Note:** Windows Defender may flag the `.exe` on first run. Click "More info" → "Run anyway". This is normal for PyInstaller executables.

### Option 2 — Run from source

**Requires Python 3.11**

```bash
git clone https://github.com/CmdrRyder/uPrime.git
cd uPrime
pip install -r requirements.txt
python main.py
```

---

## Quick Start

1. Launch uPrime
2. Click **Select .dat Files** and choose your snapshots
3. Set acquisition type (Time-Resolved or Non-Time-Resolved) and sampling frequency
4. Use the **Display** panel to explore mean fields
5. Select an analysis from the **Analysis** panel and click **Run Analysis**

---

## License

**CC BY-NC-ND 4.0** — Creative Commons Attribution-NonCommercial-NoDerivatives 4.0

Free for personal, academic, and research use with attribution. If used in published research, please cite:

> Jibu Tom Jose, *uPrime — Open-source fluid velocity field analysis*, Technion (2026).
> https://github.com/CmdrRyder/uPrime

Commercial use and redistribution are not permitted without explicit written permission.
Full license: https://creativecommons.org/licenses/by-nc-nd/4.0/

---

## Author

**Jibu Tom Jose**
Postdoctoral Research Fellow
Department of Mechanical Engineering
Technion — Israel Institute of Technology, Haifa, Israel

Built with assistance from [Claude](https://www.anthropic.com) (Anthropic).
