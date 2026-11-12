---
title: uPrime
site:
  hide_outline: true
  hide_title_block: true
---

::::::{div}
:class: up-hero col-page-right

:::::{div}
:class: up-brand

:::{image} logo_mark.png
:alt: uPrime logo
:class: up-mark dark:hidden
:::

:::{image} logo_mark_dark.png
:alt: uPrime logo
:class: up-mark hidden dark:block
:::

# uPrime

:::::

:::{div}
:class: up-tagline

_Turbulence is complex. Analysis shouldn\'t be._
:::

:::{div}
:class: up-badges

![License](https://img.shields.io/badge/license-GPLv3-blue)
![Version](https://img.shields.io/badge/version-0.9.2-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![PyPI](https://img.shields.io/pypi/v/uprime-piv)](https://pypi.org/project/uprime-piv/)
:::

::::{div}
:class: up-pip

```text
pip install uprime-piv
```

:::{div}
:class: up-pip-run

then run `uprime`
:::
::::

:::{div}
:class: up-buttons

{button}`Download v0.9.2 <https://github.com/CmdrRyder/uPrime/releases/latest>`
{button}`View on GitHub <https://github.com/CmdrRyder/uPrime>`
{button}`DOI <https://doi.org/10.5281/zenodo.19376184>`
{button}`Manual (PDF) <https://cmdrryder.github.io/uPrime/docs/Manual.pdf>`
:::

::::::

:::{div}
:class: up-shot-wrap col-page-right

```{image} gui_v08.png
:alt: uPrime v0.8 GUI
:class: up-shot
```
:::

[uPrime](https://cmdrryder.github.io/uPrime/) [@jose2026uprime] is a standalone desktop application for post-processing and
turbulence analysis of velocity field data from PIV and CFD. Reads
Tecplot ASCII .dat (LaVision DaVis) and MATLAB .mat files.

_No scripting.  No setup. Just analysis._

:::::::{div}
:class: up-split col-page-right

::::::{div}
:class: up-features

## Key Features

:::::{div}
:class: up-feat-grid

::::{div}
:class: up-feat up-ic-convergence

**Convergence verification**

Cumulative moments at any point, with automatic N\* detection.
::::

::::{div}
:class: up-feat up-ic-large

**Large dataset support**

Datasets over 4 GB are memory-mapped: 10,000+ snapshots without running out of RAM.
::::

::::{div}
:class: up-feat up-ic-mask

**Non-destructive masking**

Mask walls, shadows and reflections; the raw data is never modified.
::::

::::{div}
:class: up-feat up-ic-export

**Publication-ready export**

PNG at 300 DPI, plus PDF and SVG with editable text.
::::

::::{div}
:class: up-feat up-ic-exe

**Standalone executable**

A Windows `.exe` with no Python needed; macOS and Linux install with pip.
::::

::::{div}
:class: up-feat up-ic-transform

**Transform & alignment**

Rotate, shift and mirror fields to correct camera misalignment.
::::

::::{div}
:class: up-feat up-ic-pipeline

**Full turbulence pipeline**

Stresses, TKE budget, spectra, POD, DMD and vortices in one environment.
::::

::::{div}
:class: up-feat up-ic-cases

**Multi-case comparison**

Overlay earlier results from CSV, with per-case styling and export.
::::

:::::

::::::

::::::{div}
:class: up-modules up-panel

## Analysis Modules

- Mean & Convergence
- Compare Cases
- Reynolds Stress Analysis
- TKE Budget
- Space-Time Spectral Analysis
- Anisotropy Invariants {span .up-tag}`Stereo only`
- Correlation Analysis
- POD Analysis
- DMD Analysis {span .up-tag}`TR only`
- Vortex Identification
- Probability Analysis
- Mask Editor
- Coordinate Transform

::::::

:::::::

::::::{div}
:class: up-split up-split-news col-page-right

:::::{div}
:class: up-news-col

## What's new

:::{dropdown} What's new in v0.9.2
:open:
:class: up-news

- **`uprime-piv` command alias**: the command can now be typed as the
  package name, `uprime-piv`, as well as `uprime`. An existing 0.9.1
  install gets it with `pip install --upgrade uprime-piv`.
- **Install instructions show the run command**: the README, the API docs
  and this site now say that you install `uprime-piv` but run `uprime`.
- **Publishing is release-only**: a GitHub release is the only trigger for a
  PyPI upload, and the upload still waits for approval.
- **README images and links pinned to the release tag**, so the description
  frozen into each PyPI release keeps working if files move.
- **PyPI documentation link** now opens the user manual.
- **CI actions on Node 24**, replacing versions GitHub flagged as deprecated.
:::

::::{dropdown} Previous releases
:class: up-news

**v0.9.1** (2026-11-11)

- **Published to PyPI as `uprime-piv`**: `pip install uprime-piv` installs
  uPrime and the `uprime` command on Windows, macOS and Linux.
- **pyFFTW is an optional extra**: install it with
  `pip install "uprime-piv[fft]"`. Without it only the Spatial E(k) using FFT
  tab is disabled. The Windows executable still bundles it.

**v0.9.0** (2026-11-08)

- **Corrected invalid-vector handling**: a vector marked invalid is now
  excluded in every snapshot it is invalid in, not just the first.
  **Statistics produced from `.dat` data by v0.4.1–v0.8.2 should be
  recomputed.**
- **Installable as a package**: `pip install -e .` gives a `uprime` command
  and an importable `uprime` package.
- **Documented Python API**: `docs/API.md` covers every public function in
  `uprime.core`, with a worked example.
- **One grid convention**: `.dat`, `.vc7` and `.mat` all load with row 0 at
  the smallest y and column 0 at the smallest x.
- **DMD handles gaps like POD**, and both report the fraction of filled
  values.
- **CI on Windows, macOS and Linux.**

**v0.8.x** (2026-09-01 to 2026-09-03)

- **Probability Analysis module** (0.8.0): PDF, Flow Direction, Binary
  Space-Time and Quadrant tabs, with a deadband for indeterminate flow
  direction. A drawn ROI restricts the computation, not only the display.
- **Manual x/y line entry in Anisotropy Invariants** (0.8.1), with the line
  recorded in the export.
- **Lumley triangle comparison in Compare Cases** (0.8.2), with a merged CSV
  export.

The full history is in the
[CHANGELOG on GitHub](https://github.com/CmdrRyder/uPrime/blob/main/CHANGELOG.md).
::::
:::::

:::::{div}
:class: up-formats-col

:::{div}
:class: up-formats up-panel

## Input Formats

- **Tecplot ASCII `.dat`**  One file per snapshot, as exported by LaVision DaVis.
- **MATLAB `.mat`**  Full time series in one file, classic and v7.3 (HDF5).
- **DaVis `.vc7` / `.vec`**  Needs lvpyio; works from source or a pip install, not the .exe.
:::
:::::

::::::


:::{div}
:class: up-footer

Developed by **Jibu Tom Jose** · [Transient Fluid Mechanics Laboratory](https://www.tfmltechnion.com/), Technion

Built with assistance from
[Claude](https://www.anthropic.com) (Anthropic)

[GitHub](https://github.com/CmdrRyder/uPrime) ·
[Releases](https://github.com/CmdrRyder/uPrime/releases) ·
[Zenodo](https://doi.org/10.5281/zenodo.19376184) ·
[GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html)

These are derived quantities, not conclusions. Interpret with care.
:::
