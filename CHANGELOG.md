# Changelog

All notable changes to uPrime are documented here.  

---

## [v0.3] — Alpha Release

### Added

**POD Analysis module** (`core/pod.py`, `gui/pod_window.py`)
- Snapshot POD via temporal correlation matrix and eigendecomposition.
- Energy Spectrum tab: bar chart of modal energy with cumulative curve and 50/80/90% thresholds.
- Spatial Modes tab: contourf of any mode and component with energy percentage in title.
- Temporal Coefficients tab (TR data only): time series and Welch PSD of modal coefficients.
- Reconstruction tab: original vs reconstructed snapshot side by side with residual field and RMS error readout.
- Configurable number of modes (default 25).
- Live snapshot slider for reconstruction.
- Export: all spatial modes as Tecplot `.dat`, temporal coefficients as `.csv`.

**Correlation Analysis — improvements**
- Four integral scale methods: zero crossing, exponential fit, 1/e point, domain integral — consistent with Fuchs et al. (2022, *Phys. Fluids*).
- `one_over_e` method corrected to integrate R(Δr) up to the crossing point rather than returning the crossing location.
- `exp_fit` range restricted to [0, r_{1/e}] to avoid fitting the noisy tail.
- Method-aware legend: only markers relevant to the selected method are shown on each plot.
- Spatial tab split into 2D correlation map (top) and two independent 1D slice panels (bottom, x and y).
- "Show Diagnostic Plot" button opens cumulative integral on demand instead of always being visible.
- Vertical dashed red line marks the computed L on each 1D correlation plot.
- Scale method dropdown moved above the plot in both spatial and temporal tabs.
- Reconstruction tab enabled for non-TR data (temporal coefficients tab remains greyed out).

**Main window**
- Window title updated to `uPrime v0.3`.
- POD Analysis entry added to the Analysis panel.

### Fixed

- `numpy.trapz` → `_trapz` alias applied consistently throughout `core/two_point_corr.py` to support NumPy 2.0+.
- `_integral_to_zero` undefined name error in `compute_spatial_correlation_roi` — replaced with `compute_length_scale`.
- `compute_time_scales` return value unpacking corrected (now returns 3 values: T, lambda_t, extras).
- `CorrelationWindow` missing `combo_scale_method` attribute — dropdown now created in `_build_spatial_tab`.
- POD reconstruction shape mismatch `(n_modes, 1, 1)` vs `(n_modes, ny, nx, Nc)` — einsum-based reconstruction implemented correctly.
- POD reconstruction minimum modes set to 1 (was 0, which caused a crash).
- POD reconstruction component selector (U/V/W) now updates all three panels (original, reconstructed, residual).

---

## [v0.2] — Alpha Release

### Added

- **TKE budget** module: production, convection, turbulent diffusion, residual, ∂k/∂t (TR only).
- **Anisotropy invariant analysis**: Lumley triangle (line-profile mode) and barycentric RGB map (ROI mode). Stereo PIV only.
- **Correlation analysis**: two-point spatial correlation (point and ROI modes), temporal autocorrelation (TR only), integral length and time scales.
- **Coordinate transform tool**: rotation up to ±10° with linear or cubic interpolation, origin shift by click or manual entry.
- **Space–time spectral analysis**: spatial E(k), temporal E(f) via Welch, and space–time E(k,f).
- Unified `SpectraWindow` replacing separate spectral windows.
- DrawAwareToolbar suppresses pick/draw events during zoom and pan.
- Convergence warnings on window open (snapshot count and recording duration).
- Export to Tecplot `.dat` and `.csv` in all analysis windows.

### Changed

- Migrated from PyQt5 to PyQt6 throughout.
- Consistent font sizes (FONT_AX=9, FONT_TICK=8, FONT_LEG=8) and colormap conventions (RdBu_r diverging, viridis sequential) across all windows.
- Colorbar `extend='neither'` enforced everywhere.

---

## [v0.1] —  Initial Alpha Release

### Added

- Load multi-snapshot DaVis Tecplot `.dat` files (2D and stereo PIV).
- Mean field viewer: Mean U/V/W, speed, vorticity, Std(U), Std(V) with contourf and vector overlay.
- Reynolds stress analysis: all $R_{ij}$ components, 2D maps, line profiles, ±1σ uncertainty bands.
- TKE viewer: 2D contour and line profile.
- Basic temporal spectral analysis (Welch PSD at a point or ROI).
- PyInstaller `.exe` build for Windows.
