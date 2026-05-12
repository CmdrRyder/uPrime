# Changelog

All notable changes to uPrime are documented here.  

---

## [v0.6.1]

### Fixed
- `.mat` loader filled masked-out grid points with 0 instead of NaN after applying the validity mask. All analysis modules treat 0 as a valid velocity value, so masked regions appeared as zero-velocity patches rather than being excluded from statistics, spectra, and profiles. Fixed by applying `U[~mask_2d, :] = np.nan` (and V, W) in-place immediately after the mask is finalised, matching the behaviour of the `.dat` loader.
- `get_masked()` in `core/dataset_utils.py` promoted velocity arrays to float64 unconditionally on every call (`astype(float)` in Python is float64), doubling memory usage for float32 datasets and causing OOM in any module that called it on float16 datasets. Now preserves the stored dtype via `field.copy()`.
- `.mat` variable confirmation dialog now includes an explicit **Mask convention** dropdown next to the isValid selector. Default is "1 = valid (uPrime default)" matching the `.dat` convention. Select "1 = invalid (mask out)" for files that use the inverted convention (e.g. a `Mask` variable where 1 means "invalid"). The previous auto-detection heuristic was logically flawed: it sampled a mid-stack velocity frame before NaN was applied, causing it to fire incorrectly on files where >50 % of the field is valid, double-inverting an already-correct mask. Auto-detection is removed entirely in favor of explicit user choice. A status-bar warning fires if the resulting masked fraction is extreme (< 5 % or > 95 %) to help catch wrong-convention loads without auto-fixing them.
- All `contourf` calls across every analysis window now wrap the field in `np.ma.masked_invalid()` before rendering. Previously, NaN values at masked grid points were rendered as the colormap minimum (dark blue in RdBu_r) instead of the white axes background, making masked regions indistinguishable from valid low-velocity regions.

---

## [v0.6.0]

### Changed
- **TKE Budget line profile** now shows TKE k in a dedicated upper panel and budget terms (P, C, D, R) in a lower panel sharing the same x-axis, with a 1:2 height ratio. Multi-case comparison uses the same two-panel layout.
- `.mat` loader now reads metadata first; variable confirmation dialog appears before any large array reads, allowing the user to cancel a wrong-file selection without waiting for the load.
- 2D meshgrid x and y arrays in `.mat` files are now automatically extracted to 1D coordinate vectors.
- Variable confirmation dialog is now shown on every `.mat` load (not only when auto-detection is incomplete). Auto-detected names are pre-filled for quick confirmation.

### Added
- Opt-in float16 (half precision) storage mode for `.mat` files > 5 GB. Halves in-memory size, allowing files that would otherwise not fit in RAM to load. All downstream computations run at the chosen precision; the user is warned that fluctuation statistics, POD, DMD, and FFT spectra may be inaccurate. The checkbox appears in the variable confirmation dialog only when the estimated load size exceeds 5 GB. Default remains float32.
- Snapshot subset selection in the `.mat` variable confirmation dialog (start, end, step), matching the existing `.dat` loader behavior. Allows loading a strided subsample of very large `.mat` files that would otherwise exhaust memory.
- Single-snapshot `.mat` files are detected from metadata and rejected with a clear error message before any data is loaded.

### Fixed
- Spectral Analysis crashed with `MemoryError` on float16-loaded `.mat` datasets because `get_masked()` promoted the full velocity stack to float64. With that root cause fixed (see v0.6.1), the remaining issue is that FFT routines do not support float16; float16→float32 guards are added at the slice level inside each spectral computation routine so FFT inputs are always at least float32.
- `.mat` loader progress bar previously did not update during large file reads. v7.3 (HDF5) loads now show determinate per-snapshot progress; classic `.mat` loads show an indeterminate busy indicator with file size in the status bar.
- Field display crashed with `ZeroDivisionError` when all x-coordinates were identical (degenerate grid, e.g. single-column `.mat` datasets). Guard added: if the computed vector arrow length is zero or non-finite, it falls back to 1.0 and shows a status-bar warning. `contourf` colormap normalisation also guarded against uniform (all-zero) fields.
- `.mat` loader now accepts 2D meshgrid coordinate arrays (`xm`, `ym`) in either `(Ny, Nx)` or `(Nx, Ny)` orientation; meshgrid orientation is now detected primarily from the variance pattern in the array values (x should vary along columns, y along rows), with shape comparison as a fallback for ambiguous cases.
- `.mat` loader incorrectly swapped `Ny` and `Nx` for files where the velocity array is stored in non-standard `(Nx, Ny, Nt)` order (CFD/ndgrid convention) instead of the standard `(Ny, Nx, Nt)` order. The loader now peeks at the x-coordinate shape before any large read and selects the correct chunk-transpose; velocity, coordinate, and mask arrays are all loaded in the correct `(Ny, Nx)` orientation. `dx`/`dy` were computed as 0 from wrongly-oriented arrays, breaking grid-aware features (canvas aspect, picker click registration, FFT spacing).
- `.mat` loader now estimates `dx` and `dy` from the coordinate arrays using a robust median-of-differences formula and stores them in the dataset; status bar now shows the correct grid spacing instead of `0.000 mm`. A warning is appended to the status bar if spacing variation exceeds 1% (non-uniform grid).
- Mean & Convergence Tab 2 (Convergence) picker placed the marker at grid corner `(0, 0)` instead of the click location for elongated fields; canvas height is now computed from the actual widget width instead of a hardcoded constant, eliminating white margins that caused click dead-zones (same aspect-ratio pattern as all other modules).
- Mean & Convergence multi-case comparison crashed with `TypeError` on `np.isfinite` when imported CSV columns were not coerced to float dtype. `_Series` now retries element-wise conversion on bulk-parse failure, mapping empty/None cells to NaN and only falling back to object dtype for genuinely non-numeric columns; `_plot_comparison` also coerces each column to float immediately before the `isfinite` mask.

---

## [v0.5.1]

### Added
- **Mean & Convergence module** (`core/mean_convergence_core.py`, `gui/mean_convergence_tab.py`): first entry in the Analysis panel.
  - **Mean Velocity sub-tab**: viridis |⟨U⟩| field display with free / horizontal / vertical line modes, ±grid-point spatial averaging band, optional U_m normalization, CSV export of ⟨U⟩, ⟨V⟩, ⟨W⟩ profiles, and **multi-case import/compare** matching the Reynolds Stress workflow exactly (Import Case, Compare, Edit Styles, Manage Cases, Export All Cases). Per-case U_m header is read from each imported CSV; mismatch against the current normalization setting is reported in the status bar per case. Line geometry is session-only (no JSON import/export).
  - **Convergence sub-tab**: kernel-averaged point statistics computed via Welford's online algorithm (single pass, O(N)) for cumulative 1st, 2nd, and 3rd central moments of u′, v′, w′. Boundary-margin shading with rejection feedback. Configurable kernel size (odd integers), per-order thresholds, and trailing-window stability criterion (W = max(50, N // 10)). Grid layout shows raw physical-unit q_N with dashed threshold band (q_final ± ε·scale) per subplot. Single normalized layout shows q_N/scale converging to q_final/scale, with per-component dashed threshold bands around each asymptote. Annotated N* convergence lines per curve. CSV export with full header block (point coordinates, kernel, thresholds, N* per quantity).
  - Mean fields cached on window open; both sub-tabs read from the same cache. Recomputed only when the window is re-opened with a different dataset.

### Changed
- **Convergence warnings removed**: all pop-up `QMessageBox` convergence warnings removed from Reynolds Stresses, TKE Budget, TKE, Spectral Analysis, Correlation, Anisotropy, DMD, and main window load.
- **Inline convergence notice**: Reynolds Stresses, TKE Budget, and Spectral Analysis now display a persistent amber top-bar label ("⚠ Only N=… snapshots loaded…") when the loaded dataset has fewer than `CONVERGENCE_WARNING_N = 500` snapshots. Label is hidden for larger datasets. Other modules (Correlation, Anisotropy, DMD, etc.) are silent.
- Added `core/constants.py` with `CONVERGENCE_WARNING_N = 500`.

### Fixed
- **Mean Velocity tab**: stale picker state caused new line drawings to be ignored after a previous line was completed. Fixed null-coordinate guard in press handler, explicit button-1 checks in motion and release handlers, rubber-band cleanup on too-short release, and consistent reset of all press-state variables on every exit path.

---

## [v0.5.0]

### Added
- Added Export menu in main window with Tecplot ASCII mean field export (U, V, |V|, optional W)
- Added multi-case comparison to spectra, correlation, TKE budget, and anisotropy windows via CSV import
- Added Case Manager dialog for renaming, recoloring, and removing cases
- Added quantity selector and layout mode chooser for multi-case comparison plots
- Added combined multi-case CSV export with case column
- Added term visibility checkboxes to TKE budget line plot (P, C, D, R), all checked by default

### Changed
- Changed Reynolds stress sigma envelope to hidden by default

---

## [v0.4.1] — Alpha Release

### Added
- **DMD Analysis module** (TR only): frequency–growth rate spectrum with bubble size/color encoding amplitude, spatial mode viewer (U/V/W panels for stacked decomposition), Strouhal number toggle, growth rate and minimum frequency filters, Prev/Next mode navigation sorted by amplitude, mode label overlay, export spectrum CSV and mode field.
- **Vortex Identification module**: five scalar criteria — vorticity (ω), Q-criterion, swirling strength (λci), lambda-2 (λ2), and Γ1/Γ2 (Graftieaux et al. 2001). Per-vortex statistics (area, circulation, aspect ratio), spatial probability map, histogram of vortex properties split by rotation sign, export vortex table as CSV.
- **Non-destructive masking**: raw velocity data is never modified. Mask stored as a single 2D boolean array (`MASK`, shape [ny, nx]) applied at compute time via `get_masked()`. User-drawn masks are layered on top of the loaded isValid mask and can be added or removed without reloading.
- **Mask Editor**: draw Rectangle, Polygon, Circle, or Ellipse masks directly on the velocity field. Option to mask inside or outside the drawn shape. Multiple layers, undo per layer, save/load mask files.
- **Unit auto-detection**: coordinate and velocity units read directly from variable name strings in the `.dat` file header (e.g. `"x [mm]"`, `"Velocity u [m/s]"`). Defaults to mm and m/s if not specified. Conversion factors applied at load time; all internal computation uses meters and m/s.
- **Acquisition type popup**: a dialog appears automatically after each data load asking for TR/Non-TR and fs. Settings also accessible at any time from the dataset info ribbon.
- **Application logo**: uPrime logo (light and dark variants) shown in sidebar, About dialog, and window taskbar icon. Manual accessible via F1 or the `? Manual` button in the top-right of the main window.
- **QThread background workers**: all heavy computations (POD, DMD, TKE budget, correlation analysis, spectral analysis, vortex identification) now run in a background thread. The main window remains fully responsive during computation. A progress indicator appears below the compute button while work is in progress.
- **Large dataset support (>4 GB)**: datasets exceeding 4 GB are automatically memory-mapped to a temporary binary file in the system temp folder (`%TEMP%`) instead of being loaded into RAM. The subsampling dialog displays a warning with the temp path and required free disk space when this threshold is exceeded. The temp file is deleted automatically when uPrime closes.
- **Test suite**: 30 automated pytest tests covering all core modules (TKE budget, correlation, vortex identification, POD, DMD, loader) and GUI smoke tests for all analysis windows. Tests run headlessly with no real `.dat` files required.

### Changed
- Acquisition type (TR/Non-TR) and fs moved from left sidebar to the dataset info ribbon.
- Sidebar renumbered: Load Data (1), Preprocess (2), Analysis (3).
- Correlation analysis: 1/e scale method now returns the interpolated lag where R first drops to 1/e (not the integral up to that point); red marker placed at the actual crossing lag. Zero-crossing marker placed at the crossing lag rather than the integral value L.
- Correlation analysis: zero-crossing detection uses sustained crossing (5 consecutive negative points) with post-crossing mean check to reject noise dips.
- Default window size increased; window centered on screen at launch.
- Close confirmation dialog suppressed when no dataset is loaded.

### Fixed
- TKE budget: spatial gradients now correctly use meters (mm→m conversion applied via 1D coordinate arrays passed to `np.gradient`). NaN inpainting applied before differentiation to prevent spurious gradients at mask boundaries.
- Correlation analysis: ROI drawing changed from right-click to left-click drag, consistent with all other modules.
- Cursor value readout (bottom-left `x, y, value` display) now reports correct values after a mirror transform. Previously the coordinate-to-array index mapping was not updated after mirroring, causing the displayed value to correspond to the pre-mirror position.

---

## [v0.3.4] — Alpha Release

### Added
- Streamline support: rake-based seed drawing on the field, multiple rakes drawn cumulatively, reset button to clear all, color picker with preset palette, and line width control.
- Spatial FFT tab: 2D ROI-based spatial spectra using pyFFTW, averaged over all snapshots, with 1D marginal spectra E(kx) and E(ky) shown alongside.
- Vector controls ribbon: skip x/y, length, and arrow size controls in a dedicated second toolbar ribbon that appears only when Vectors or Streamlines is selected.

### Changed
- Welch spatial spectra tab restricted to Horizontal and Vertical line modes only; Rectangle ROI removed and is now exclusive to the FFT tab.
- Default Welch segment size changed from N//4 to N//2.
- Improved memory usage: velocity arrays loaded as float32 instead of float64. Invalid vectors masked to NaN at load time.
- Subset loading and reload: users can specify a snapshot range (start, end) and step/skip when loading. A reload button restores the full original dataset.
- Export improvements: PNG export at 300 DPI. Clean export mode available as checkboxes before saving.
- Version updated to v0.3.4 throughout.

### Fixed
- FFT tab: NaN regions filled with zero before FFT to prevent empty plot for partially masked ROIs.
- FFT tab: masked region warning shown only when masked fraction exceeds 5%.
- requirements.txt encoding corrected from UTF-16 to UTF-8.

---

## [v0.3] — Alpha Release

### Added
- **POD Analysis module**: snapshot POD via temporal correlation matrix. Energy Spectrum, Spatial Modes, Temporal Coefficients (TR only), and Reconstruction tabs. Export spatial modes as Tecplot `.dat` and temporal coefficients as `.csv`.
- **Correlation Analysis improvements**: four integral scale methods (zero crossing, exponential fit, 1/e point, domain integral). Spatial tab split into 2D map and two independent 1D slice panels. Diagnostic cumulative integral plot on demand.

### Fixed
- `numpy.trapz` → `_trapz` alias for NumPy 2.0+ compatibility.
- Various `CorrelationWindow` attribute and return value fixes.

---

## [v0.2] — Alpha Release

### Added
- TKE budget module: production, convection, turbulent diffusion, residual, ∂k/∂t (TR only).
- Anisotropy invariant analysis: Lumley triangle and barycentric RGB map. Stereo PIV only.
- Correlation analysis: two-point spatial and temporal autocorrelation, integral length and time scales.
- Coordinate transform tool: rotation, origin shift, mirror.
- Space–time spectral analysis: spatial E(k), temporal E(f) via Welch, space–time E(k,f).
- Migrated from PyQt5 to PyQt6.

---

## [v0.1] — Initial Alpha Release

### Added
- Load multi-snapshot DaVis Tecplot `.dat` files (2D2C and 2D3C).
- Mean field viewer with contourf and vector overlay.
- Reynolds stress analysis: all components, 2D maps, line profiles, ±1σ uncertainty bands.
- TKE viewer: 2D contour and line profile.
- Basic temporal spectral analysis.
- PyInstaller `.exe` build for Windows.
