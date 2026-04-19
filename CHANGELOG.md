# Changelog

All notable changes to uPrime are documented here.

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
