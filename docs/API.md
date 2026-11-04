# uPrime `uprime.core` API

## Overview

`uprime.core` holds all of uPrime's computation: loading, transforms, turbulence
statistics, spectra, correlations, modal decomposition, vortex identification,
probability analysis and export. None of it imports Qt, so every function here
can be called from a plain Python script without opening the GUI — useful for
batch processing, reproducible analysis scripts and testing.

`uprime.gui` is **not** a public API. It is the PyQt6 front end, its modules
import `uprime.core` freely, and its internals change without notice. Import
from `uprime.core` only.

```python
from uprime.core.loader import load_dataset
from uprime.core.reynolds_stress import compute_reynolds_stresses
```

---

## Data conventions

**Array shapes.** A time series is `[ny, nx, Nt]` — rows, columns, snapshots.
A single 2D field (a mean, a stress component, a vorticity map) is `[ny, nx]`.
Row 0 is the smallest `y`: the `.dat` loader flips rows when the file stores
them top-to-bottom, so this holds regardless of DaVis's own ordering.

**The third component.** `W` is `None` for planar (2D2C) data, not an array of
zeros or NaN. Core functions take `W` as a positional argument and branch on
`None` — `compute_reynolds_stresses(U, V, None)` returns `None` for `ww`, `uw`
and `vw`, and computes `k` as `0.5 * (uu + vv)`. The matching dataset flag is
`dataset["is_stereo"]`.

**Units.** `x` and `y` are in **mm**; velocities are in **m/s**. The `.dat`
loader reads the units out of the Tecplot variable names and stores the
conversion factors as `header["xy_to_mm"]` and `header["vel_to_ms"]`, applying
them during the load — so a file written in m and mm/s still arrives as mm and
m/s. Derived quantities follow: Reynolds stresses are (m/s)², integral length
scales are mm, time scales are s.

**Invalid vectors and NaN.** Invalid points are represented by NaN, and every
core routine uses NaN-aware reductions (`np.nanmean`, `np.nanstd`) so masked
points drop out of statistics instead of biasing them. *When* the NaN appears
depends on the loader — see [Masking](#masking) below, and the note in
[Known inconsistencies](#known-inconsistencies).

### The dataset dict

`load_dataset`, `load_mat_dataset` and `load_davis` all return a dict with the
same core keys:

| Key | Type | Meaning |
|---|---|---|
| `x`, `y` | `[ny, nx]` | coordinate grids, mm |
| `U`, `V` | `[ny, nx, Nt]` float32 | velocity, m/s |
| `W` | `[ny, nx, Nt]` or `None` | third component; `None` if planar |
| `vort` | `[ny, nx, Nt]` or `None` | vorticity, when the files carry it |
| `valid` | `[ny, nx]` bool | validity mask as loaded, `True` = valid |
| `valid_frac` | `[ny, nx]` float32 | `valid` as 0.0/1.0 |
| `MASK` | `[ny, nx]` bool | the mask `get_masked` applies |
| `MASK_LOADED` | `[ny, nx]` bool | copy of the loaded mask, so Mask Editor edits can be reverted |
| `mask_active` | bool | whether `get_masked` applies `MASK` |
| `is_stereo` | bool | `True` when `W` is present |
| `has_vort` | bool | `True` when the files carry vorticity |
| `Nt`, `nx`, `ny` | int | array dimensions |
| `files` | list of str | source files |
| `header` | dict | as returned by `parse_header` |
| `_memmap_path` | str or `None` | temp-file stem when memory-mapped |

`.mat` and DaVis datasets add `dx` and `dy` (grid spacing in mm); `.dat`
datasets do not. `load_mat_dataset` also attaches `_mat_status`, which the GUI
consumes and removes.

### Masking

`get_masked(dataset, key)` is how you read velocity for analysis:

```python
from uprime.core.dataset_utils import get_masked
U = get_masked(dataset, "U")      # [ny, nx, Nt], invalid points NaN
```

It returns a **copy** with `dataset["MASK"]` applied (invalid points set to
NaN), never touching the stored array, and preserves the stored dtype —
including float16, which supports NaN like any IEEE 754 float. If
`mask_active` is `False` it returns an unmasked copy; if the field is `None`
(asking for `"W"` on planar data) it returns `None`.

### Memory-mapped datasets

A `.dat` dataset whose estimated size exceeds `loader.SIZE_THRESHOLD` (4 GB) is
written to memory-mapped temp files instead of RAM. When that happens,
`dataset["_memmap_path"]` is set, and you should call `cleanup_memmap(dataset)`
when finished to delete the temp files:

```python
from uprime.core.loader import load_dataset, cleanup_memmap
dataset = load_dataset(files)
try:
    ...
finally:
    cleanup_memmap(dataset)      # safe no-op when not memory-mapped
```

`cleanup_memmap` is harmless on an in-memory dataset. `sweep_orphan_memmaps()`
removes temp files left behind by sessions that crashed, skipping the current
process's own files and any locked by another running instance; it returns the
number of bytes freed.

---

## Example

[`examples/core_api_example.py`](../examples/core_api_example.py) builds a
synthetic field with a known answer — a uniform mean flow plus one standing
mode — and checks the Reynolds stresses and POD against it.

```python
import numpy as np
from uprime.core.pod import compute_pod
from uprime.core.reynolds_stress import compute_reynolds_stresses

# Time series are [ny, nx, Nt]; a 2D field is [ny, nx].
U = U0 + phi[:, :, np.newaxis] * a
V = np.zeros_like(U)
W = None                                   # None means planar (2D2C) data

stresses, k = compute_reynolds_stresses(U, V, W)
stress_error = np.max(np.abs(stresses["uu"] - phi**2))

pod = compute_pod(U, V, W, n_modes=5)
first_mode_energy = float(pod["energy_frac"][0])
```

With one mode in the field, `<u'u'>` matches `phi**2` to roundoff and the first
POD mode holds all the fluctuation energy:

```
Largest error in <u'u'>:     1.665e-16 (m/s)^2
Energy in the first POD mode: 100.0000 %
```

---

## Reference by task

Shapes are written as `[ny, nx]` / `[ny, nx, Nt]`. "dict with keys" lists every
key the function actually produces.

### Loading data

**`loader.load_dataset(file_list, progress_callback=None)`**
Load a list of DaVis Tecplot `.dat` snapshots into one dataset.
Files are sorted by name, so snapshot order follows filename order; the first
file sets the grid, units and validity mask. `progress_callback` is called with
an int percentage. Raises `ValueError` on an empty list.
→ the dataset dict above.

**`loader.parse_header(filepath)`**
Read the Tecplot header and locate its columns.
→ dict with keys `variables`, `nx`, `ny`, `n_skip`, `col_x`, `col_y`, `col_u`,
`col_v`, `col_w`, `col_vort`, `col_valid`, `has_valid`, `is_stereo`,
`has_vort`, `x_unit`, `vel_unit`, `xy_to_mm`, `vel_to_ms`.
Raises `ValueError` if the ZONE line has no `I=`/`J=`.

**`loader.load_grid(filepath, header)`**
Read the coordinate grid, flipped so row 0 is the smallest `y`, scaled to mm.
→ `(x, y)`, each `[ny, nx]`.

**`loader.load_single_file(filepath, header)`**
Read one snapshot.
→ `(u, v, w, valid, vort)` — `u`, `v` are `[ny, nx]` m/s; `w` is `[ny, nx]` or
`None`; `valid` is `[ny, nx]` bool; `vort` is `[ny, nx]` or `None`. The mask is
applied to `vort` only; `u`/`v`/`w` keep raw values.

**`loader.estimate_dataset_size(file_list, header, stride=1)`**
Estimated float32 footprint. → int, bytes.

**`loader.cleanup_memmap(dataset)`** → `None`. Deletes memmap temp files.

**`loader.sweep_orphan_memmaps()`** → int, bytes freed.

**`mat_loader.detect_variables(filepath)`**
Inspect a `.mat` file's variables without reading the big arrays.
→ `(var_info, is_v73)` — dict of per-variable metadata, and whether the file is
MATLAB v7.3 (HDF5).

**`mat_loader.detect_u_candidate(var_info)`** → `(name, info)` or `(None, None)`.

**`mat_loader.auto_detect_mapping(var_info)`** → `(mapping, is_complete)`.

**`mat_loader.load_mat_dataset(filepath, mapping, progress_callback=None, status_callback=None, snap_start=0, snap_end=None, snap_step=1, dtype=np.float32, mask_convention='valid')`**
Load a `.mat` file using a confirmed variable mapping. `mask_convention` is
`'valid'` (1 = valid) or the inverted convention.
→ `(dataset, status_str)`.

**`davis_io.davis_snapshot_count(path_or_paths)`** → int.

**`davis_io.load_davis(paths, subset=None, progress_callback=None)`**
Load DaVis `.set`/`.vc7`/`.vec` input. Requires `lvpyio` (see
[Limits](#limits)). → dataset dict.

**`dataset_utils.get_masked(dataset, key)`**
Masked copy of `"U"`, `"V"` or `"W"`.
→ `[ny, nx, Nt]` in the stored dtype, or `None` if the field is `None`.

### Transforms and masks

All four transforms modify the dataset **in place and return `None`**. They
rewrite `x`, `y` and the velocity components together, so the dataset stays
self-consistent.

**`transform.apply_rotation(dataset, angle_deg, method='linear', chunk_size=200, progress_callback=None)`**
Rotate the field, interpolating onto the rotated grid. Points outside the
original domain become NaN. → `None`.

**`transform.apply_shift(dataset, dx_mm, dy_mm)`** Shift the origin. → `None`.

**`transform.apply_mirror_x(dataset)`**
Mirror left-right: negates `x` and `U`, then flips column order so `x` stays
ascending. → `None`.

**`transform.apply_mirror_y(dataset)`**
Mirror top-bottom: negates `y` and `V`, then flips row order. → `None`.

**`transform.transform_status_string(dataset)`**
Human-readable summary of the transforms applied. → str (empty if none).

### Mean flow and Reynolds stresses

**`mean_convergence_core.compute_mean_fields(dataset)`**
Time-averaged velocity fields.
→ dict with keys `U_mean`, `V_mean`, `W_mean` (`None` if planar), `speed`,
each `[ny, nx]`.

**`mean_convergence_core.compute_convergence(dataset, i_pt, j_pt, kernel)`**
Running statistical convergence at one point (Welford).
→ dict with keys `N_total`, `components`, `stats`.

**`mean_convergence_core.find_convergence_n(q_arr, scale, threshold)`**
First sample count where the running value stays inside `threshold`.
→ int, or `None` if it never settles.

**`reynolds_stress.compute_reynolds_stresses(U, V, W)`**
All available Reynolds stress components.
→ **a tuple** `(stresses, k)`. `stresses` is a dict with keys `uu`, `vv`, `uv`,
`ww`, `uw`, `vw`, each `[ny, nx]`, with `ww`/`uw`/`vw` set to `None` when `W`
is `None`. `k` is `[ny, nx]`, equal to `0.5*(uu+vv+ww)` or `0.5*(uu+vv)` for
planar data.

**`reynolds_stress.compute_reynolds_stress_std(U, V, W)`**
Standard deviation of the instantaneous stress products.
→ dict with the same keys as `stresses`.

**`reynolds_stress.compute_tke_std(U, V, W, mode='2d')`** → `[ny, nx]`.

**`reynolds_stress.extract_line_profile(field, x, y, x0, y0, x1, y1, mode='free', avg_band=0)`**
Sample a 2D field along a line. `mode` is `'free'`, `'horizontal'` or
`'vertical'`; `avg_band` averages over a band of rows/columns.
→ `(vals, dist, xpts, ypts)`, all 1D arrays.

**`line_sample.sample_along_line(x, y, field, p0, p1)`**
Free point-to-point sampling between two coordinates.
→ `(s, values)` — distance along the line in mm, and the sampled values.

**`anisotropy.compute_reynolds_tensor(U, V, W)`** → `(R, k)`.

**`anisotropy.compute_anisotropy_tensor(R, k)`** → `b`, the normalised
anisotropy tensor.

**`anisotropy.compute_invariants(b)`**
→ `(-II, III)`, the second and third invariants for the Lumley triangle.

**`anisotropy.compute_invariants_fast(b)`**
Vectorised equivalent of `compute_invariants`. → `(-II, III)`.

**`anisotropy.compute_barycentric(b)`** → `(C1c, C2c, C3c, RGB)`.

**`anisotropy.points_near_line(x, y, x0, y0, x1, y1)`** → `(rows, cols, dist)`.

### TKE budget

**`tke_budget.compute_tke_budget(U, V, W, x, y, mask=None, smooth_kernel=3, compute_dkdt=False)`**
Turbulent kinetic energy and its budget terms.
→ dict with keys `k`, `P` (production), `C` (convection), `D` (diffusion),
`R` (residual) and `dkdt`, each `[ny, nx]`. `dkdt` is only computed when
`compute_dkdt=True`.

### Spectra

**`spectra.compute_spectra(**kw)`**
Dispatcher used by the GUI worker. `kw["tab"]` selects the branch:
`"spatial"`, `"3d_spatial"`, `"temporal"` or `"st"`.
→ dict always carrying `tab`, plus `k`/`psds`/`direction` (spatial),
`result`/`roi`/`mask_pct` (3d_spatial), `type`/`freq`/`psds`/`title`
(temporal), or `k`/`f`/`psds`/`direction` (st). Raises `ValueError` on an
unknown tab, or when the 3-D ROI has too few valid vectors.

**`spectral.psd_at_point(U, V, W, row, col, fs, nperseg=None, noverlap=None)`**
Welch temporal PSD at one grid point. → `(freq, psds)`.

**`spectral.psd_in_region(U, V, W, x, y, x0, x1, y0, y1, fs, nperseg=None, noverlap=None)`**
Temporal PSD averaged over a rectangle.
→ `(freq, psd_avg, n_points)`.

**`spectral.nearest_grid_point(x, y, x_click, y_click)`** → `(row, col)`.

**`spatial_spectra.spatial_psd_line(U, V, W, x, y, x0, y0, x1, y1, direction, avg_band, nperseg, noverlap, subtract_spatial_mean)`**
Spatial PSD along a line, averaged over snapshots. → `(k, psds)`.

**`spatial_spectra.spatial_psd_roi(U, V, W, x, y, x0, x1, y0, y1, nperseg, noverlap, subtract_spatial_mean)`**
→ `(results, n_lines)`.

**`spatial_spectra_fft.compute_3d_spectra(U, V, W, Lx, Ly, Lz)`**
FFTW-based 3-D spectrum. → `(k_bin_centers, spectrum_3d)`.

**`spatial_spectra_fft.compute_1d_spectra(U, V, W, Lx, Ly, Lz)`**
→ dict with keys `kx`, `ky`, `kz` and `u_kx`, `u_ky`, `u_kz`, `v_kx`, `v_ky`,
`v_kz`, `w_kx`, `w_ky`, `w_kz`.

**`spatial_spectra_fft.subtract_temporal_mean(U, V, W)`** → `(U_f, V_f, W_f)`.

**`spatial_spectra_fft.compute_spectra_from_fluctuations(U_fluct, V_fluct, W_fluct, Lx, Ly, Lz)`**
→ dict, the combined 3-D and 1-D spectra.

**`spatial_spectra_fft.create_fft_arrays_and_plan(nz, ny, nx)`**
→ `(input_array, output_array, fft_plan)`.

**`spatiotemporal_spectra.compute_st_spectra(U, V, W, x, y, x0, y0, x1, y1, direction, avg_band, fs)`**
E(k, f) along a line. → `(k, f, psds)`.

### Correlations

**`two_point_corr.get_fluctuations(U, V, W)`**
Subtract the time mean from each component. → `(Up, Vp, Wp)`, `Wp` is `None`
when `W` is.

**`two_point_corr.select_component(Up, Vp, Wp, component)`**
Pick `'uu'`, `'vv'` or `'ww'`. → `[ny, nx, Nt]`. Raises `ValueError` for an
unknown name, or for `'ww'` on planar data.

**`two_point_corr.nearest_grid_point(x, y, x_click, y_click)`** → `(row, col)`.

**`two_point_corr.compute_spatial_correlation_point(U, V, W, row, col, component='uu', use_kernel=False)`**
R(x_ref, x) about one point; `use_kernel` averages over a 3×3 patch.
→ `(R_norm, R_x, R_y)` — the 2D map and the two 1D slices.

**`two_point_corr.compute_spatial_scales_point(R_norm, row, col, x, y, method='zero_crossing')`**
→ `(Lx, Ly, extras_x, extras_y)`.

**`two_point_corr.compute_spatial_correlation_roi(U, V, W, x, y, x0, x1, y0, y1, component='uu')`**
Spatially averaged correlation over a rectangle.
→ `(delta_x, R_x, delta_y, R_y, Lx, Ly)`.

**`two_point_corr.compute_temporal_correlation(U, V, W, row, col, component='uu', use_kernel=False, roi_coords=None, x=None, y=None, max_lag_fraction=0.5)`**
FFT-based autocorrelation R(tau). → `(R_tau, lags)`.

**`two_point_corr.compute_length_scale(R_1d, spacing, method='zero_crossing')`**
Integral scale from a normalised correlation curve. `method` is
`'zero_crossing'`, `'exp_fit'`, `'one_over_e'` or `'domain'`.
→ `(L, extras)` — `L` is a float (NaN if the method cannot be applied, e.g. no
zero crossing), `extras` a dict with keys `cutoff_idx`, `fit_L`, `fit_r`,
`fit_R`, `cumulative`, `r_axis`, `no_crossing`.

**`two_point_corr.compute_time_scales(R_tau, lags, dt, method='zero_crossing')`**
→ `(T, extras)`, same shape of result in seconds.

### POD and DMD

**`pod.compute_pod(U, V, W, n_modes=25)`**
Snapshot POD of the fluctuating field.
→ dict with keys `eigenvalues`, `modes`, `coeffs`, `energy_frac`,
`cumul_energy`, `mask`, `n_modes`, `Nc`, `mean_U`, `mean_V`, `mean_W`.
`energy_frac` is the per-mode energy fraction, `cumul_energy` its cumulative
sum.

**`pod.reconstruct_snapshot(pod_result, snapshot_idx, n_modes)`**
Rebuild one snapshot from the leading modes. → `(U_rec, V_rec, W_rec)`.

**`dmd.build_snapshot_matrix(U, V, W, component='stacked', mask=None)`**
Flatten a time series into a snapshot matrix. → `(X, n_per)`.

**`dmd.compute_dmd(X, rank=None)`**
Dynamic mode decomposition.
→ dict with keys `modes`, `eigenvalues`, `amplitudes`, `growth_rates`,
`frequencies`, `rank`. `frequencies` and `growth_rates` are in discrete
(per-snapshot) units.

**`dmd.scale_to_physical(dmd_result, fs)`**
Add physical-unit copies. → the same dict plus `frequencies_hz` and
`growth_rates_phys`.

**`dmd.get_mode_components(mode_flat, n_per, n_components, ny, nx, mask)`**
Unflatten one mode. → list of `[ny, nx]` arrays, one per component.

### Vortex identification

**`vortex_id.compute_gradients(U, V, x, y, mask)`**
→ dict with keys `dudx`, `dudy`, `dvdx`, `dvdy`.

**`vortex_id.compute_vortex_fields(grads)`**
Vortex criteria from the gradient tensor.
→ dict with keys `omega` (vorticity), `Q`, `lambda_ci`, `lambda2`.

**`vortex_id.compute_gamma(U, V, x, y, S=2)`**
Graftieaux Γ₁/Γ₂ over a kernel of half-width `S`. → `(gamma1, gamma2)`.

**`vortex_id.detect_vortices(field, omega, x, y, threshold, sign_filter='all', min_area_mm2=1.0, frame=0)`**
Threshold a criterion field and label blobs. `sign_filter` is `'all'`,
`'positive'` or `'negative'`.
→ list of dicts, each with keys `id`, `x_center`, `y_center`, `area_mm2`,
`sign`, `circulation`, `aspect_ratio`, `frame`.

**`vortex_id.compute_spatial_probability(field, threshold, sign_filter='all')`**
Fraction of snapshots exceeding the threshold at each point. → `[ny, nx]`.

### Probability analysis

**`probability.compute_direction_probability(field, mask_2d, deadband=0.0, chunk=200, progress_cb=None, region=None)`**
Forward/reverse/indeterminate flow fractions.
→ dict with keys `p_forward`, `p_reverse`, `p_indeterminate`, `n_valid`,
`deadband`, `region`.

**`probability.estimate_bin_edges(field, mask_2d, region, nbins=101, stride=10, robust=True)`**
Histogram edges from a strided sample. → 1D array of length `nbins + 1`.

**`probability.accumulate_histogram(field, mask_2d, region, bin_edges, chunk=200, progress_cb=None)`**
Chunked histogram over the time axis. → `(counts, n_valid)`.

**`probability.histogram_stats(counts, bin_edges)`**
→ dict with keys `n`, `mean`, `std`, `skewness`, `kurtosis`, `min`, `max`.
All NaN with `n = 0` for an empty histogram.

**`probability.extract_space_time(field, x, y, x0, y0, x1, y1, direction, avg_band=0)`**
Space-time map along a line. → `(q_st, s_mm, info)`.

**`probability.binarize_space_time(q_st, deadband=0.0)`**
→ array of −1 / 0 / +1 states.

**`probability.compute_quadrant(field_a, field_b, mask_2d, hole=0.0, region=None, joint_bins=101, chunk=200, progress_cb=None)`**
Quadrant analysis of two fluctuation fields.
→ dict with keys `time_frac`, `stress_frac`, `joint_hist`, `edges_a`,
`edges_b`, `a_rms`, `b_rms`, `hole`, `region`.

**`probability.quadrant_hole_sweep(field_a, field_b, mask_2d, region, holes=None, chunk=200, progress_cb=None)`**
→ `(stress_frac, holes)`.

### Export

**`export.export_2d_tecplot(filepath, x, y, fields, field_names, settings_info, fmt='%.6e', nan_repr=None)`**
The one canonical 2D writer: a `#` settings header, `VARIABLES`, a
`ZONE T=..., I=NX, J=NY, F=POINT` line, then rows in POINT order. → `None`.

**`export.export_line_csv(filepath, dist, xpts, ypts, mean_dict, std_dict, settings_info)`**
1D line profile with a settings header. → `None`.

**`export.export_spectra_csv(filepath, freq, psd_dict, settings_info)`** → `None`.

**`vortex_id.export_vortex_csv(vortex_list, filepath)`**
Write detected vortices. → `None`.

**`case_io.read_case_file(path)`**
Read a uPrime export back in, for cross-case comparison.
→ a `FileRecord` dataclass with fields `source_file`, `path`, `is_2d`,
`source_module`, `x_type` and `series`.

**`case_io.detect_quantity_key(col, meta=None)`**,
**`case_io.component_key_from_varname(name)`**,
**`case_io.is_signed_component(key)`**
Helpers that map exported column names onto uPrime quantity keys.

---

## Limits

**DaVis input needs `lvpyio`.** `davis_io.load_davis` requires LaVision's
`lvpyio`, which is separately licensed, is **not** GPL, and is not a dependency
of uPrime. `davis_io` guards its own import, and `davis_io.HAS_LVPYIO` reports
whether it is available. See `requirements-lvpyio.txt`.

**Some modules are internal.** `uprime.core.workers` and
`uprime.core.case_registry` exist to support the GUI — `workers` wraps
computations for background QThreads and `case_registry` is process-global
mutable state behind the Compare Cases window. Neither is part of the public
API; call the computation functions directly instead.

**Functions print to the console.** Several routines report progress or
diagnostics with `print` rather than the `logging` module: the loader prints
detected units and warnings for unreadable files, `compute_pod` prints
reconstruction diagnostics, and the FFT spectra branch prints ROI and mask
statistics. In a batch script this goes to stdout and cannot be silenced
through an API argument. Where a `progress_callback` or `progress_cb` argument
exists, it is the supported way to observe progress.

**`pyfftw` is required.** `spatial_spectra_fft` imports it at module level. It
is a hard dependency of the package, so a normal install always has it.

### Known inconsistencies

These are real behaviours of the current code, noted so scripts do not rely on
the tidier version:

- **The `.dat` loader does not NaN-mask velocities at load time.** `load_dataset`
  stores raw `U`/`V`/`W` and leaves masking to `get_masked`, while
  `load_mat_dataset` and `load_davis` apply `U[~mask] = np.nan` during the
  load. Always go through `get_masked` and the distinction stops mattering.
  (The `loader.py` module docstring still claims invalid vectors are set to
  NaN; that is true of vorticity only.)
- **`has_vort` can be `True` while `vort` is `None`.** The memory-mapped branch
  of `load_dataset` never fills a vorticity array, so for datasets above 4 GB
  the flag is set from the header but the data is absent. Test `vort is not
  None`, not `has_vort`.
- **The validity mask comes from the first snapshot only.** `load_dataset`
  takes `mask_2d` from file index 0 and applies it to the whole series.
