# core/mat_loader.py — uPrime MATLAB .mat file loader
#
# Copyright (C) 2025  CmdrRyder
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Loads PIV datasets from MATLAB .mat files.
One .mat file = full time series. Velocity arrays must have shape (Ny, Nx, N_snap).

Supports:
  - Classic .mat (v5/v6/v7.0) via scipy.io.loadmat  — one blocking read into RAM.
  - HDF5-based .mat v7.3 via h5py — velocity arrays read in chunks so the
    progress bar advances and peak temporary memory stays bounded.

HDF5 / MATLAB axis ordering note
---------------------------------
MATLAB stores arrays in Fortran (column-major) order; HDF5 stores them in C
(row-major) order.  A MATLAB array of logical size [Ny, Nx, Nt] is therefore
written to HDF5 with shape (Nt, Nx, Ny).  h5py reads the raw HDF5 shape, so
every array read via h5py is transposed (.T or equivalent) before use.
"""

import gc
import numpy as np

# Number of snapshots read from HDF5 per iteration.  Larger = fewer calls but
# bigger float64 temporary per pass.  100 keeps the spike under ~1–2 GB for
# typical PIV grid sizes; tune up if I/O is the bottleneck.
_CHUNK = 100

# Role → ordered candidate variable names (checked in order, case-insensitive)
_CANDIDATES = {
    "x":       ["x", "X", "xc", "xx"],
    "y":       ["y", "Y", "yc", "yy"],
    "u":       ["u", "U", "Vx", "vx"],
    "v":       ["v", "V", "Vy", "vy"],
    "w":       ["w", "W", "Vz", "vz"],
    "isValid": ["isValid", "isvalid", "mask", "valid", "IsValid"],
}

_REQUIRED = ("x", "y", "u", "v")
_OPTIONAL = ("w", "isValid")


def detect_variables(filepath):
    """
    Read the variable list from a .mat file without loading array data.

    Returns
    -------
    var_info : dict  {name: shape_tuple}  logical (MATLAB) shapes
    use_h5py : bool  True when the file is HDF5-based v7.3
    """
    try:
        import scipy.io
        info_list = scipy.io.whosmat(filepath)          # [(name, shape, dtype), ...]
        var_info  = {name: tuple(shape) for name, shape, _ in info_list}
        return var_info, False
    except NotImplementedError:
        # v7.3 HDF5-based .mat — scipy cannot handle it.
        # 'with' guarantees __exit__ (close + libhdf5 cache flush) runs before
        # this function returns, so a rejected-load's file ID cannot bleed into
        # the next detect_variables call via HDF5's internal metadata cache.
        import h5py
        var_info = {}
        with h5py.File(filepath, "r") as fh:
            for name, obj in fh.items():
                if isinstance(obj, h5py.Dataset):
                    # HDF5 stores MATLAB arrays with reversed axis order
                    var_info[name] = tuple(reversed(obj.shape))
        return var_info, True


def detect_u_candidate(var_info):
    """
    Return (name, shape) of the first u-candidate found in var_info by name match,
    regardless of dimensionality.  Returns (None, None) if none found.

    Used for the early snapshot-count check before any data is loaded.
    """
    lower_to_name = {k.lower(): k for k in var_info}
    for cand in _CANDIDATES["u"]:
        actual = lower_to_name.get(cand.lower())
        if actual is not None:
            return actual, var_info[actual]
    return None, None


def auto_detect_mapping(var_info):
    """
    Match roles to variable names using candidate lists and shape sanity checks.

    Shape rules:
      x, y     — 1-D or 2-D
      u, v, w  — strictly 3-D
      isValid  — strictly 3-D

    Returns
    -------
    mapping     : dict  {role: varname or None}
    is_complete : bool  True when x, y, u, v all resolved
    """
    # Build case-insensitive reverse lookup: lower_name → original_name
    lower_to_name = {k.lower(): k for k in var_info}

    mapping = {}
    for role, candidates in _CANDIDATES.items():
        matched = None
        for cand in candidates:
            actual = lower_to_name.get(cand.lower())
            if actual is None:
                continue
            shape = var_info[actual]
            ndim  = len(shape)
            if role in ("x", "y"):
                if ndim in (1, 2):
                    matched = actual
                    break
            elif role in ("u", "v", "w"):
                if ndim == 3:
                    matched = actual
                    break
            elif role == "isValid":
                if ndim == 3:
                    matched = actual
                    break
        mapping[role] = matched

    is_complete = all(mapping.get(r) for r in _REQUIRED)
    return mapping, is_complete


def load_mat_dataset(filepath, mapping, progress_callback=None, status_callback=None,
                     snap_start=0, snap_end=None, snap_step=1, dtype=np.float32,
                     mask_convention="valid"):
    """
    Load a .mat file using the resolved role → variable-name mapping.

    Parameters
    ----------
    filepath          : str  Path to the .mat file.
    mapping           : dict {role: varname or None}
    progress_callback : callable(int) or None
        Called with values 0–100 as the load progresses.
        For v7.3 (h5py) files the bar advances per chunk; for classic .mat
        files only a few discrete values are emitted (blocking read).
    status_callback   : callable(str) or None
        Called with a short human-readable status string after each read chunk.
        Intended for updating the main-window status bar during long loads.
    snap_start        : int  First snapshot index to load (0-based, inclusive).
    snap_end          : int or None  Last snapshot index to load (inclusive).
                        None means load to the last snapshot in the file.
    snap_step         : int  Stride between loaded snapshots (1 = every snapshot).

    Returns
    -------
    dataset    : dict  Same structure as core.loader.load_dataset output.
    status_str : str   Human-readable summary (shown in the status bar).
    """
    if progress_callback:
        progress_callback(0)
    if status_callback:
        status_callback("Opening file…")

    # --- Determine format ---
    use_h5py = False
    try:
        import scipy.io
        scipy.io.whosmat(filepath)   # raises NotImplementedError for v7.3
    except NotImplementedError:
        use_h5py = True

    if use_h5py:
        import h5py
        data = h5py.File(filepath, "r")
    else:
        import scipy.io
        mat  = scipy.io.loadmat(filepath)
        # Separate pass-through dict so we can del entries to release float64
        data = {k: v for k, v in mat.items() if not k.startswith("_")}
        del mat
        gc.collect()

    try:
        if use_h5py:
            # ----------------------------------------------------------------
            # h5py path: chunked reads, float32 output, bounded peak memory
            #
            # Standard MATLAB convention: u stored as (Ny, Nx, Nt).
            # HDF5 reverses all dims → (Nt, Nx, Ny).
            # Some files (CFD / ndgrid) use non-standard (Nx, Ny, Nt) order,
            # giving HDF5 shape (Nt, Ny, Nx).  We detect which convention is
            # in use by peeking at the x-coordinate array shape before any
            # large read, then choose the appropriate chunk transpose.
            # ----------------------------------------------------------------
            h5u     = data[mapping["u"]]
            Nt_file = h5u.shape[0]    # total snapshots in file
            Nx_h5   = h5u.shape[1]
            Ny_h5   = h5u.shape[2]

            # Peek at x-coord (small) to detect non-standard (Nx, Ny, Nt) storage.
            # Standard  → xpeek.shape == (Ny_h5, Nx_h5): HDF5 spatial dims are (Nx, Ny)
            # Non-std   → xpeek.shape == (Nx_h5, Ny_h5): HDF5 spatial dims are (Ny, Nx)
            _xpeek = np.array(data[mapping["x"]]).T.squeeze()
            if (_xpeek.ndim == 2 and Nx_h5 != Ny_h5
                    and _xpeek.shape == (Nx_h5, Ny_h5)):
                # Velocity stored as (Nx, Ny, Nt) in MATLAB → swap spatial dims
                Ny_h5, Nx_h5 = Nx_h5, Ny_h5
                _u_axes = (1, 2, 0)   # (chunk, Ny, Nx) → (Ny, Nx, chunk)
                if status_callback:
                    status_callback(
                        "Note: velocity arrays detected in (Nx, Ny, Nt) order; "
                        "transposing to standard (Ny, Nx, Nt).")
            else:
                _u_axes = (2, 1, 0)   # standard: (chunk, Nx, Ny) → (Ny, Nx, chunk)

            # Resolve subset indices (uniform stride → use slice, not fancy index)
            _snap_end    = snap_end if snap_end is not None else Nt_file - 1
            snap_indices = list(range(snap_start, _snap_end + 1, snap_step))
            Nt           = len(snap_indices)   # snapshots actually loaded

            # Bytes per snapshot for progress messages
            _bpf      = Nx_h5 * Ny_h5 * np.dtype(dtype).itemsize
            _total_gb = Nt * _bpf / 1e9

            # --- u ---
            # u arrays in .mat files are stored as (Ny, Nx, N_snap)
            # where Ny = number of rows along y, Nx = number of columns along x
            U = np.empty((Ny_h5, Nx_h5, Nt), dtype=dtype)
            for i in range(0, Nt, _CHUNK):
                j   = min(i + _CHUNK, Nt)
                s   = snap_indices[i]
                e   = snap_indices[j - 1] + 1   # exclusive HDF5 end
                U[:, :, i:j] = h5u[s:e:snap_step].transpose(*_u_axes).astype(dtype, copy=False)
                if progress_callback:
                    progress_callback(int(j / Nt * 40))
                if status_callback:
                    status_callback(
                        f"Reading u · {j:,} / {Nt:,} frames"
                        f"  ({j * _bpf / 1e9:.1f} / {_total_gb:.1f} GB)"
                    )

            # --- v ---
            h5v = data[mapping["v"]]
            V   = np.empty((Ny_h5, Nx_h5, Nt), dtype=dtype)
            for i in range(0, Nt, _CHUNK):
                j = min(i + _CHUNK, Nt)
                s = snap_indices[i]
                e = snap_indices[j - 1] + 1
                V[:, :, i:j] = h5v[s:e:snap_step].transpose(*_u_axes).astype(dtype, copy=False)
                if progress_callback:
                    progress_callback(40 + int(j / Nt * 25))
                if status_callback:
                    status_callback(
                        f"Reading v · {j:,} / {Nt:,} frames"
                        f"  ({j * _bpf / 1e9:.1f} / {_total_gb:.1f} GB)"
                    )

            Ny, Nx = Ny_h5, Nx_h5

            def _arr_small(name):
                """Load a small (coordinate) array via h5py — not for large 3-D fields."""
                return np.array(data[name]).T

        else:
            # ----------------------------------------------------------------
            # scipy path: blocking read — release float64 as soon as float32
            # copies are ready to cap peak RAM.
            # ----------------------------------------------------------------
            def _arr(name):
                return np.asarray(data[name])

            # Peek at x coord before the large read to detect non-standard storage
            _xpeek_s = _arr(mapping["x"]).squeeze() if mapping.get("x") else None

            if status_callback:
                status_callback("Reading u (blocking)…")
            U = _arr(mapping["u"]).squeeze().astype(dtype, copy=False)
            try:
                del data[mapping["u"]]
            except KeyError:
                pass
            gc.collect()

            if U.ndim != 3:
                raise ValueError(
                    f"Variable '{mapping['u']}' has shape {U.shape}; "
                    "expected strictly 3-D (Ny, Nx, N_snap)."
                )

            # Detect non-standard (Nx, Ny, Nt) storage: u.shape[0] matches x cols
            # u arrays in .mat files are stored as (Ny, Nx, N_snap)
            # where Ny = number of rows along y, Nx = number of columns along x
            _scipy_transposed = (
                _xpeek_s is not None
                and _xpeek_s.ndim == 2
                and U.shape[0] != U.shape[1]
                and U.shape[0] == _xpeek_s.shape[1]
                and U.shape[1] == _xpeek_s.shape[0]
            )
            if _scipy_transposed:
                U = np.ascontiguousarray(U.transpose(1, 0, 2))
                if status_callback:
                    status_callback(
                        "Note: velocity arrays detected in (Nx, Ny, Nt) order; "
                        "transposing to standard (Ny, Nx, Nt).")
            Ny, Nx, Nt_file = U.shape

            # Apply snapshot subset (slice is a view; ascontiguousarray copies
            # only the selected frames so the full-file float32 can be freed)
            _snap_end = snap_end if snap_end is not None else Nt_file - 1
            if snap_start != 0 or _snap_end != Nt_file - 1 or snap_step != 1:
                if status_callback:
                    status_callback(
                        f"Slicing u subset {snap_start}:{_snap_end + 1}:{snap_step}…"
                        "  (classic .mat: full array must be read before slicing)"
                    )
                U = np.ascontiguousarray(U[:, :, snap_start:_snap_end + 1:snap_step])
            Nt = U.shape[2]

            if status_callback:
                status_callback("Reading v…")
            V = _arr(mapping["v"]).squeeze().astype(dtype, copy=False)
            try:
                del data[mapping["v"]]
            except KeyError:
                pass
            gc.collect()
            if _scipy_transposed:
                V = np.ascontiguousarray(V.transpose(1, 0, 2))
            if snap_start != 0 or _snap_end != Nt_file - 1 or snap_step != 1:
                V = np.ascontiguousarray(V[:, :, snap_start:_snap_end + 1:snap_step])

            if progress_callback:
                progress_callback(70)

            def _arr_small(name):
                return _arr(name)

        # Shape sanity check
        if U.ndim != 3 or U.shape != (Ny, Nx, Nt):
            raise ValueError(
                f"Variable '{mapping['u']}' has shape {U.shape}; "
                "expected strictly 3-D (Ny, Nx, N_snap)."
            )

        if progress_callback:
            progress_callback(72)

        # --- Coordinates ---
        x_raw = _arr_small(mapping["x"]).squeeze()
        y_raw = _arr_small(mapping["y"]).squeeze()

        def _detect_meshgrid_orientation(arr, role):
            """Return 'as_is', 'transpose', or None (ambiguous) for a 2-D coord array.

            For x: correct orientation has x varying along axis=1 (columns).
            For y: correct orientation has y varying along axis=0 (rows).
            Requires a 10× std ratio to commit; returns None if ambiguous.
            """
            std0 = float(np.nanstd(arr, axis=0).mean())
            std1 = float(np.nanstd(arr, axis=1).mean())
            RATIO = 10.0
            if role == "x":
                if std1 > RATIO * std0:
                    return "as_is"
                if std0 > RATIO * std1:
                    return "transpose"
            else:  # "y"
                if std0 > RATIO * std1:
                    return "as_is"
                if std1 > RATIO * std0:
                    return "transpose"
            return None

        def _expand_coord(arr, role):
            """Return a (Ny, Nx) float64 coordinate grid from a 1-D or 2-D input.

            For 2-D arrays the orientation is determined first from the variance
            pattern in the data (x should vary along columns, y along rows), with
            shape comparison as a fallback when the variance signal is ambiguous.
            """
            name = mapping[role]
            if arr.ndim == 1:
                if role == "x":
                    if arr.size == Nx:
                        return np.tile(arr, (Ny, 1))
                    if arr.size == Ny:
                        return np.tile(arr.reshape(-1, 1), (1, Nx))
                else:  # "y"
                    if arr.size == Ny:
                        return np.tile(arr.reshape(-1, 1), (1, Nx))
                    if arr.size == Nx:
                        return np.tile(arr, (Ny, 1))
                raise ValueError(
                    f"1-D {role} variable '{name}' has {arr.size} elements "
                    f"but grid is {Ny}×{Nx}."
                )
            if arr.ndim == 2:
                orient = _detect_meshgrid_orientation(arr, role)
                if orient == "as_is":
                    arr_oriented = arr
                elif orient == "transpose":
                    if status_callback:
                        status_callback(
                            f"Note: {role}-coord '{name}' transposed "
                            f"(variance pattern indicates non-standard orientation).")
                    arr_oriented = arr.T
                else:
                    # Variance ambiguous → fall back to shape comparison
                    if arr.shape == (Ny, Nx):
                        arr_oriented = arr
                    elif arr.shape == (Nx, Ny):
                        if status_callback:
                            status_callback(
                                f"Note: meshgrid '{name}' was transposed "
                                f"(shape {(Nx, Ny)} detected) to match velocity orientation.")
                        arr_oriented = arr.T
                    else:
                        raise ValueError(
                            f"2-D {role} variable '{name}' has shape {arr.shape}; "
                            f"expected {(Ny, Nx)} or {(Nx, Ny)}.")
                if arr_oriented.shape != (Ny, Nx):
                    raise ValueError(
                        f"{role} array '{name}' could not be aligned to expected shape "
                        f"({Ny}, {Nx}); got {arr_oriented.shape} after orientation detection.")
                return arr_oriented.astype(np.float64)
            raise ValueError(
                f"{role} variable '{name}' has {arr.ndim}-D shape {arr.shape}; "
                f"expected 1-D or 2-D."
            )

        x = _expand_coord(x_raw, "x")
        y = _expand_coord(y_raw, "y")

        # Robust grid spacing: median of first-differences on the extracted
        # 1-D slices (median is robust against minor edge rounding; abs handles
        # axes where coordinates decrease).
        x_1d = x[0, :]
        y_1d = y[:, 0]

        if x_1d.size > 1:
            _dxv = np.diff(x_1d.astype(np.float64))
            dx   = float(np.abs(np.median(_dxv)))
            _dx_nonuniform = bool(dx > 0 and float(np.std(_dxv)) / dx > 0.01)
        else:
            dx = 0.0
            _dx_nonuniform = False

        if y_1d.size > 1:
            _dyv = np.diff(y_1d.astype(np.float64))
            dy   = float(np.abs(np.median(_dyv)))
            _dy_nonuniform = bool(dy > 0 and float(np.std(_dyv)) / dy > 0.01)
        else:
            dy = 0.0
            _dy_nonuniform = False

        if progress_callback:
            progress_callback(80)

        # --- Stereo (W) ---
        W         = None
        is_stereo = False
        if mapping.get("w"):
            try:
                if use_h5py:
                    h5w = data[mapping["w"]]
                    W   = np.empty((Ny, Nx, Nt), dtype=dtype)
                    for i in range(0, Nt, _CHUNK):
                        j = min(i + _CHUNK, Nt)
                        s = snap_indices[i]
                        e = snap_indices[j - 1] + 1
                        W[:, :, i:j] = h5w[s:e:snap_step].transpose(*_u_axes).astype(dtype, copy=False)
                        if progress_callback:
                            progress_callback(80 + int(j / Nt * 8))
                        if status_callback:
                            status_callback(f"Reading w · {j:,} / {Nt:,} frames")
                    is_stereo = True
                else:
                    if status_callback:
                        status_callback("Reading w…")
                    W_raw = _arr_small(mapping["w"]).squeeze().astype(dtype, copy=False)
                    try:
                        del data[mapping["w"]]
                    except KeyError:
                        pass
                    # Apply subset if needed (W has Nt_file snapshots like U did)
                    if _scipy_transposed and W_raw.ndim == 3:
                        W_raw = np.ascontiguousarray(W_raw.transpose(1, 0, 2))
                    if W_raw.ndim == 3 and W_raw.shape[2] == Nt_file and Nt != Nt_file:
                        W_raw = np.ascontiguousarray(
                            W_raw[:, :, snap_start:_snap_end + 1:snap_step])
                    if W_raw.shape == (Ny, Nx, Nt):
                        W         = W_raw
                        is_stereo = True
                    else:
                        print(
                            f"[mat_loader] W variable '{mapping['w']}' has shape "
                            f"{W_raw.shape} — expected ({Ny},{Nx},{Nt}), ignoring."
                        )
                        del W_raw
            except Exception as exc:
                print(f"[mat_loader] Could not load W: {exc}")

        if progress_callback:
            progress_callback(90)

        # --- Validity mask ---
        mask_2d = None
        if mapping.get("isValid"):
            try:
                if use_h5py:
                    # Accumulate the 2-D all-time mask in chunks to avoid
                    # materialising the full 3-D isValid field (can be several GB
                    # if stored as float64 in MATLAB).
                    h5valid = data[mapping["isValid"]]
                    mask_2d = np.ones((Ny, Nx), dtype=bool)
                    for i in range(0, Nt, _CHUNK):
                        j = min(i + _CHUNK, Nt)
                        s = snap_indices[i]
                        e = snap_indices[j - 1] + 1
                        # h5valid[s:e:step] shape: (j-i, dim1, dim2)
                        # After axis=0 reduction: (dim1, dim2).
                        # Standard (_u_axes=(2,1,0)): dims are (Nx, Ny) → need .T
                        # Non-std  (_u_axes=(1,2,0)): dims are (Ny, Nx) → no .T
                        chunk = h5valid[s:e:snap_step].astype(bool)
                        _plane = np.all(chunk, axis=0)
                        mask_2d &= (_plane.T if _u_axes == (2, 1, 0) else _plane)
                        del chunk
                        if progress_callback:
                            progress_callback(90 + int(j / Nt * 5))
                        if status_callback:
                            status_callback(f"Reading isValid · {j:,} / {Nt:,} frames")
                else:
                    if status_callback:
                        status_callback("Reading isValid…")
                    valid_raw = _arr_small(mapping["isValid"]).squeeze().astype(bool)
                    try:
                        del data[mapping["isValid"]]
                    except KeyError:
                        pass
                    gc.collect()
                    if valid_raw.ndim == 3 and valid_raw.shape[2] == Nt_file and Nt != Nt_file:
                        valid_raw = valid_raw[:, :, snap_start:_snap_end + 1:snap_step]
                    if valid_raw.shape == (Ny, Nx, Nt):
                        mask_2d = np.all(valid_raw, axis=2)
                        del valid_raw
                    elif valid_raw.shape == (Ny, Nx):
                        mask_2d = valid_raw
                    else:
                        print(
                            f"[mat_loader] isValid shape {valid_raw.shape} "
                            f"doesn't match ({Ny},{Nx},{Nt}) — using all-valid mask."
                        )
                        del valid_raw
            except Exception as exc:
                print(f"[mat_loader] Could not load isValid: {exc}")

        # Apply user-specified mask convention.
        # mask_2d raw: True where the variable value == 1 (no auto-detection).
        # "valid"   → True means valid (uPrime isValid convention, no change)
        # "invalid" → True means invalid; invert so True = valid for downstream code.
        if mask_2d is not None and mask_convention == "invalid":
            mask_2d = ~mask_2d

        if mask_2d is None:
            mask_2d = np.ones((Ny, Nx), dtype=bool)

        # Apply mask in-place: invalid points → NaN (matches .dat loader behaviour).
        # mask_2d is True=valid; ~mask_2d selects points to blank.
        if not np.all(mask_2d):
            U[~mask_2d, :] = np.nan
            V[~mask_2d, :] = np.nan
            if W is not None:
                W[~mask_2d, :] = np.nan

        # Safety check: warn if the resulting invalid fraction looks extreme.
        # This does not auto-fix; the user can reload with the other convention.
        if mapping.get("isValid") and status_callback:
            _nan_frac = float(np.isnan(U[:, :, Nt // 2]).mean())
            _mask_name = mapping["isValid"]
            if _nan_frac > 0.95:
                status_callback(
                    f"Warning: mask '{_mask_name}' marks {_nan_frac * 100:.0f}% of "
                    "the field as invalid. If this looks wrong, reload with the "
                    "other mask convention in the load dialog."
                )
            elif _nan_frac < 0.05 and not np.all(mask_2d):
                status_callback(
                    f"Warning: mask '{_mask_name}' marks only {_nan_frac * 100:.1f}% "
                    "of the field as invalid. If this looks wrong, reload with "
                    "the other mask convention in the load dialog."
                )

        # Release data dict and force a GC cycle before building the dataset
        # object so transient float64 buffers are reclaimed first.
        if not use_h5py:
            data.clear()
        gc.collect()

    finally:
        if use_h5py and hasattr(data, "close"):
            data.close()

    if status_callback:
        status_callback("Building dataset…")
    if progress_callback:
        progress_callback(100)

    # --- Status string shown in the main-window status bar ---
    if snap_start == 0 and _snap_end == Nt_file - 1 and snap_step == 1:
        n_str = f"N={Nt}"
    else:
        step_s = f" step {snap_step}" if snap_step != 1 else ""
        n_str  = f"N={Nt} of {Nt_file} (range {snap_start}:{_snap_end}{step_s})"
    status_str = (
        f"Loaded .mat: x={mapping['x']}, y={mapping['y']}, "
        f"u={mapping['u']}, v={mapping['v']}, "
        f"w={mapping.get('w') or '(none)'}, "
        f"isValid={mapping.get('isValid') or '(none)'}, "
        f"{n_str}, dtype={np.dtype(dtype).name}"
    )
    if _dx_nonuniform or _dy_nonuniform:
        status_str += (
            "  ⚠ Non-uniform grid spacing (>1% variation). "
            "FFT and Welch spectra assume uniform grids and may give incorrect results."
        )
    print(f"[mat_loader] {status_str}")

    header = {
        "nx":        Nx,
        "ny":        Ny,
        "is_stereo": is_stereo,
        "has_vort":  False,
        "has_valid": mapping.get("isValid") is not None,
        "x_unit":    "mm",
        "vel_unit":  "m/s",
        "xy_to_mm":  1.0,
        "vel_to_ms": 1.0,
        "source":    "mat",
        "mapping":   dict(mapping),
    }

    dataset = {
        "x":           x,
        "y":           y,
        "U":           U,
        "V":           V,
        "W":           W,
        "vort":        None,
        "valid":       mask_2d,
        "valid_frac":  mask_2d.astype(np.float32),
        "MASK":        mask_2d,
        "MASK_LOADED": mask_2d.copy(),
        "mask_active": True,
        "is_stereo":   is_stereo,
        "has_vort":    False,
        "Nt":          Nt,
        "nx":          Nx,
        "ny":          Ny,
        "dx":          dx,
        "dy":          dy,
        "files":       [filepath],
        "header":      header,
        "_memmap_path": None,
        "_mat_status": status_str,   # consumed by _on_load_finished, then removed
    }

    return dataset, status_str
