"""
core/davis_io.py
----------------
OPTIONAL, guarded LaVision DaVis loader via lvpyio.

DaVis input is .vc7/.vec vector files ONLY (one field per file); a time series
is a multi-file selection stacked into the snapshot axis, exactly like the .dat
loader. .set files and images (.im7/.imx) are rejected.

lvpyio is licensed "Free To Use But Restricted" and is NOT GPL-compatible, so
it must remain an optional dependency that the GPL core never hard-imports and
that is never bundled in a public distribution. This is the ONLY module in the
project that touches lvpyio; everything else goes through ``HAS_LVPYIO`` and
``load_davis``.

The returned dataset dict is byte-for-byte contract-compatible with
``core.loader.load_dataset`` / ``core.mat_loader.load_mat_dataset`` so DaVis
data plugs into the pipeline with zero downstream changes:

    x, y          : (Ny, Nx) float64, physical coords in mm, row 0 = smallest y
    U, V, W       : (Ny, Nx, Nt) float32; W is None for 2D2C; invalid -> NaN
    MASK / valid  : (Ny, Nx) bool, True = valid  (uPrime convention)
    is_stereo     : bool
    Nt, nx, ny    : ints
    dx, dy        : float, median grid spacing in mm
    files, header, _memmap_path, ...

API NOTES (verified against lvpyio 1.4.0 installed in this environment):
  * lvpyio.read_buffer(path)   -> single Buffer (.vc7 / .vec).
  * Buffer.as_masked_array()   -> structured masked ndarray, dtype
                                  vec2c ('u','v') or vec3c ('u','v','w');
                                  values are SCALED to physical velocity units;
                                  masked entries = invalid vectors.
  * Buffer.frames[0].scales.x/.y  -> Scale(slope, offset, unit, ...).
  * Buffer.frames[0].grid.x/.y    -> Grid step (pixels per vector node).
  * Buffer.frames[0].is_3c        -> 2D2C vs 2D3C.
  * physical coord = (index + 0.5) * grid * scale.slope + scale.offset
    (lvpyio's own lvpyio.types.plot_utils.scaled_coordinate).
"""

import os
import tempfile

import numpy as np

# --- Guarded, optional import. The GPL core must never hard-import lvpyio. ---
try:
    import lvpyio as _lv
except ImportError:
    try:
        # lvpyio_wrapped is the alternative distribution that ships its own Qt,
        # used when lvpyio's bundled Qt clashes with PyQt6 (see build recipe).
        import lvpyio_wrapped as _lv
    except ImportError:
        _lv = None

HAS_LVPYIO = _lv is not None

# Reuse the same >4GB memmap threshold as the .dat/.mat path.
SIZE_THRESHOLD = 4 * 1024 ** 3

# Extensions we explicitly refuse (images, not vector fields).
_IMAGE_EXTS = {".im7", ".imx", ".im8"}


# --------------------------------------------------------------------------- #
# Unit auto-detection — identical convention to core.loader.parse_header
# --------------------------------------------------------------------------- #

def _xy_to_mm_factor(unit):
    """Coordinate unit string -> factor to millimetres (uPrime convention)."""
    u = (unit or "").lower()
    if "mm" in u:
        return 1.0
    if "m" in u:            # metres
        return 1000.0
    return 1.0              # pixel / unknown -> leave as-is, treat as mm


def _vel_to_ms_factor(unit):
    """Velocity unit string -> factor to m/s (uPrime convention)."""
    u = (unit or "").lower()
    if "mm/s" in u:
        return 0.001
    if "m/s" in u:
        return 1.0
    return 1.0              # unknown -> leave as-is


# --------------------------------------------------------------------------- #
# Buffer -> arrays
# --------------------------------------------------------------------------- #

def _buffer_geometry(buffer):
    """Return (x_1d_mm, y_1d_mm, flip_x, flip_y, is_stereo) from a buffer.

    Uses lvpyio's own coordinate formula. DaVis stores rows top-down (row 0 =
    largest y); we detect that and flag a row flip so the result matches
    uPrime's convention (row 0 = smallest y), like core.loader.load_grid.
    """
    frame = buffer.frames[0]
    ny, nx = frame.shape
    sc = frame.scales
    grid = frame.grid

    xy_f = _xy_to_mm_factor(sc.x.unit)
    # physical coordinate = (index + 0.5) * grid * slope + offset  (lvpyio)
    x = ((np.arange(nx) + 0.5) * grid.x * sc.x.slope + sc.x.offset) * xy_f
    y = ((np.arange(ny) + 0.5) * grid.y * sc.y.slope + sc.y.offset) \
        * _xy_to_mm_factor(sc.y.unit)

    flip_x = x.size > 1 and x[0] > x[-1]
    flip_y = y.size > 1 and y[0] > y[-1]      # DaVis top-down -> usually True
    if flip_x:
        x = x[::-1]
    if flip_y:
        y = y[::-1]

    is_stereo = bool(getattr(frame, "is_3c", False))
    return x.astype(np.float64), y.astype(np.float64), flip_x, flip_y, is_stereo


def _buffer_components(buffer, flip_x, flip_y, vel_f):
    """Return (u, v, w_or_None, valid) as (Ny, Nx) float32/bool from a buffer.

    as_masked_array() gives SCALED physical velocities; masked = invalid.
    Orientation is corrected to match the geometry (flip rows/cols) so every
    array lines up with the x/y axes.
    """
    ma = buffer.as_masked_array()
    names = ma.dtype.names or ()

    def _get(field):
        data = np.asarray(ma[field].filled(np.nan), dtype=np.float32) * vel_f
        if flip_y:
            data = data[::-1, :]
        if flip_x:
            data = data[:, ::-1]
        return data

    u = _get("u")
    v = _get("v")
    w = _get("w") if "w" in names else None

    # masked = invalid -> valid mask is the inverse
    invalid = np.ma.getmaskarray(ma["u"])
    if "v" in names:
        invalid = invalid | np.ma.getmaskarray(ma["v"])
    valid = ~invalid
    if flip_y:
        valid = valid[::-1, :]
    if flip_x:
        valid = valid[:, ::-1]
    return u, v, w, valid.astype(bool)


# --------------------------------------------------------------------------- #
# Set/Buffer resolution
# --------------------------------------------------------------------------- #

_VECTOR_EXTS = {".vc7", ".vec"}
# Absolute tolerance (mm) for judging two coordinate axes "the same grid".
_AXIS_ATOL = 1e-6


def _ext_of(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == "" and os.path.isdir(path):
        return ".set"          # a DaVis .set may be a directory
    return ext


def _natural_key(path):
    """Natural/numeric sort key on the file name (B0001 < B0002 < ... < B0010)."""
    import re
    base = os.path.basename(path)
    return [int(tok) if tok.isdigit() else tok.lower()
            for tok in re.split(r"(\d+)", base)]


def _sorted_vector_files(path_or_paths):
    """Return the selection as a naturally-sorted list of .vc7 / .vec paths.

    DaVis input is .vc7/.vec vector files ONLY (one field per file). A .set,
    image files (.im7/.imx), or any other type are rejected with a clear
    message. Sorting is natural/numeric so B0001, B0002, ... B0010 order
    correctly (the temporal order of a DaVis time series).
    """
    paths = list(path_or_paths) if isinstance(path_or_paths, (list, tuple)) \
        else [path_or_paths]
    if not paths:
        raise ValueError("No DaVis file selected.")

    exts = [_ext_of(p) for p in paths]
    if any(e == ".set" or e in _IMAGE_EXTS for e in exts):
        raise ValueError(
            "uPrime reads DaVis .vc7/.vec vector files; export or convert "
            ".set to .vc7.")
    other = [p for p, e in zip(paths, exts) if e not in _VECTOR_EXTS]
    if other:
        bad = sorted({_ext_of(p) or "(none)" for p in other})
        raise ValueError(
            f"Unsupported DaVis file type(s): {', '.join(bad)}. "
            "Expected .vc7 / .vec vector files.")

    return sorted(paths, key=_natural_key)


def davis_snapshot_count(path_or_paths):
    """Number of snapshots a DaVis selection resolves to (one per .vc7/.vec file).

    Used by the GUI to offer subset selection and to reject a single-snapshot
    load before any heavy read, like the .mat loader.
    """
    if not HAS_LVPYIO:
        raise RuntimeError("lvpyio is not available.")
    return len(_sorted_vector_files(path_or_paths))


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def load_davis(paths, subset=None, progress_callback=None):
    """
    Load a DaVis .vc7/.vec time series into a uPrime dataset dict.

    Parameters
    ----------
    paths : str or list[str]
        One-or-more .vc7 / .vec vector files (one field per file), stacked into
        the snapshot axis exactly like the .dat loader stacks one file per
        snapshot. A single file is a single snapshot and is rejected. .set and
        image files are rejected.
    subset : list[int] or None
        0-based snapshot indices to load (from the existing subset dialog),
        applied over the naturally-sorted file list. None loads every snapshot.
    progress_callback : callable(int) or None  0-100 progress.

    Returns
    -------
    dataset : dict  Same contract as core.loader.load_dataset.
    """
    if not HAS_LVPYIO:
        raise RuntimeError(
            "DaVis loading requires lvpyio, which is not installed.")

    files = _sorted_vector_files(paths)          # one snapshot per file
    if len(files) < 2:
        raise ValueError(
            "Only one .vc7 / .vec file was selected — that is a single "
            "snapshot. Select multiple vector files (one per snapshot) so "
            "uPrime can compute turbulence statistics.")
    return _stack_buffers(
        fetch=lambda i: (_lv.read_buffer(files[i]), files[i]),
        n_total=len(files), subset=subset, source_files=files,
        check_grid=True, progress_callback=progress_callback)


def _stack_buffers(fetch, n_total, subset, source_files, check_grid,
                   progress_callback=None):
    """Stack per-snapshot buffers into a uPrime dataset dict.

    fetch(i) -> (buffer, source_name) reads snapshot i (lazily for files).
    When check_grid is True (multi-file case) every snapshot's grid size,
    coordinate axes and component set are verified against the first, aborting
    with a clear message naming the offending file rather than misaligning.
    """
    indices = list(range(n_total)) if subset is None else list(subset)
    indices = [i for i in indices if 0 <= i < n_total]
    if not indices:
        raise ValueError("No valid snapshots selected.")
    Nt = len(indices)

    # Geometry & metadata from the first selected snapshot.
    first_buf, _first_src = fetch(indices[0])
    x_1d, y_1d, flip_x, flip_y, is_stereo = _buffer_geometry(first_buf)
    ny, nx = first_buf.frames[0].shape
    vel_f = _vel_to_ms_factor(first_buf.frames[0].scales.i.unit)

    # Storage: in-memory or memmap over 4 GB (same strategy as .dat/.mat).
    est_size = ny * nx * Nt * (3 if is_stereo else 2) * 4
    use_memmap = est_size > SIZE_THRESHOLD
    tmp_path = None
    if use_memmap:
        tmp_path = os.path.join(tempfile.gettempdir(),
                                f"uprime_memmap_{os.getpid()}.bin")
        shape = (ny, nx, Nt)
        U = np.memmap(tmp_path + "_U", dtype="float32", mode="w+", shape=shape)
        V = np.memmap(tmp_path + "_V", dtype="float32", mode="w+", shape=shape)
        W = (np.memmap(tmp_path + "_W", dtype="float32", mode="w+", shape=shape)
             if is_stereo else None)
    else:
        U = np.empty((ny, nx, Nt), dtype=np.float32)
        V = np.empty((ny, nx, Nt), dtype=np.float32)
        W = np.empty((ny, nx, Nt), dtype=np.float32) if is_stereo else None
    U[:] = np.nan
    V[:] = np.nan
    if W is not None:
        W[:] = np.nan

    valid_all = np.ones((ny, nx), dtype=bool)
    loaded_files = []

    for out_i, snap in enumerate(indices):
        buf, src = (first_buf, _first_src) if out_i == 0 else fetch(snap)

        if check_grid and out_i > 0:
            gx, gy, _fx, _fy, g_stereo = _buffer_geometry(buf)
            bny, bnx = buf.frames[0].shape
            if (bny, bnx) != (ny, nx):
                raise ValueError(
                    f"Grid size mismatch: '{os.path.basename(src)}' is "
                    f"{bny}×{bnx}, but the first file is {ny}×{nx}. All "
                    ".vc7/.vec files must share the same grid.")
            if g_stereo != is_stereo:
                raise ValueError(
                    f"Component mismatch: '{os.path.basename(src)}' is "
                    f"{'2D3C' if g_stereo else '2D2C'}, but the first file is "
                    f"{'2D3C' if is_stereo else '2D2C'}. All files must have "
                    "the same velocity components.")
            if not (np.allclose(gx, x_1d, atol=_AXIS_ATOL)
                    and np.allclose(gy, y_1d, atol=_AXIS_ATOL)):
                raise ValueError(
                    f"Coordinate axes of '{os.path.basename(src)}' differ from "
                    "the first file. All .vc7/.vec files must share the same "
                    "coordinate grid.")

        u, v, w, valid = _buffer_components(buf, flip_x, flip_y, vel_f)
        # invalid -> NaN in-place (matches .dat/.mat contract)
        u = u.copy(); u[~valid] = np.nan
        v = v.copy(); v[~valid] = np.nan
        U[:, :, out_i] = u
        V[:, :, out_i] = v
        if W is not None and w is not None:
            w = w.copy(); w[~valid] = np.nan
            W[:, :, out_i] = w
        valid_all &= valid
        loaded_files.append(src)
        if progress_callback:
            progress_callback(int(100 * (out_i + 1) / Nt))

    if use_memmap:
        U.flush(); V.flush()
        if W is not None:
            W.flush()

    X, Y = np.meshgrid(x_1d, y_1d)
    dx = float(np.abs(np.median(np.diff(x_1d)))) if x_1d.size > 1 else 0.0
    dy = float(np.abs(np.median(np.diff(y_1d)))) if y_1d.size > 1 else 0.0

    header = {
        "nx": nx, "ny": ny,
        "is_stereo": is_stereo,
        "has_vort": False,
        "has_valid": True,
        "x_unit": "mm",
        "vel_unit": "m/s",
        "xy_to_mm": 1.0,
        "vel_to_ms": 1.0,
        "source": "davis",
    }

    files_key = source_files if len(source_files) == 1 else loaded_files
    return {
        "x": X, "y": Y,
        "U": U, "V": V, "W": W,
        "vort": None,
        "valid": valid_all,
        "valid_frac": valid_all.astype(np.float32),
        "MASK": valid_all,
        "MASK_LOADED": valid_all.copy(),
        "mask_active": True,
        "is_stereo": is_stereo,
        "has_vort": False,
        "Nt": Nt, "nx": nx, "ny": ny,
        "dx": dx, "dy": dy,
        "files": files_key,
        "header": header,
        "_memmap_path": tmp_path,
    }
