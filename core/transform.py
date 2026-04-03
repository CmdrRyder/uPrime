"""
core/transform.py
-----------------
In-place coordinate and velocity transformations for PIV datasets.

Operations
----------
apply_rotation(dataset, angle_deg, method, chunk_size)
    Rotate the grid and interpolate velocity onto a new regular grid with the
    same dx/dy spacing, clipped to the original axis-aligned extent (Option A).
    Rotation is about the data centroid.
    In-place -- original arrays are overwritten. No second copy retained.

apply_shift(dataset, dx_mm, dy_mm)
    Translate the coordinate origin. Pure subtraction, zero extra memory.

Both functions append an entry to dataset["transform_log"] for status display.

Memory strategy for rotation
-----------------------------
Peak extra memory = one [ny, nx, chunk_size] float32 array for the velocity
buffer during chunked interpolation (~100 MB for chunk_size=200).
x, y are [ny, nx] -- negligible.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------

def apply_rotation(dataset, angle_deg, method="linear", chunk_size=200,
                   progress_callback=None):
    """
    Rotate the PIV dataset in-place.

    Parameters
    ----------
    dataset      : dict -- the main dataset dict (modified in-place)
    angle_deg    : float -- rotation angle in degrees (positive = CCW).
                   This is the CORRECTION angle: the amount to rotate the data
                   so that a tilted feature becomes horizontal/vertical.
    method       : "linear" or "cubic" -- interpolation method
    chunk_size   : int -- number of time frames per interpolation chunk
    progress_callback : callable(int 0..100) or None

    Raises
    ------
    ValueError if abs(angle_deg) > 10.
    """
    if abs(angle_deg) > 10.0:
        raise ValueError(
            f"Rotation angle {angle_deg:.2f}° exceeds the ±10° safety limit.")

    if abs(angle_deg) < 1e-6:
        return   # nothing to do

    x    = dataset["x"].astype(np.float64)   # [ny, nx]
    y    = dataset["y"].astype(np.float64)
    U    = dataset["U"]   # [ny, nx, Nt] float32
    V    = dataset["V"]
    W    = dataset["W"]   # may be None

    ny, nx = x.shape
    Nt     = U.shape[2]

    theta   = np.deg2rad(angle_deg)
    cos_t   = np.cos(theta)
    sin_t   = np.sin(theta)

    # Centroid of the original grid
    cx = float(np.nanmean(x))
    cy = float(np.nanmean(y))

    # Rotate every grid point about the centroid
    xc = x - cx
    yc = y - cy
    x_rot = cx + xc * cos_t + yc * sin_t
    y_rot = cy - xc * sin_t + yc * cos_t

    # Build new regular grid with same dx/dy, same axis-aligned extent
    # (Option A: clip to original extent -- corners that fall outside the
    #  rotated source data become NaN)
    dx = float(abs(x[0, 1] - x[0, 0]))
    dy = float(abs(y[1, 0] - y[0, 0]))

    # Use ORIGINAL axis-aligned extent for the target grid
    x_min_orig = float(np.nanmin(x))
    x_max_orig = float(np.nanmax(x))
    y_min_orig = float(np.nanmin(y))
    y_max_orig = float(np.nanmax(y))

    x_new_1d = np.arange(x_min_orig, x_max_orig + dx * 0.5, dx)
    y_new_1d = np.arange(y_min_orig, y_max_orig + dy * 0.5, dy)

    # Clamp to same ny, nx to keep array shape identical
    x_new_1d = x_new_1d[:nx]
    y_new_1d = y_new_1d[:ny]

    x_new, y_new = np.meshgrid(x_new_1d, y_new_1d)   # [ny, nx]

    # The SOURCE grid is the rotated one (x_rot, y_rot).
    # The TARGET is the new regular grid (x_new, y_new).
    # For each target point, find where it came from in the original source grid.
    # This requires the INVERSE rotation: rotate target points by -theta to get
    # the corresponding source coordinates.
    # Inverse of R(theta) is R(-theta):
    #   x_src = cx + xc_new * cos(theta) + yc_new * sin(theta)
    #   y_src = cy - xc_new * sin(theta) + yc_new * cos(theta)
    xc_new = x_new - cx
    yc_new = y_new - cy
    x_src  = cx + xc_new * cos_t + yc_new * sin_t   # inverse rotation
    y_src  = cy - xc_new * sin_t + yc_new * cos_t

    # The source grid is the original REGULAR grid (x, y).
    # Use 1D axes for RegularGridInterpolator (requires sorted, uniform).
    x_src_1d = x[0, :]   # shape [nx], the original x axis
    y_src_1d = y[:, 0]   # shape [ny], the original y axis

    # Query points for interpolator: (y, x) ordering
    pts = np.stack([y_src.ravel(), x_src.ravel()], axis=1)   # [ny*nx, 2]

    # Validity mask: query points outside the original grid become NaN
    x_in  = (x_src >= x_src_1d[0]) & (x_src <= x_src_1d[-1])
    y_in  = (y_src >= y_src_1d[0]) & (y_src <= y_src_1d[-1])
    valid = x_in & y_in    # [ny, nx] bool

    # Clip query points to valid range for interpolator (NaN handled after)
    pts_clipped = pts.copy()
    pts_clipped[:, 0] = np.clip(pts_clipped[:, 0],
                                 y_src_1d[0], y_src_1d[-1])
    pts_clipped[:, 1] = np.clip(pts_clipped[:, 1],
                                 x_src_1d[0], x_src_1d[-1])

    valid_flat = valid.ravel()   # [ny*nx]

    # Rotate x, y in-place
    dataset["x"][:] = x_new.astype(dataset["x"].dtype)
    dataset["y"][:] = y_new.astype(dataset["y"].dtype)

    # Velocity rotation: rotate velocity by the same angle theta as the grid.
    # Standard 2D rotation:
    #   U' = U*cos(theta) - V*sin(theta)
    #   V' = U*sin(theta) + V*cos(theta)
    cos_f = np.float32(cos_t)
    sin_f = np.float32(sin_t)

    def _interpolate_field(arr_2d):
        """Interpolate a [ny, nx] snapshot onto the new grid."""
        interp = RegularGridInterpolator(
            (y_src_1d, x_src_1d), arr_2d,
            method=method, bounds_error=False, fill_value=np.nan)
        result = interp(pts_clipped).reshape(ny, nx).astype(arr_2d.dtype)
        result[~valid] = np.nan
        return result

    # Process velocity in time chunks
    total_chunks = (Nt + chunk_size - 1) // chunk_size
    for chunk_idx, t_start in enumerate(range(0, Nt, chunk_size)):
        t_end = min(t_start + chunk_size, Nt)

        # Interpolate U
        for t in range(t_start, t_end):
            u_snap = U[:, :, t].astype(np.float64)
            v_snap = V[:, :, t].astype(np.float64)

            u_interp = _interpolate_field(u_snap)
            v_interp = _interpolate_field(v_snap)

            # Rotate velocity components by same angle as the grid
            # U' = U*cos(theta) - V*sin(theta)
            # V' = U*sin(theta) + V*cos(theta)
            U[:, :, t] = (u_interp * cos_f - v_interp * sin_f)
            V[:, :, t] = (u_interp * sin_f + v_interp * cos_f)

            if W is not None:
                W[:, :, t] = _interpolate_field(W[:, :, t].astype(np.float64))

        if progress_callback is not None:
            pct = int((chunk_idx + 1) / total_chunks * 100)
            progress_callback(pct)

    # Update transform log
    dataset.setdefault("transform_log", [])
    dataset["transform_log"].append({
        "type"          : "rotation",
        "angle_deg"     : float(angle_deg),
        "interpolation" : method,
        "centroid_x"    : cx,
        "centroid_y"    : cy,
    })


# ---------------------------------------------------------------------------
# Shift
# ---------------------------------------------------------------------------

def apply_shift(dataset, dx_mm, dy_mm):
    """
    Shift the coordinate origin in-place.

    Parameters
    ----------
    dataset  : dict -- modified in-place
    dx_mm    : float -- amount to subtract from all x values
    dy_mm    : float -- amount to subtract from all y values

    No velocity change. Zero extra memory.
    """
    if abs(dx_mm) < 1e-9 and abs(dy_mm) < 1e-9:
        return

    dataset["x"] -= np.float32(dx_mm)
    dataset["y"] -= np.float32(dy_mm)

    dataset.setdefault("transform_log", [])
    dataset["transform_log"].append({
        "type" : "shift",
        "dx"   : float(dx_mm),
        "dy"   : float(dy_mm),
    })


# ---------------------------------------------------------------------------
# Status string from log
# ---------------------------------------------------------------------------

def transform_status_string(dataset):
    """
    Return a short human-readable summary of all applied transforms,
    suitable for display in the main window status strip.
    """
    log = dataset.get("transform_log", [])
    if not log:
        return ""

    parts = []
    for entry in log:
        if entry["type"] == "rotation":
            parts.append(f"Rot {entry['angle_deg']:+.2f}\u00b0 ({entry['interpolation']})")
        elif entry["type"] == "shift":
            parts.append(f"Shift ({entry['dx']:+.2f}, {entry['dy']:+.2f}) mm")

    return "  |  ".join(parts)
