"""
core/line_sample.py
-------------------
Shared free-line sampler for 2D fields.

Used by the line-profile modules (Reynolds stresses, TKE budget, mean
velocity) for both drawn free lines and manually entered start/end
coordinates, so that both entry methods sample identically.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def sample_along_line(x, y, field, p0, p1):
    """
    Sample a 2D field along a straight line between two physical points.

    Parameters
    ----------
    x, y  : 1D physical-coordinate axes in mm (x has length Nx, y has Ny)
    field : 2D array of shape (Ny, Nx); may be masked / contain NaN
    p0    : (x0, y0) start point in mm
    p1    : (x1, y1) end point in mm

    Returns
    -------
    s      : 1D arc length from p0 [mm]
    values : 1D sampled field values (NaN over masked / out-of-bounds points)

    Notes
    -----
    N = max(2, ceil(line_length / min(dx, dy))) evenly spaced points are
    taken from p0 to p1 and interpolated linearly. Masked / NaN fields are
    fed as-is, so points over masked regions correctly come back NaN.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    field = np.asarray(field, dtype=float)

    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])

    line_length = np.hypot(x1 - x0, y1 - y0)

    dx = np.abs(np.mean(np.diff(x))) if x.size > 1 else 1.0
    dy = np.abs(np.mean(np.diff(y))) if y.size > 1 else 1.0
    step = min(dx, dy)
    if not np.isfinite(step) or step <= 0:
        step = line_length if line_length > 0 else 1.0

    N = max(2, int(np.ceil(line_length / step)))

    xs = np.linspace(x0, x1, N)
    ys = np.linspace(y0, y1, N)
    s  = np.linspace(0.0, line_length, N)

    interp = RegularGridInterpolator(
        (y, x), field, method="linear",
        bounds_error=False, fill_value=np.nan)
    values = interp(np.column_stack([ys, xs]))

    return s, values
