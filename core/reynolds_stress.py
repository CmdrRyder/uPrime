"""
core/reynolds_stress.py
-----------------------
Computes Reynolds stress tensor components from velocity arrays.

Components:
  uu, vv, ww (normal stresses)
  uv, uw, vw (shear stresses)

For 2D data: uu, vv, uv only (ww, uw, vw are None)
"""

import numpy as np


def compute_reynolds_stresses(U, V, W):
    """
    Compute all available Reynolds stress components.

    Parameters
    ----------
    U, V, W : [ny, nx, Nt]  W may be None for 2D data

    Returns
    -------
    stresses : dict with keys 'uu','vv','ww','uv','uw','vw'
               each is [ny, nx] or None if not available
    k        : [ny, nx] TKE = 0.5*(uu+vv+ww), or 0.5*(uu+vv) for 2D
    """
    mean_u = np.nanmean(U, axis=2, keepdims=True)
    mean_v = np.nanmean(V, axis=2, keepdims=True)

    up = U - mean_u
    vp = V - mean_v

    stresses = {}
    stresses["uu"] = np.nanmean(up * up, axis=2)
    stresses["vv"] = np.nanmean(vp * vp, axis=2)
    stresses["uv"] = np.nanmean(up * vp, axis=2)

    if W is not None:
        mean_w = np.nanmean(W, axis=2, keepdims=True)
        wp = W - mean_w
        stresses["ww"] = np.nanmean(wp * wp, axis=2)
        stresses["uw"] = np.nanmean(up * wp, axis=2)
        stresses["vw"] = np.nanmean(vp * wp, axis=2)
        k = 0.5 * (stresses["uu"] + stresses["vv"] + stresses["ww"])
    else:
        stresses["ww"] = None
        stresses["uw"] = None
        stresses["vw"] = None
        k = 0.5 * (stresses["uu"] + stresses["vv"])

    return stresses, k


def extract_line_profile(field, x, y, x0, y0, x1, y1):
    """
    Extract values of a 2D field along a line from (x0,y0) to (x1,y1).
    Uses nearest grid point (no interpolation).

    Returns
    -------
    vals : 1D array of field values along line
    dist : 1D array of distances from start point [mm]
    xpts, ypts : coordinates of the selected grid points
    """
    Npts = max(x.shape[0], x.shape[1]) * 2
    t    = np.linspace(0, 1, Npts)
    xl   = x0 + t * (x1 - x0)
    yl   = y0 + t * (y1 - y0)

    rows, cols = [], []
    for xi, yi in zip(xl, yl):
        dist2 = (x - xi)**2 + (y - yi)**2
        r, c  = np.unravel_index(np.argmin(dist2), dist2.shape)
        rows.append(r)
        cols.append(c)

    # Deduplicate preserving order
    seen = set()
    ur, uc = [], []
    for r, c in zip(rows, cols):
        if (r, c) not in seen:
            seen.add((r, c))
            ur.append(r)
            uc.append(c)

    ur = np.array(ur)
    uc = np.array(uc)

    vals = field[ur, uc]
    xpts = x[ur, uc]
    ypts = y[ur, uc]
    dist = np.sqrt((xpts - x0)**2 + (ypts - y0)**2)

    return vals, dist, xpts, ypts
