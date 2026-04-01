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


def extract_line_profile(field, x, y, x0, y0, x1, y1,
                         mode="free", avg_band=0):
    """
    Extract values of a 2D field along a line.

    Parameters
    ----------
    field    : [ny, nx] 2D field
    x, y     : [ny, nx] coordinate arrays
    x0,y0,x1,y1 : line endpoints
    mode     : "free"       -- nearest grid point to drawn line
               "horizontal" -- snap to nearest row at y0, limit x0..x1
               "vertical"   -- snap to nearest col at x0, limit y0..y1
    avg_band : int >= 0, number of grid points to average either side
               (only applied for horizontal/vertical modes)

    Returns
    -------
    vals : 1D averaged profile
    dist : distance from start point [mm]
    xpts, ypts : coordinates along profile
    """
    ny, nx = x.shape

    if mode == "horizontal":
        # Find nearest row index to y0
        row0 = int(np.argmin(np.abs(y[:, 0] - y0)))
        # Find col range for x0..x1
        x_min, x_max = min(x0, x1), max(x0, x1)
        cols = np.where((x[0, :] >= x_min) & (x[0, :] <= x_max))[0]
        if len(cols) == 0:
            cols = np.arange(nx)

        xpts = x[row0, cols]
        ypts = y[row0, cols]
        dist = xpts - xpts[0]

        if avg_band > 0:
            r0 = max(0, row0 - avg_band)
            r1 = min(ny - 1, row0 + avg_band)
            vals = np.nanmean(field[r0:r1+1, :][:, cols], axis=0)
        else:
            vals = field[row0, cols]

    elif mode == "vertical":
        # Find nearest col index to x0
        col0 = int(np.argmin(np.abs(x[0, :] - x0)))
        # Find row range for y0..y1
        y_min, y_max = min(y0, y1), max(y0, y1)
        rows = np.where((y[:, 0] >= y_min) & (y[:, 0] <= y_max))[0]
        if len(rows) == 0:
            rows = np.arange(ny)

        xpts = x[rows, col0]
        ypts = y[rows, col0]
        dist = ypts - ypts[0]

        if avg_band > 0:
            c0 = max(0, col0 - avg_band)
            c1 = min(nx - 1, col0 + avg_band)
            vals = np.nanmean(field[:, c0:c1+1][rows, :], axis=1)
        else:
            vals = field[rows, col0]

    else:
        # Free line -- nearest grid point
        Npts = max(ny, nx) * 2
        t    = np.linspace(0, 1, Npts)
        xl   = x0 + t * (x1 - x0)
        yl   = y0 + t * (y1 - y0)

        rows_list, cols_list = [], []
        for xi, yi in zip(xl, yl):
            d2   = (x - xi)**2 + (y - yi)**2
            r, c = np.unravel_index(np.argmin(d2), d2.shape)
            rows_list.append(r)
            cols_list.append(c)

        seen = set()
        ur, uc = [], []
        for r, c in zip(rows_list, cols_list):
            if (r, c) not in seen:
                seen.add((r, c))
                ur.append(r)
                uc.append(c)

        ur   = np.array(ur)
        uc   = np.array(uc)
        vals = field[ur, uc]
        xpts = x[ur, uc]
        ypts = y[ur, uc]
        dist = np.sqrt((xpts - x0)**2 + (ypts - y0)**2)

    return np.array(vals, dtype=float), np.array(dist, dtype=float),            np.array(xpts, dtype=float), np.array(ypts, dtype=float)


def compute_reynolds_stress_std(U, V, W):
    """
    Compute temporal standard deviation of each Reynolds stress component.
    This measures how much the instantaneous Rij varies around its mean
    across snapshots -- useful for uncertainty bands on line plots.

    Returns
    -------
    std_stresses : dict with same keys as compute_reynolds_stresses,
                   each [ny, nx] array of std dev
    """
    mean_u = np.nanmean(U, axis=2, keepdims=True)
    mean_v = np.nanmean(V, axis=2, keepdims=True)

    up = U - mean_u
    vp = V - mean_v

    # Instantaneous Rij per snapshot, then std over time
    std = {}
    std["uu"] = np.nanstd(up * up, axis=2)
    std["vv"] = np.nanstd(vp * vp, axis=2)
    std["uv"] = np.nanstd(up * vp, axis=2)

    if W is not None:
        mean_w = np.nanmean(W, axis=2, keepdims=True)
        wp = W - mean_w
        std["ww"] = np.nanstd(wp * wp, axis=2)
        std["uw"] = np.nanstd(up * wp, axis=2)
        std["vw"] = np.nanstd(vp * wp, axis=2)
    else:
        std["ww"] = None
        std["uw"] = None
        std["vw"] = None

    return std


def compute_tke_std(U, V, W, mode="2d"):
    """
    Compute temporal standard deviation of TKE across snapshots.

    mode : "2d" uses 0.5*(u^2+v^2)
           "3d" uses 0.5*(u^2+v^2+w^2)
    """
    mean_u = np.nanmean(U, axis=2, keepdims=True)
    mean_v = np.nanmean(V, axis=2, keepdims=True)
    up = U - mean_u
    vp = V - mean_v

    if mode == "3d" and W is not None:
        mean_w = np.nanmean(W, axis=2, keepdims=True)
        wp = W - mean_w
        k_inst = 0.5 * (up**2 + vp**2 + wp**2)
    else:
        k_inst = 0.5 * (up**2 + vp**2)

    return np.nanstd(k_inst, axis=2)
