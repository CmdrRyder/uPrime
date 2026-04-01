"""
core/tke_budget.py
------------------
TKE budget computation from stereo PIV data.

2D plane assumption: all d/dz terms are neglected.

Terms computed:
  k   : TKE = 0.5*(uu + vv + ww) or 0.5*(uu+vv) for 2D
  P   : Production  = -<u'i u'j> * dUi/dxj  (in-plane only)
  C   : Convection  = U*dk/dx + V*dk/dy
  D   : Turbulent diffusion = -d/dx(0.5*<u'i u'i u'>) - d/dy(0.5*<u'i u'i v'>)
  R   : Residual = -(C - P - D)  absorbs dissipation + missing terms

For TR data:
  dkdt: temporal derivative of k (finite diff of instantaneous k, then averaged)

All normalized by Um^3/L when requested.

Reference: Pope (2000) Ch 5; Kasagi & Matsunaga (1995)
"""

import numpy as np
from scipy.ndimage import median_filter


def _grad2(field, dx, dy):
    """Central-difference gradient of a 2D field [ny,nx].
    Returns dfdx [ny,nx], dfdy [ny,nx]."""
    dfdx = np.gradient(field, dx, axis=1)   # along x (cols)
    dfdy = np.gradient(field, dy, axis=0)   # along y (rows)
    return dfdx, dfdy


def compute_tke_budget(U, V, W, x, y,
                       smooth_kernel=3,
                       compute_dkdt=False):
    """
    Compute all TKE budget terms from PIV velocity arrays.

    Parameters
    ----------
    U, V, W  : [ny, nx, Nt]  W may be None for 2D
    x, y     : [ny, nx] grid in mm
    smooth_kernel : int, median filter size for triple correlations
    compute_dkdt  : bool, compute temporal TKE derivative (TR only)

    Returns
    -------
    results : dict with keys:
        'k'    : [ny,nx] mean TKE [m^2/s^2]
        'P'    : [ny,nx] production
        'C'    : [ny,nx] convection
        'D'    : [ny,nx] turbulent diffusion
        'R'    : [ny,nx] residual (dissipation proxy)
        'dkdt' : [ny,nx] or None
    All in m^2/s^3 (before normalization)
    """
    ny, nx, Nt = U.shape

    # Grid spacing in metres
    dx = abs(x[0, 1] - x[0, 0]) / 1000.0
    dy = abs(y[1, 0] - y[0, 0]) / 1000.0

    # --- Time means ---
    mean_U = np.nanmean(U, axis=2)
    mean_V = np.nanmean(V, axis=2)

    # --- Fluctuations ---
    up = U - mean_U[:, :, np.newaxis]
    vp = V - mean_V[:, :, np.newaxis]

    # --- Reynolds stresses ---
    uu = np.nanmean(up * up, axis=2)
    vv = np.nanmean(vp * vp, axis=2)
    uv = np.nanmean(up * vp, axis=2)

    if W is not None:
        mean_W = np.nanmean(W, axis=2)
        wp = W - mean_W[:, :, np.newaxis]
        ww = np.nanmean(wp * wp, axis=2)
        uw = np.nanmean(up * wp, axis=2)
        vw = np.nanmean(vp * wp, axis=2)
        k  = 0.5 * (uu + vv + ww)
    else:
        wp = None
        ww = np.zeros((ny, nx))
        uw = np.zeros((ny, nx))
        vw = np.zeros((ny, nx))
        k  = 0.5 * (uu + vv)

    # --- Mean velocity gradients ---
    dUdx, dUdy = _grad2(mean_U, dx, dy)
    dVdx, dVdy = _grad2(mean_V, dx, dy)

    # --- Production (2D plane, no dz terms) ---
    # P = -(uu*dU/dx + uv*dU/dy + uv*dV/dx + vv*dV/dy
    #       + uw*dW/dx + vw*dW/dy)  [uw,vw terms need dW gradients]
    P = -(uu * dUdx + uv * dUdy + uv * dVdx + vv * dVdy)
    if W is not None:
        mean_W = np.nanmean(W, axis=2)
        dWdx, dWdy = _grad2(mean_W, dx, dy)
        P -= (uw * dWdx + vw * dWdy)

    # --- TKE gradients for convection ---
    dkdx, dkdy = _grad2(k, dx, dy)

    # --- Convection ---
    C = mean_U * dkdx + mean_V * dkdy

    # --- Turbulent diffusion ---
    # D = -d/dx(0.5*<q^2 u'>) - d/dy(0.5*<q^2 v'>)
    # where q^2 = u'^2 + v'^2 + w'^2
    q2 = up**2 + vp**2
    if wp is not None:
        q2 += wp**2

    # Triple correlations
    q2u = np.nanmean(q2 * up, axis=2)  # <q^2 u'>
    q2v = np.nanmean(q2 * vp, axis=2)  # <q^2 v'>

    # Smooth triple correlations (noisy)
    if smooth_kernel > 1:
        # Fill NaN before filtering
        def _smooth(arr, k):
            tmp = arr.copy()
            nan_mask = np.isnan(tmp)
            tmp[nan_mask] = 0.0
            tmp = median_filter(tmp, size=k)
            tmp[nan_mask] = np.nan
            return tmp
        q2u = _smooth(q2u, smooth_kernel)
        q2v = _smooth(q2v, smooth_kernel)

    d_q2u_dx, _ = _grad2(q2u, dx, dy)
    _, d_q2v_dy = _grad2(q2v, dx, dy)

    D = -0.5 * (d_q2u_dx + d_q2v_dy)

    # --- Residual (absorbs dissipation + missing terms) ---
    # Budget: dk/dt + C = P - eps + D  => eps_residual = P - C - D (steady)
    R = P - C - D

    # --- Temporal derivative (TR only) ---
    dkdt = None
    if compute_dkdt and Nt > 1:
        # Instantaneous TKE per snapshot
        k_inst = 0.5 * (up**2 + vp**2)
        if wp is not None:
            k_inst += 0.5 * wp**2
        # Time derivative via finite difference, then average
        dk_dt_inst = np.diff(k_inst, axis=2)  # [ny,nx,Nt-1]
        dkdt = np.nanmean(dk_dt_inst, axis=2)  # [ny,nx]

    return {
        "k"    : k,
        "P"    : P,
        "C"    : C,
        "D"    : D,
        "R"    : R,
        "dkdt" : dkdt,
    }
