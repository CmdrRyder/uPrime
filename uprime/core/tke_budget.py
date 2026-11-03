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
  R   : Residual = P + D - C  absorbs dissipation + missing terms

For TR data:
  dkdt: temporal derivative of k (finite diff of instantaneous k, then averaged)

All normalized by Um^3/L when requested.

Reference: Pope (2000) Ch 5; Kasagi & Matsunaga (1995)
"""

import sys
import numpy as np
from scipy.ndimage import median_filter, uniform_filter, binary_dilation


def _inpaint_nans(field, iterations=3):
    """
    Replace NaN pixels with the mean of their valid neighbors.
    Repeat for a few iterations to fill gaps up to a few pixels wide.
    This is used only for gradient computation -- NaNs are restored after.
    """
    out = field.copy()
    for _ in range(iterations):
        nan_mask = np.isnan(out)
        if not nan_mask.any():
            break
        filled = np.where(nan_mask, 0.0, out)
        count  = np.where(nan_mask, 0.0, 1.0)
        neighbor_sum   = uniform_filter(filled, size=3) * 9
        neighbor_count = uniform_filter(count,  size=3) * 9
        neighbor_sum   -= filled
        neighbor_count -= count
        valid_fill = np.where(neighbor_count > 0,
                              neighbor_sum / np.maximum(neighbor_count, 1),
                              0.0)
        out = np.where(nan_mask, valid_fill, out)
    return out


def _grad2(field, x_mm, y_mm, nan_mask=None):
    """
    Gradient of 2D field [ny, nx] with coordinates in mm.
    Inpaints NaN regions before differentiating to avoid edge artifacts.
    Restores NaN mask afterward.
    Returns (dfdx, dfdy) in [field_units / meter].
    """
    x_1d_m = x_mm[0, :].astype(float) / 1000.0
    y_1d_m = y_mm[:, 0].astype(float) / 1000.0

    field_filled = _inpaint_nans(field, iterations=5)

    dfdx = np.gradient(field_filled, x_1d_m, axis=1)
    dfdy = np.gradient(field_filled, y_1d_m, axis=0)

    if nan_mask is None:
        nan_mask = np.isnan(field)
    dilated = binary_dilation(nan_mask, iterations=2)
    dfdx[dilated] = np.nan
    dfdy[dilated] = np.nan

    return dfdx, dfdy


def compute_tke_budget(U, V, W, x, y, mask=None,
                       smooth_kernel=3,
                       compute_dkdt=False):
    """
    Compute all TKE budget terms from PIV velocity arrays.

    Parameters
    ----------
    U, V, W  : [ny, nx, Nt]  W may be None for 2D
    x, y     : [ny, nx] grid in mm
    mask     : [ny, nx] bool array, True = valid pixels (optional)
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

    # --- Global NaN mask (union of invalid locations) ---
    nan_mask = np.isnan(mean_U) | np.isnan(mean_V)
    if mask is not None:
        nan_mask = nan_mask | (~mask)

    # --- Mean velocity gradients ---
    dUdx, dUdy = _grad2(mean_U, x, y, nan_mask)
    dVdx, dVdy = _grad2(mean_V, x, y, nan_mask)

    # --- Production (2D plane, no dz terms) ---
    # Pope (2000) Eq. 5.132: P = -<u'i u'j> dUi/dxj
    # 2D in-plane: P = -(uu*dU/dx + uv*(dU/dy + dV/dx) + vv*dV/dy)
    P = -(uu * dUdx + uv * (dUdy + dVdx) + vv * dVdy)
    if W is not None:
        dWdx, dWdy = _grad2(mean_W, x, y, nan_mask)
        # Note: dU/dz, dV/dz out-of-plane terms are unavailable and neglected
        P -= (uw * dWdx + vw * dWdy)

    # --- TKE gradients for convection ---
    dkdx, dkdy = _grad2(k, x, y, nan_mask)

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
        def _smooth(arr, k):
            tmp = arr.copy()
            nm = np.isnan(tmp)
            tmp[nm] = 0.0
            tmp = median_filter(tmp, size=k)
            tmp[nm] = np.nan
            return tmp
        q2u = _smooth(q2u, smooth_kernel)
        q2v = _smooth(q2v, smooth_kernel)

    d_q2u_dx, _ = _grad2(q2u, x, y, nan_mask)
    _, d_q2v_dy  = _grad2(q2v, x, y, nan_mask)

    D = -0.5 * (d_q2u_dx + d_q2v_dy)

    # --- Residual: steady-state budget dk/dt + C = P + D - ε => R = P + D - C ---
    R = P + D - C

    # --- Apply nan_mask to all budget outputs ---
    for arr in (k, P, C, D, R):
        arr[nan_mask] = np.nan

    print(f"[TKE] k={np.nanmax(k):.3f} m\u00b2/s\u00b2, "
          f"P={np.nanmax(np.abs(P)):.1f}, "
          f"C={np.nanmax(np.abs(C)):.1f}, "
          f"D={np.nanmax(np.abs(D)):.1f} m\u00b2/s\u00b3",
          file=sys.stderr, flush=True)

    # --- Temporal derivative (TR only) ---
    dkdt = None
    if compute_dkdt and Nt > 1:
        k_inst = 0.5 * (up**2 + vp**2)
        if wp is not None:
            k_inst += 0.5 * wp**2
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
