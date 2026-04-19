"""
core/two_point_corr.py
----------------------
Two-point correlation analysis for PIV velocity fields.

Modes
-----
Point  : single grid point, with optional 3x3 kernel average around it
ROI    : full spatial average over a user-drawn rectangle (as in Jose et al. 2026)

Spatial correlation
  - Point mode : R(x_ref, x) returned as 2D map + 1D slices through ref point
  - ROI mode   : R(Delta_x) and R(Delta_y) as 1D curves (spatially averaged
                 over the ROI, first along rows then along cols), following
                 Eq. (2) of Jose et al. 2026

Temporal autocorrelation : R(tau) at point/ROI, FFT-based biased estimator

Integral scales
  - Length scale L  : integral of R to first zero crossing  (mm)
  - Time scale   T  : integral of R(tau) to first zero crossing (ms)
  - Taylor microscale lambda_t from curvature of R(tau) near tau=0 (ms)
"""

import numpy as np

# NumPy version-safe trapezoid
_trapz = getattr(np, "trapz", None) or getattr(np, "trapezoid", None)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def nearest_grid_point(x, y, x_click, y_click):
    dist = (x - x_click) ** 2 + (y - y_click) ** 2
    return np.unravel_index(np.argmin(dist), dist.shape)


def get_fluctuations(U, V, W):
    Up = U - np.nanmean(U, axis=2, keepdims=True)
    Vp = V - np.nanmean(V, axis=2, keepdims=True)
    Wp = (W - np.nanmean(W, axis=2, keepdims=True)) if W is not None else None
    return Up, Vp, Wp


def select_component(Up, Vp, Wp, component):
    if component == 'uu':
        return Up
    elif component == 'vv':
        return Vp
    elif component == 'ww':
        if Wp is None:
            raise ValueError("W component not available for 2D data.")
        return Wp
    raise ValueError(f"Unknown component: {component}.")


def _kernel_signal(field, row, col, use_kernel):
    """
    Return time signal at (row, col).
    If use_kernel=True, average over 3x3 patch around point.
    """
    if not use_kernel:
        return field[row, col, :]
    ny, nx, _ = field.shape
    r0 = max(0, row - 1);  r1 = min(ny, row + 2)
    c0 = max(0, col - 1);  c1 = min(nx, col + 2)
    patch = field[r0:r1, c0:c1, :]
    return np.nanmean(patch.reshape(-1, patch.shape[2]), axis=0)


def _roi_bounds_from_coords(x, y, x0, x1, y0, y1):
    """
    Return row/col index bounds for the ROI defined by coordinate extents.
    Ensures at least 1 row and 1 col.
    """
    ny, nx = x.shape
    xs = np.sort([x0, x1]);  ys = np.sort([y0, y1])
    cols = np.where((x[0, :] >= xs[0]) & (x[0, :] <= xs[1]))[0]
    rows = np.where((y[:, 0] >= ys[0]) & (y[:, 0] <= ys[1]))[0]
    if len(cols) == 0:
        cols = [np.argmin(np.abs(x[0, :] - 0.5*(xs[0]+xs[1])))]
    if len(rows) == 0:
        rows = [np.argmin(np.abs(y[:, 0] - 0.5*(ys[0]+ys[1])))]
    return int(rows[0]), int(rows[-1]+1), int(cols[0]), int(cols[-1]+1)


# ---------------------------------------------------------------------------
# Integral scale helper
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Integral scale computation -- four methods
# ---------------------------------------------------------------------------

_SUSTAINED_N   = 5      # consecutive negative points to confirm zero crossing
_FIT_THRESHOLD = 0.05   # minimum R value used in exponential fit
_POST_CROSS_WINDOW = 10  # points examined after a candidate crossing
_POST_CROSS_THRESH = 0.05  # if post-crossing mean > this, treat as noise dip

def _first_sustained_zero(R_1d):
    """
    Return index of first confirmed zero crossing.

    A crossing is confirmed only when:
    1. R stays <= 0 for at least _SUSTAINED_N consecutive points, AND
    2. The mean of the next _POST_CROSS_WINDOW points after the candidate
       start is <= _POST_CROSS_THRESH (i.e. R does not recover to positive).

    This prevents noise dips from triggering a false zero crossing.
    Returns None if no confirmed crossing is found.
    """
    n = len(R_1d)
    count = 0
    i = 0
    while i < n:
        if R_1d[i] <= 0:
            count += 1
            if count >= _SUSTAINED_N:
                i_cross = i - _SUSTAINED_N + 1   # start of the negative run
                # Confirmation: check the mean of the next window of points
                win_end  = min(n, i_cross + _POST_CROSS_WINDOW)
                post_mean = float(np.mean(R_1d[i_cross:win_end]))
                if post_mean <= _POST_CROSS_THRESH:
                    return i_cross      # confirmed crossing
                # Noise dip — skip past this run and keep scanning
                i = i + 1
                count = 0
                continue
        else:
            count = 0
        i += 1
    return None


def compute_length_scale(R_1d, spacing, method="zero_crossing"):
    """
    Compute integral length/time scale from a 1D correlation curve.

    Parameters
    ----------
    R_1d    : 1D array, normalized correlation (R[0] should be ~1)
    spacing : grid spacing in mm (spatial) or dt in s (temporal)
    method  : one of:
        "zero_crossing"  -- integrate to first sustained zero crossing
        "exp_fit"        -- fit R = exp(-r/L), integrate analytically -> L
        "one_over_e"     -- find r where R first drops to 1/e
        "domain"         -- integrate over full domain (lower bound)

    Returns
    -------
    L       : float, scale in same units as spacing (mm or s)
              NaN if method cannot be applied
    extras  : dict with diagnostic info:
        "cutoff_idx"    : index used as integration limit (zero_crossing/domain)
        "fit_L"         : fitted L from exp (exp_fit only)
        "fit_r"         : r array used in fit (exp_fit only)
        "fit_R"         : fitted R array (exp_fit only)
        "cumulative"    : cumulative integral array (all methods)
        "r_axis"        : r values for cumulative plot
        "no_crossing"   : bool, True if no zero crossing found
    """
    import numpy as np

    n   = len(R_1d)
    r   = np.arange(n) * spacing
    L   = np.nan
    extras = {
        "cutoff_idx" : None,
        "fit_L"      : None,
        "fit_r"      : None,
        "fit_R"      : None,
        "cumulative" : None,
        "r_axis"     : r,
        "no_crossing": False,
        "method_used": method,
    }

    if method == "zero_crossing":
        idx = _first_sustained_zero(R_1d)
        if idx is None:
            extras["no_crossing"] = True
            L = np.nan
        elif idx < 2:
            L = np.nan
        else:
            L = float(_trapz(R_1d[:idx], dx=spacing))
            extras["cutoff_idx"]   = idx
            extras["crossing_lag"] = idx * spacing   # Δr where R crosses zero

    elif method == "exp_fit":
        # Fit only up to the first 1/e crossing; fall back to R > threshold
        target = 1.0 / np.e
        one_e_idx = next((i for i in range(1, n) if R_1d[i] <= target), None)
        if one_e_idx is not None:
            mask = np.zeros(n, dtype=bool)
            mask[:one_e_idx + 1] = True
            mask &= np.isfinite(R_1d)
        else:
            mask = (R_1d > _FIT_THRESHOLD) & np.isfinite(R_1d)
        if np.sum(mask) < 4:
            L = np.nan
        else:
            r_fit  = r[mask]
            lnR    = np.log(np.clip(R_1d[mask], 1e-10, None))
            # Linear fit: ln(R) = -r/L  =>  slope = -1/L
            coeffs = np.polyfit(r_fit, lnR, 1)
            slope  = coeffs[0]
            if slope >= 0:
                L = np.nan
            else:
                L = float(-1.0 / slope)
                # Build fitted curve over full r range for plotting
                r_plot = np.linspace(0, r[-1], 200)
                R_plot = np.exp(-r_plot / L)
                extras["fit_L"] = L
                extras["fit_r"] = r_plot
                extras["fit_R"] = R_plot

    elif method == "one_over_e":
        ONE_OVER_E = 1.0 / np.e
        idx = None
        for i in range(n):
            if R_1d[i] <= ONE_OVER_E:
                idx = i
                break
        if idx is None:
            L = np.nan
            extras["marker_x"] = np.nan
        elif idx == 0:
            L = 0.0
            extras["marker_x"] = 0.0
            extras["marker_y"] = ONE_OVER_E
        else:
            r0 = (idx - 1) * spacing
            r1 = idx * spacing
            R0 = R_1d[idx - 1]
            R1 = R_1d[idx]
            L = r0 + (ONE_OVER_E - R0) / (R1 - R0) * (r1 - r0)
            extras["marker_x"] = L
            extras["marker_y"] = ONE_OVER_E

    elif method == "domain":
        # Integrate to first zero crossing if available, else full domain
        idx = _first_sustained_zero(R_1d)
        cutoff = idx if idx is not None else n
        if cutoff < 2:
            L = np.nan
        else:
            L = float(_trapz(R_1d[:cutoff], dx=spacing))
            extras["cutoff_idx"]   = cutoff
            extras["crossing_lag"] = cutoff * spacing   # domain end or crossing lag
            if idx is None:
                extras["no_crossing"] = True

    # Always compute cumulative integral for diagnostic plot
    cumul = np.zeros(n)
    for i in range(1, n):
        cumul[i] = float(_trapz(R_1d[:i+1], dx=spacing))
    extras["cumulative"] = cumul

    return L, extras


# ---------------------------------------------------------------------------
# POINT MODE  -- spatial correlation
# ---------------------------------------------------------------------------

def compute_spatial_correlation_point(U, V, W, row, col,
                                       component='uu', use_kernel=False):
    """
    Compute normalized 2D spatial correlation map R(x_ref, x).

    R = <u'(x_ref,t) * u'(x,t)> / (sigma_ref * sigma(x))

    Parameters
    ----------
    use_kernel : bool -- if True, average reference over 3x3 patch

    Returns
    -------
    R_norm : [ny, nx]
    R_x    : [nx]  slice along row of ref point
    R_y    : [ny]  slice along col of ref point
    Lx     : float -- integral length scale in x direction (mm)
    Ly     : float -- integral length scale in y direction (mm)
    """
    Up, Vp, Wp = get_fluctuations(U, V, W)
    field  = select_component(Up, Vp, Wp, component)
    ref    = _kernel_signal(field, row, col, use_kernel)      # [Nt]

    sigma_ref   = np.nanstd(ref)
    R_raw       = np.nanmean(field * ref[np.newaxis, np.newaxis, :], axis=2)
    sigma_field = np.nanstd(field, axis=2)
    denom       = sigma_ref * sigma_field
    denom[denom < 1e-12] = np.nan
    R_norm = R_raw / denom                                    # [ny, nx]

    R_x = R_norm[row, :]
    R_y = R_norm[:, col]

    # Grid spacing in mm (x,y assumed in mm)
    dx = float(np.abs(U.shape[1] > 1 and 1.0))   # placeholder -- passed from GUI
    return R_norm, R_x, R_y


def compute_spatial_scales_point(R_norm, row, col, x, y, method="zero_crossing"):
    """
    Compute integral length scales from 1D slices of R_norm.

    Returns
    -------
    Lx, Ly  : floats in mm
    ex, ey  : extras dicts for diagnostic plotting
    """
    R_x = R_norm[row, col:]
    R_y = R_norm[row:, col]

    dx = float(np.abs(x[0, 1] - x[0, 0])) if x.shape[1] > 1 else 1.0
    dy = float(np.abs(y[1, 0] - y[0, 0])) if y.shape[0] > 1 else 1.0

    Lx, ex = compute_length_scale(R_x, dx, method)
    Ly, ey = compute_length_scale(R_y, dy, method)
    return Lx, Ly, ex, ey


# ---------------------------------------------------------------------------
# ROI MODE  -- spatial correlation  (Jose et al. 2026, Eq. 2)
# ---------------------------------------------------------------------------

def compute_spatial_correlation_roi(U, V, W, x, y,
                                     x0, x1, y0, y1,
                                     component='uu'):
    """
    Compute spatially averaged 1D correlation functions R(Delta_x) and
    R(Delta_y) following Eq. (2) of Jose et al. 2026.

    For R(Delta_x):
      - For each row in the ROI, compute the 1D autocorrelation along that row
      - Average over all rows in the ROI

    For R(Delta_y):
      - For each col in the ROI, compute the 1D autocorrelation along that col
      - Average over all cols in the ROI

    Returns
    -------
    delta_x : [nx]  separation in mm (x-direction)
    R_x     : [nx]  spatially averaged correlation in x
    delta_y : [ny]  separation in mm (y-direction)
    R_y     : [ny]  spatially averaged correlation in y
    Lx      : float -- integral length scale x (mm)
    Ly      : float -- integral length scale y (mm)
    """
    r0, r1, c0, c1 = _roi_bounds_from_coords(x, y, x0, x1, y0, y1)

    Up, Vp, Wp = get_fluctuations(U, V, W)
    field = select_component(Up, Vp, Wp, component)   # [ny, nx, Nt]

    ny_roi = r1 - r0
    nx_roi = c1 - c0

    dx = float(np.abs(x[0, 1] - x[0, 0])) if x.shape[1] > 1 else 1.0
    dy = float(np.abs(y[1, 0] - y[0, 0])) if y.shape[0] > 1 else 1.0

    # --- R(Delta_x): average over rows of ROI ---
    Rx_rows = []
    for row in range(r0, r1):
        line = field[row, c0:c1, :]   # [nx_roi, Nt]
        R_line = _spatial_autocorr_1d(line)
        Rx_rows.append(R_line)
    R_x = np.nanmean(np.array(Rx_rows), axis=0)
    delta_x = np.arange(len(R_x)) * dx

    # --- R(Delta_y): average over cols of ROI ---
    Ry_cols = []
    for col in range(c0, c1):
        line = field[r0:r1, col, :]   # [ny_roi, Nt]
        R_line = _spatial_autocorr_1d(line)
        Ry_cols.append(R_line)
    R_y = np.nanmean(np.array(Ry_cols), axis=0)
    delta_y = np.arange(len(R_y)) * dy

    Lx, _ = compute_length_scale(R_x, dx, "zero_crossing")
    Ly, _ = compute_length_scale(R_y, dy, "zero_crossing")

    return delta_x, R_x, delta_y, R_y, Lx, Ly


def _spatial_autocorr_1d(line):
    """
    Compute normalized spatial autocorrelation along a 1D spatial line.

    line : [n_points, Nt]
    Returns R : [n_points] normalized, R(0) = 1
    """
    n, Nt = line.shape
    # Time-mean fluctuation already subtracted upstream
    # R(Delta) = < u'(z,t) * u'(z+Delta, t) >_t  averaged over z
    # Use the standard unbiased spatial lag approach:
    # for each lag Delta, average over all valid (z, z+Delta) pairs and all t
    R = np.zeros(n)
    counts = np.zeros(n)
    for lag in range(n):
        pairs_a = line[:n - lag, :]   # [n-lag, Nt]
        pairs_b = line[lag:, :]       # [n-lag, Nt]
        prod = np.nanmean(pairs_a * pairs_b, axis=1)   # average over time [n-lag]
        R[lag] = np.nanmean(prod)                      # average over space
        counts[lag] = n - lag

    if R[0] > 1e-12:
        R = R / R[0]
    return R


# ---------------------------------------------------------------------------
# TEMPORAL autocorrelation  (point or ROI)
# ---------------------------------------------------------------------------

def compute_temporal_correlation(U, V, W, row, col,
                                  component='uu',
                                  use_kernel=False,
                                  roi_coords=None,
                                  x=None, y=None,
                                  max_lag_fraction=0.5):
    """
    Compute normalized temporal autocorrelation R(tau).

    If roi_coords=(x0,x1,y0,y1) is provided, the reference signal is the
    spatial average over the ROI (consistent with ROI spatial mode).
    Otherwise uses point + optional 3x3 kernel.

    Returns
    -------
    R_tau : [n_lags]
    lags  : [n_lags]  integer lag indices
    """
    Up, Vp, Wp = get_fluctuations(U, V, W)
    field = select_component(Up, Vp, Wp, component)

    if roi_coords is not None and x is not None:
        x0, x1, y0, y1 = roi_coords
        r0, r1, c0, c1 = _roi_bounds_from_coords(x, y, x0, x1, y0, y1)
        patch = field[r0:r1, c0:c1, :]
        sig = np.nanmean(patch.reshape(-1, patch.shape[2]), axis=0)
    else:
        sig = _kernel_signal(field, row, col, use_kernel)

    Nt     = len(sig)
    sig    = np.where(np.isnan(sig), 0.0, sig)
    n_lags = int(max_lag_fraction * Nt)

    nfft    = 2 * Nt
    F       = np.fft.rfft(sig, n=nfft)
    acf     = np.fft.irfft(F * np.conj(F))[:Nt] / Nt

    if acf[0] > 1e-12:
        acf = acf / acf[0]

    return acf[:n_lags], np.arange(n_lags)


# ---------------------------------------------------------------------------
# Scales
# ---------------------------------------------------------------------------

def compute_time_scales(R_tau, lags, dt, method="zero_crossing"):
    """
    Integral time scale T and Taylor microscale lambda_t.

    Returns
    -------
    T        : float, integral time scale [s]
    lambda_t : float, Taylor microscale [s]
    extras   : dict with diagnostic info for plotting
    """
    T, extras = compute_length_scale(R_tau, dt, method)

    lambda_t = np.nan
    if len(R_tau) >= 5:
        tau_fit = lags[:5] * dt
        try:
            coeffs = np.polyfit(tau_fit, R_tau[:5], 2)
            a = coeffs[0]
            if a < 0:
                lambda_t = float(np.sqrt(-1.0 / (2.0 * a)))
        except Exception:
            pass

    return T, lambda_t, extras
