"""
core/spectral.py
----------------
Temporal spectral analysis using Welch's method.

For a point  : compute PSD of u(t), v(t), w(t) at that grid point.
For a region : average PSD across all valid points inside the rectangle.

Uses scipy.signal.welch.
"""

import numpy as np
from scipy.signal import welch


def nearest_grid_point(x, y, x_click, y_click):
    """
    Return (row, col) of the grid point nearest to (x_click, y_click).
    x, y are 2D arrays [ny, nx].
    """
    dist = (x - x_click)**2 + (y - y_click)**2
    idx  = np.unravel_index(np.argmin(dist), dist.shape)
    return idx   # (row, col)


def psd_at_point(U, V, W, row, col, fs, nperseg=None, noverlap=None):
    """
    Compute Welch PSD at a single grid point.

    Parameters
    ----------
    U, V, W : [ny, nx, Nt]  velocity arrays (W may be None)
    row, col : grid indices
    fs       : sampling frequency in Hz
    nperseg  : Welch segment length (default: Nt//4)
    noverlap : overlap samples (default: nperseg//2)

    Returns
    -------
    freq : 1D array of frequencies [Hz]
    psd  : dict with keys 'u', 'v', 'w' (w may be None)
           each value is a 1D PSD array
    """
    Nt = U.shape[2]
    if nperseg is None:
        nperseg  = max(16, Nt // 4)
    if noverlap is None:
        noverlap = nperseg // 2

    Nt = U.shape[2]

    # Guard: nperseg cannot exceed signal length
    nperseg  = min(nperseg, Nt)
    noverlap = min(noverlap, nperseg - 1)

    freq = None
    psd  = {}

    for label, arr in [("u", U), ("v", V), ("w", W)]:
        if arr is None:
            psd[label] = None
            continue

        signal = arr[row, col, :].astype(np.float64)

        # Skip if mostly NaN
        n_valid = np.sum(np.isfinite(signal))
        if n_valid < max(nperseg, 4):
            psd[label] = None
            continue

        # Replace NaN with local mean (simple gap fill)
        mean_val = np.nanmean(signal)
        signal   = np.where(np.isfinite(signal), signal, mean_val)

        # Subtract mean (compute spectrum of fluctuations)
        signal -= np.mean(signal)

        f, p = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap,
                     window="hann", scaling="density")

        if freq is None:
            freq = f
        psd[label] = p

    if freq is None:
        raise ValueError(
            f"No valid data found at the selected point (row={row}, col={col}). "
            f"The point may be in a masked region, or Nt={Nt} is too small for "
            f"the chosen segment length ({nperseg})."
        )

    return freq, psd


def psd_in_region(U, V, W, x, y, x0, x1, y0, y1, fs, nperseg=None, noverlap=None):
    """
    Compute average Welch PSD for all valid points inside a rectangle.

    Rectangle defined by x in [x0,x1] and y in [y0,y1].

    Returns
    -------
    freq   : 1D array of frequencies [Hz]
    psd    : dict with keys 'u', 'v', 'w' -- averaged PSDs
    n_pts  : number of valid points averaged
    """
    Nt = U.shape[2]
    if nperseg is None:
        nperseg  = max(16, Nt // 4)
    if noverlap is None:
        noverlap = nperseg // 2

    # Find all grid points inside the rectangle
    mask = (x >= min(x0, x1)) & (x <= max(x0, x1)) & \
           (y >= min(y0, y1)) & (y <= max(y0, y1))

    rows, cols = np.where(mask)

    if len(rows) == 0:
        raise ValueError("No grid points found inside the selected rectangle.")

    # Compute PSD at each point, accumulate
    freq_ref = None
    accum    = {"u": None, "v": None, "w": None}
    counts   = {"u": 0,    "v": 0,    "w": 0}

    for r, c in zip(rows, cols):
        freq, psd = psd_at_point(U, V, W, r, c, fs, nperseg, noverlap)

        if freq_ref is None:
            freq_ref = freq
            for key in accum:
                if psd[key] is not None:
                    accum[key] = np.zeros_like(freq)

        for key in accum:
            if psd[key] is not None and accum[key] is not None:
                accum[key]  += psd[key]
                counts[key] += 1

    # Average
    psd_avg = {}
    for key in accum:
        if accum[key] is not None and counts[key] > 0:
            psd_avg[key] = accum[key] / counts[key]
        else:
            psd_avg[key] = None

    return freq_ref, psd_avg, len(rows)
