"""
core/spatiotemporal_spectra.py
------------------------------
Spatiotemporal spectral analysis E(k, f) from PIV data.

Procedure:
  1. Extract u'(x, t) along a line (horizontal -> kx, vertical -> ky)
  2. Subtract temporal mean per point -> fluctuations
  3. For each snapshot, apply spatial Hann window
  4. Apply temporal Hann window across snapshots
  5. 2D FFT -> E(k, f) = |u_hat(k, f)|^2 / (dk * df)
  6. Average over all lines in spatial averaging band

Axes (standard in turbulence, e.g. del Alamo & Jimenez 2009):
  k : wavenumber [rad/m]  (x-axis, log scale)
  f : frequency  [Hz]     (y-axis, log scale)
  E : log10(PSD)          (color, linear scale)

Reference: del Alamo & Jimenez (2009), Hwang & Cossu (2010)
"""

import numpy as np


def _st_psd_1line(signal_xt, dx_m, fs):
    """
    Compute 2D spatiotemporal PSD of a space-time slice.

    Parameters
    ----------
    signal_xt : [nx, Nt] array, velocity fluctuation u'(x, t)
    dx_m      : spatial grid spacing [m]
    fs        : temporal sampling frequency [Hz]

    Returns
    -------
    k   : wavenumber array [rad/m], one-sided, length nx//2+1
    f   : frequency array [Hz], one-sided, length Nt//2+1
    E   : 2D PSD [nx//2+1, Nt//2+1] in (m/s)^2 / (rad/m * Hz)
    """
    nx, Nt = signal_xt.shape

    # Replace NaN with zero (already fluctuations)
    sig = np.where(np.isfinite(signal_xt), signal_xt, 0.0)

    # Apply 2D Hann window (separable)
    win_x = np.hanning(nx)        # [nx]
    win_t = np.hanning(Nt)        # [Nt]
    win2d = np.outer(win_x, win_t)  # [nx, Nt]
    sig   = sig * win2d

    # Window normalisation factor
    win_norm = np.mean(win_x**2) * np.mean(win_t**2)

    # 2D FFT
    fft2 = np.fft.fft2(sig)       # [nx, Nt]

    # One-sided in both dimensions
    nx_h = nx // 2 + 1
    Nt_h = Nt // 2 + 1

    # Two-sided PSD
    E2 = (np.abs(fft2) ** 2) / (nx * Nt * win_norm)

    # Fold to one-sided (double non-DC, non-Nyquist bins)
    E_ks = E2[:nx_h, :Nt_h].copy()
    # Spatial fold
    if nx % 2 == 0:
        E_ks[1:-1, :] *= 2
    else:
        E_ks[1:, :] *= 2
    # Temporal fold
    if Nt % 2 == 0:
        E_ks[:, 1:-1] *= 2
    else:
        E_ks[:, 1:] *= 2

    # Frequency axes
    dk = 2 * np.pi / (nx * dx_m)     # rad/m per bin
    df = fs / Nt                      # Hz per bin

    k = np.arange(nx_h) * dk         # rad/m
    f = np.arange(Nt_h) * df         # Hz

    # Normalise so integral over k and f gives variance
    E_ks /= (dk / (2 * np.pi)) * df  # -> (m/s)^2 / (rad/m * Hz)

    return k, f, E_ks


def compute_st_spectra(U, V, W, x, y, x0, y0, x1, y1,
                       direction, avg_band, fs):
    """
    Compute spatiotemporal PSD E(k, f) along a line.

    Parameters
    ----------
    direction : "x" (horizontal line, gives kx) or "y" (vertical, gives ky)
    avg_band  : number of grid lines to average either side
    fs        : sampling frequency [Hz]

    Returns
    -------
    k    : wavenumber [rad/m]
    f    : frequency [Hz]
    psds : dict {comp: E [nk, nf]}  averaged over lines and band
    """
    ny, nx_g = x.shape
    Nt = U.shape[2]

    arrays = {"u": U, "v": V}
    if W is not None:
        arrays["w"] = W

    # Time means
    time_means = {c: np.nanmean(arr, axis=2) for c, arr in arrays.items()}

    # Determine which lines to use
    if direction == "x":
        row0 = int(np.argmin(np.abs(y[:, 0] - y0)))
        r0   = max(0, row0 - avg_band)
        r1   = min(ny - 1, row0 + avg_band)
        x_min, x_max = min(x0, x1), max(x0, x1)
        cols = np.where((x[0, :] >= x_min) & (x[0, :] <= x_max))[0]
        if len(cols) == 0:
            cols = np.arange(nx_g)
        lines = list(range(r0, r1 + 1))
        dx_m  = abs(x[0, 1] - x[0, 0]) / 1000.0

        def get_slice(arr, mean_arr, line_idx):
            # [nx_cols, Nt]
            inst = arr[line_idx, cols, :]        # [ncols, Nt]
            mean = mean_arr[line_idx, cols][:, np.newaxis]
            return inst - mean

    else:  # "y"
        col0 = int(np.argmin(np.abs(x[0, :] - x0)))
        c0   = max(0, col0 - avg_band)
        c1   = min(nx_g - 1, col0 + avg_band)
        y_min, y_max = min(y0, y1), max(y0, y1)
        rows = np.where((y[:, 0] >= y_min) & (y[:, 0] <= y_max))[0]
        if len(rows) == 0:
            rows = np.arange(ny)
        lines = list(range(c0, c1 + 1))
        dx_m  = abs(y[1, 0] - y[0, 0]) / 1000.0

        def get_slice(arr, mean_arr, line_idx):
            inst = arr[rows, line_idx, :]        # [nrows, Nt]
            mean = mean_arr[rows, line_idx][:, np.newaxis]
            return inst - mean

    k_ref = None
    f_ref = None
    accum  = {c: None for c in arrays}
    counts = {c: 0    for c in arrays}

    for comp, arr in arrays.items():
        mean_arr = time_means[comp]
        for line_idx in lines:
            fluct = get_slice(arr, mean_arr, line_idx)  # [npts, Nt]
            # Need at least 4 points in each dimension
            if fluct.shape[0] < 4 or fluct.shape[1] < 4:
                continue
            k, f, E = _st_psd_1line(fluct, dx_m, fs)
            if k_ref is None:
                k_ref = k
                f_ref = f
            if accum[comp] is None:
                accum[comp] = np.zeros_like(E)
            accum[comp] += E
            counts[comp] += 1

    psds = {}
    for comp in arrays:
        if accum[comp] is not None and counts[comp] > 0:
            psds[comp] = accum[comp] / counts[comp]
        else:
            psds[comp] = None

    return k_ref, f_ref, psds
