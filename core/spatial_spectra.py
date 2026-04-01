"""
core/spatial_spectra.py
-----------------------
Spatial spectral analysis following Pope (2000) "Turbulent Flows".

Procedure:
  1. Subtract temporal mean <u>(x) from each snapshot -> u'(x,t)
  2. Optionally subtract spatial mean of u'(x,t) along the line (removes DC)
  3. Apply Hann window + Welch averaging over snapshots
  4. Average PSD over all lines in the spatial averaging band
  5. Return wavenumber k in rad/m, PSD in (m/s)^2 / (rad/m)

Units:
  - dx input is in mm (from DaVis), converted to m internally
  - k output is in rad/m
  - PSD output is in (m/s)^2 / (rad/m)  [matches Pope's convention]

Reference: Pope (2000), Chapter 6
"""

import numpy as np
from scipy.signal import welch


def _psd_1d(signal, dx_m, nperseg, noverlap, subtract_spatial_mean):
    """
    Compute Welch PSD of a 1D spatial fluctuation signal.

    Parameters
    ----------
    signal               : 1D array, velocity fluctuation u'(x) [m/s]
    dx_m                 : grid spacing in METRES
    nperseg              : Welch segment length [grid points]
    noverlap             : overlap [grid points]
    subtract_spatial_mean: if True, subtract mean along line before FFT
                           (removes DC, optional -- time mean already removed)

    Returns
    -------
    k   : wavenumber [rad/m]
    psd : one-sided PSD [(m/s)^2 / (rad/m)]
    """
    n = len(signal)
    finite = np.isfinite(signal)
    if finite.sum() < max(nperseg, 4):
        return None, None

    # Fill NaN with zero (already fluctuations, so zero is appropriate)
    sig = np.where(finite, signal, 0.0)

    if subtract_spatial_mean:
        sig = sig - np.mean(sig)

    # Spatial sampling frequency in cycles/m
    fs_spatial = 1.0 / dx_m    # [cycles/m]

    nperseg  = min(nperseg, n)
    noverlap = min(noverlap, nperseg - 1)

    # scipy welch returns:
    #   f  in cycles/m (since fs is in cycles/m)
    #   Pxx in (m/s)^2 / (cycles/m)
    f, pxx = welch(sig, fs=fs_spatial, nperseg=nperseg,
                   noverlap=noverlap, window="hann", scaling="density")

    # Convert from cycles/m to rad/m:
    #   k = 2*pi*f
    #   E(k) = Pxx(f) / (2*pi)   [so that integral over k gives same variance]
    k   = 2 * np.pi * f
    psd = pxx / (2 * np.pi)

    return k, psd


def _accumulate_psds(arrays, time_means, signals_fn, dx_m,
                     nperseg, noverlap, subtract_spatial_mean):
    """
    Core accumulator: subtract time mean, then compute and average PSDs.

    Parameters
    ----------
    arrays       : dict {comp: [ny, nx, Nt]}
    time_means   : dict {comp: [ny, nx]}  pre-computed temporal means
    signals_fn   : callable(arr, mean_arr, t) -> list of (signal, mean_signal) pairs
    dx_m         : grid spacing in metres
    """
    k_ref  = None
    accum  = {c: None for c in arrays}
    counts = {c: 0    for c in arrays}

    for comp, arr in arrays.items():
        Nt       = arr.shape[2]
        mean_arr = time_means[comp]

        for t in range(Nt):
            pairs = signals_fn(arr, mean_arr, t)
            for (sig_inst, sig_mean) in pairs:
                # Step 1: subtract temporal mean -> fluctuation u'(x,t)
                fluct = sig_inst.astype(np.float64) - sig_mean.astype(np.float64)

                k, p = _psd_1d(fluct, dx_m, nperseg, noverlap, subtract_spatial_mean)
                if p is None:
                    continue
                if k_ref is None:
                    k_ref = k
                if accum[comp] is None:
                    accum[comp] = np.zeros(len(k_ref))
                accum[comp]  += p
                counts[comp] += 1

    psds = {}
    for comp in arrays:
        if accum[comp] is not None and counts[comp] > 0:
            psds[comp] = accum[comp] / counts[comp]
        else:
            psds[comp] = None

    return k_ref, psds


def _build_arrays_and_means(U, V, W):
    """Build component arrays and their temporal means."""
    arrays = {"u": U, "v": V}
    if W is not None:
        arrays["w"] = W
    time_means = {c: np.nanmean(arr, axis=2) for c, arr in arrays.items()}
    return arrays, time_means


def spatial_psd_line(U, V, W, x, y, x0, y0, x1, y1,
                     direction, avg_band,
                     nperseg, noverlap, subtract_spatial_mean):
    """
    Compute spatial PSD along a horizontal or vertical line.

    Parameters
    ----------
    direction : "x" (horizontal) or "y" (vertical)
    avg_band  : spatial averaging band in grid points either side

    Returns
    -------
    k    : wavenumber [rad/m]
    psds : dict {comp: PSD [(m/s)^2/(rad/m)]}
    """
    ny, nx = x.shape
    arrays, time_means = _build_arrays_and_means(U, V, W)

    # dx, dy in mm -> convert to metres
    dx_mm = abs(x[0, 1] - x[0, 0])
    dy_mm = abs(y[1, 0] - y[0, 0])

    if direction == "x":
        row0 = int(np.argmin(np.abs(y[:, 0] - y0)))
        r0   = max(0, row0 - avg_band)
        r1   = min(ny - 1, row0 + avg_band)
        x_min, x_max = min(x0, x1), max(x0, x1)
        cols = np.where((x[0, :] >= x_min) & (x[0, :] <= x_max))[0]
        if len(cols) == 0:
            cols = np.arange(nx)
        rows_to_avg = np.arange(r0, r1 + 1)
        dx_m = dx_mm / 1000.0

        def signals_fn(arr, mean_arr, t):
            return [(arr[r, cols, t], mean_arr[r, cols]) for r in rows_to_avg]

        spacing_m = dx_m

    else:  # "y"
        col0 = int(np.argmin(np.abs(x[0, :] - x0)))
        c0   = max(0, col0 - avg_band)
        c1   = min(nx - 1, col0 + avg_band)
        y_min, y_max = min(y0, y1), max(y0, y1)
        rows = np.where((y[:, 0] >= y_min) & (y[:, 0] <= y_max))[0]
        if len(rows) == 0:
            rows = np.arange(ny)
        cols_to_avg = np.arange(c0, c1 + 1)
        dy_m = dy_mm / 1000.0

        def signals_fn(arr, mean_arr, t):
            return [(arr[rows, c, t], mean_arr[rows, c]) for c in cols_to_avg]

        spacing_m = dy_m

    return _accumulate_psds(arrays, time_means, signals_fn,
                            spacing_m, nperseg, noverlap, subtract_spatial_mean)


def spatial_psd_roi(U, V, W, x, y, x0, x1, y0, y1,
                    nperseg, noverlap, subtract_spatial_mean):
    """
    Compute spatial PSD inside a rectangle for both x and y directions.

    Returns
    -------
    results : dict {"x": {"k": ..., "psds": ...}, "y": {"k": ..., "psds": ...}}
    n_lines : {"x": number of rows averaged, "y": number of cols averaged}
    """
    ny, nx_g = x.shape
    arrays, time_means = _build_arrays_and_means(U, V, W)

    dx_m = abs(x[0, 1] - x[0, 0]) / 1000.0
    dy_m = abs(y[1, 0] - y[0, 0]) / 1000.0

    x_min, x_max = min(x0, x1), max(x0, x1)
    y_min, y_max = min(y0, y1), max(y0, y1)

    cols = np.where((x[0, :] >= x_min) & (x[0, :] <= x_max))[0]
    rows = np.where((y[:, 0] >= y_min) & (y[:, 0] <= y_max))[0]

    results = {}
    n_lines = {"x": len(rows), "y": len(cols)}

    # x-direction spectra: one spectrum per row
    def sigs_x(arr, mean_arr, t):
        return [(arr[r, cols, t], mean_arr[r, cols]) for r in rows]

    k_x, psds_x = _accumulate_psds(arrays, time_means, sigs_x,
                                    dx_m, nperseg, noverlap, subtract_spatial_mean)
    results["x"] = {"k": k_x, "psds": psds_x}

    # y-direction spectra: one spectrum per column
    def sigs_y(arr, mean_arr, t):
        return [(arr[rows, c, t], mean_arr[rows, c]) for c in cols]

    k_y, psds_y = _accumulate_psds(arrays, time_means, sigs_y,
                                    dy_m, nperseg, noverlap, subtract_spatial_mean)
    results["y"] = {"k": k_y, "psds": psds_y}

    return results, n_lines