# core/mean_convergence_core.py
# ------------------------------
# Mean field computation and convergence analysis.
#
# Copyright (C) 2024  Jibu Tom Jose
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details. <https://www.gnu.org/licenses/>.

import numpy as np
from core.dataset_utils import get_masked


def compute_mean_fields(dataset):
    """
    Return time-averaged velocity fields.  Result is a new dict; the
    original dataset arrays are never modified.

    Keys in the returned dict
    -------------------------
    U_mean, V_mean : [ny, nx]  time-averaged velocity components
    W_mean         : [ny, nx]  or None if not stereo
    speed          : [ny, nx]  |<U>| = sqrt(<U>^2 + <V>^2 [+ <W>^2])
    """
    U = get_masked(dataset, "U")
    V = get_masked(dataset, "V")
    W = get_masked(dataset, "W")

    U_mean = np.nanmean(U, axis=2)
    V_mean = np.nanmean(V, axis=2)
    W_mean = np.nanmean(W, axis=2) if W is not None else None

    speed = np.sqrt(U_mean ** 2 + V_mean ** 2)
    if W_mean is not None:
        speed = np.sqrt(speed ** 2 + W_mean ** 2)

    mask = dataset["MASK"]
    U_mean[~mask] = np.nan
    V_mean[~mask] = np.nan
    speed[~mask]  = np.nan
    if W_mean is not None:
        W_mean[~mask] = np.nan

    return {"U_mean": U_mean, "V_mean": V_mean, "W_mean": W_mean, "speed": speed}


# ---------------------------------------------------------------------------
# Welford online algorithm  (single pass, O(N), no duplicate storage)
# ---------------------------------------------------------------------------

def _welford_cumulative(series):
    """
    Compute cumulative running mean, variance, and 3rd central moment using
    Welford's algorithm extended to the third moment (Knuth / Pébay 2008).

    Parameters
    ----------
    series : 1-D array, length N_total
        Per-snapshot spatially-averaged fluctuation values (already relative
        to the fixed total mean, so the final mean should be ~0).

    Returns
    -------
    mean1 : [N_total]  running <x>_N
    mean2 : [N_total]  running <(x-<x>)^2>_N  (variance)
    mean3 : [N_total]  running <(x-<x>)^3>_N  (3rd central moment)
    """
    N = len(series)
    mean1 = np.empty(N)
    mean2 = np.empty(N)
    mean3 = np.empty(N)

    n = 0
    mu = 0.0
    M2 = 0.0   # sum of squared deviations
    M3 = 0.0   # sum of cubed deviations  (Knuth notation)

    for idx in range(N):
        x     = float(series[idx])
        n_old = n
        n    += 1
        delta   = x - mu
        delta_n = delta / n
        term1   = delta * delta_n * n_old   # = delta^2 * (n-1)/n

        mu += delta_n
        # M3 must be updated *before* M2 so it uses the old M2 value
        M3 += term1 * delta_n * (n - 2) - 3.0 * delta_n * M2
        M2 += term1

        mean1[idx] = mu
        mean2[idx] = M2 / n if n > 1 else 0.0
        mean3[idx] = M3 / n if n > 1 else 0.0

    return mean1, mean2, mean3


def compute_convergence(dataset, i_pt, j_pt, kernel):
    """
    Compute cumulative running statistics for each velocity component using a
    kernel×kernel spatial average centred on grid point (i_pt, j_pt).

    Parameters
    ----------
    dataset         : dataset dict
    i_pt, j_pt     : grid indices of the selected point (row, col)
    kernel          : odd integer kernel size

    Returns
    -------
    dict with keys
      N_total    : int — total number of snapshots
      components : list of component names  ('u', 'v', [, 'w'])
      stats      : dict  component -> {
          'mean1'     : [N_total]  cumulative <u'>_N
          'mean2'     : [N_total]  cumulative <u'u'>_N
          'mean3'     : [N_total]  cumulative <u'u'u'>_N
          'final_var' : scalar    <u'u'>_final
        }
    """
    Nt        = dataset["Nt"]
    ny        = dataset["ny"]
    nx        = dataset["nx"]
    is_stereo = dataset.get("is_stereo", False)

    half = kernel // 2
    i0 = max(0, i_pt - half);  i1 = min(ny, i_pt + half + 1)
    j0 = max(0, j_pt - half);  j1 = min(nx, j_pt + half + 1)

    # Extract kernel subwindow for all snapshots — small spatial slice
    U_win = get_masked(dataset, "U")[i0:i1, j0:j1, :]   # [ky, kx, Nt]
    V_win = get_masked(dataset, "V")[i0:i1, j0:j1, :]
    W_win = (get_masked(dataset, "W")[i0:i1, j0:j1, :]
             if is_stereo else None)

    # Spatial average within kernel  -> [Nt]
    u_series = np.nanmean(U_win.reshape(-1, Nt), axis=0)
    v_series = np.nanmean(V_win.reshape(-1, Nt), axis=0)
    w_series = (np.nanmean(W_win.reshape(-1, Nt), axis=0)
                if W_win is not None else None)

    # Fixed final mean: fluctuation = raw - total_mean
    u_fluc = u_series - float(np.nanmean(u_series))
    v_fluc = v_series - float(np.nanmean(v_series))

    comp_series = {"u": u_fluc, "v": v_fluc}
    if w_series is not None:
        comp_series["w"] = w_series - float(np.nanmean(w_series))

    stats = {}
    for name, fluc in comp_series.items():
        m1, m2, m3 = _welford_cumulative(fluc)
        stats[name] = {
            "mean1":     m1,
            "mean2":     m2,
            "mean3":     m3,
            "final_var": float(m2[-1]) if Nt > 0 else 1.0,
        }

    return {
        "N_total":    Nt,
        "components": list(comp_series.keys()),
        "stats":      stats,
    }


def find_convergence_n(q_arr, scale, threshold):
    """
    Return the smallest 1-indexed N* where the trailing-window stability
    criterion is met AND remains met continuously for the next W samples
    (sustained-crossing rule).

    Stability metric at snapshot n (0-indexed):
        W(n)        = max(50, n // 10)
        range_W(n)  = max(q_norm[max(0, n-W) : n+1]) - min(...)
        met[n]      = range_W(n) / scale < threshold

    Sustained crossing: first n where met[n : n + W_min + 1].all() is True,
    where W_min = 50 (the floor of W).  W at the candidate n is 50 for
    n < 500 and grows thereafter; using W_min keeps the sustained check O(N)
    via sliding_window_view without recomputing W per offset.

    Returns None if no sustained crossing is found.
    """
    if scale == 0.0 or not np.isfinite(scale):
        return None

    N_total = len(q_arr)
    W_min   = 50
    if N_total < W_min + 2:
        return None

    q_norm = q_arr / scale

    # Step 1 — build met[n] for all n (trailing-window range criterion)
    met = np.zeros(N_total, dtype=bool)
    for n in range(N_total):
        W     = max(W_min, n // 10)
        start = max(0, n - W)
        win   = q_norm[start : n + 1]
        if len(win) >= 2:
            met[n] = float(np.max(win) - np.min(win)) < threshold

    # Step 2 — vectorized sustained check: find first n where the next
    # W_min+1 consecutive positions (including n itself) are all True.
    windows   = np.lib.stride_tricks.sliding_window_view(met, W_min + 1)
    sustained = windows.all(axis=1)   # sustained[i] = met[i : i+W_min+1].all()

    candidates = np.nonzero(sustained)[0]
    if len(candidates) == 0:
        return None

    return int(candidates[0]) + 1   # 1-indexed N*
