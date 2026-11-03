"""
core/pod.py
-----------
Proper Orthogonal Decomposition (snapshot POD) for PIV datasets.

Uses the method-of-snapshots (Sirovich 1987):
  - Temporal correlation matrix C = Q Q^T / Nt  (cheap when Nt << n_points)
  - Spatial modes recovered via projection
"""

import numpy as np


def compute_pod(U, V, W, n_modes=25):
    """Compute snapshot POD.

    Parameters
    ----------
    U, V : ndarray, shape [ny, nx, Nt]
    W    : ndarray, shape [ny, nx, Nt]  or  None  (stereo / 2D)
    n_modes : int
        Number of POD modes to retain.

    Returns
    -------
    dict with keys:
        eigenvalues   : (n_modes,)
        modes         : (n_modes, ny, nx, Nc)
        coeffs        : (Nt, n_modes)
        energy_frac   : (n_modes,)   fraction of TKE per mode
        cumul_energy  : (n_modes,)   cumulative energy fraction
        mask          : (ny, nx)  bool, True = valid point
        n_modes       : int
        Nc            : int (2 or 3)
        mean_U        : (ny, nx)
        mean_V        : (ny, nx)
        mean_W        : (ny, nx) or None
    """
    ny, nx, Nt = U.shape
    Nc = 3 if W is not None else 2

    # ------------------------------------------------------------------ #
    # 1. Velocity fluctuations                                             #
    # ------------------------------------------------------------------ #
    mean_U = np.nanmean(U, axis=2)
    mean_V = np.nanmean(V, axis=2)
    u = U - mean_U[:, :, np.newaxis]
    v = V - mean_V[:, :, np.newaxis]

    if W is not None:
        mean_W = np.nanmean(W, axis=2)
        w = W - mean_W[:, :, np.newaxis]
    else:
        mean_W = None
        w = None

    # ------------------------------------------------------------------ #
    # 2. Valid-point mask (< 10 % NaN snapshots)                          #
    # ------------------------------------------------------------------ #
    nan_frac = np.mean(np.isnan(U), axis=2)          # use U as reference
    mask = nan_frac < 0.10                            # shape [ny, nx]

    # ------------------------------------------------------------------ #
    # 3. Fill NaN with 0 on the fluctuation fields                        #
    # ------------------------------------------------------------------ #
    u = np.where(np.isnan(u), 0.0, u)
    v = np.where(np.isnan(v), 0.0, v)
    if w is not None:
        w = np.where(np.isnan(w), 0.0, w)

    # ------------------------------------------------------------------ #
    # 4. Snapshot matrix Q  [Nt, n_valid * Nc]                            #
    # ------------------------------------------------------------------ #
    n_valid = int(mask.sum())

    # Extract valid points; each component contributes n_valid columns
    def _extract(field):
        # field shape [ny, nx, Nt] -> [Nt, n_valid]
        return field[mask, :].T           # (Nt, n_valid)

    if Nc == 2:
        Q = np.concatenate([_extract(u), _extract(v)], axis=1)   # (Nt, 2*n_valid)
    else:
        Q = np.concatenate([_extract(u), _extract(v), _extract(w)], axis=1)

    # ------------------------------------------------------------------ #
    # 5. Temporal correlation matrix C = Q Q^T / Nt                       #
    # ------------------------------------------------------------------ #
    C = (Q @ Q.T) / Nt           # (Nt, Nt)

    # ------------------------------------------------------------------ #
    # 6. Eigenvalue problem                                                #
    # ------------------------------------------------------------------ #
    # eigh returns eigenvalues in ascending order; reverse to get dominant first
    lam, A = np.linalg.eigh(C)   # lam (Nt,), A (Nt, Nt)
    lam = lam[::-1].copy()       # descending
    A   = A[:, ::-1].copy()      # columns reversed accordingly

    # ------------------------------------------------------------------ #
    # 7. Clip n_modes                                                      #
    # ------------------------------------------------------------------ #
    n_modes = min(n_modes, Nt)

    lam = lam[:n_modes]
    A   = A[:, :n_modes]         # (Nt, n_modes)  temporal coefficients (raw)

    # ------------------------------------------------------------------ #
    # 8. Spatial modes  phi_n = Q^T A[:,n] / sqrt(lambda_n * Nt)         #
    # ------------------------------------------------------------------ #
    # This normalisation gives unit-L2-norm modes:
    #   ||phi_n||^2 = A[:,n]^T C A[:,n] / lambda_n = 1
    lam_safe = np.where(lam > 0, lam, np.finfo(float).tiny)
    scale = 1.0 / np.sqrt(lam_safe * Nt)               # (n_modes,)

    # phi_flat : (n_valid*Nc, n_modes)  — already unit norm, no extra step needed
    phi_flat = Q.T @ A * scale[np.newaxis, :]

    # Reshape to (n_modes, ny, nx, Nc), NaN at invalid points
    modes = np.full((n_modes, ny, nx, Nc), np.nan)

    for c in range(Nc):
        col_slice = phi_flat[c * n_valid:(c + 1) * n_valid, :]   # (n_valid, n_modes)
        for n in range(n_modes):
            plane = np.full((ny, nx), np.nan)
            plane[mask] = col_slice[:, n]
            modes[n, :, :, c] = plane

    # ------------------------------------------------------------------ #
    # 9. Temporal coefficients  a_n(t) = Q @ phi_n  (Nt, n_modes)        #
    # ------------------------------------------------------------------ #
    # Simple projection of each snapshot onto each spatial mode.
    # Reconstruction: u_fluct(t) = sum_n a_n(t) * phi_n -> Q[t,:] when
    # all Nt modes are used (residual -> 0).
    coeffs = Q @ phi_flat                               # (Nt, n_modes)

    # ------------------------------------------------------------------ #
    # Diagnostics: verify reconstruction fidelity on snapshot 0           #
    # ------------------------------------------------------------------ #
    rms_q0     = float(np.sqrt(np.mean(Q[0, :] ** 2)))
    fluct0_rec = coeffs[0, :] @ phi_flat.T              # (n_valid*Nc,)
    rms_rec    = float(np.sqrt(np.mean(fluct0_rec ** 2)))
    rms_resid  = float(np.sqrt(np.mean((Q[0, :] - fluct0_rec) ** 2)))
    print(f"[POD diagnostics] snapshot-0  RMS(Q)={rms_q0:.4e}  "
          f"RMS(reconstruction)={rms_rec:.4e}  RMS(residual)={rms_resid:.4e}")

    # ------------------------------------------------------------------ #
    # 12. Energy fractions                                                 #
    # ------------------------------------------------------------------ #
    total_energy  = np.sum(lam)
    energy_frac   = lam / total_energy if total_energy > 0 else np.zeros(n_modes)
    cumul_energy  = np.cumsum(energy_frac)

    return dict(
        eigenvalues  = lam,
        modes        = modes,
        coeffs       = coeffs,
        energy_frac  = energy_frac,
        cumul_energy = cumul_energy,
        mask         = mask,
        n_modes      = n_modes,
        Nc           = Nc,
        mean_U       = mean_U,
        mean_V       = mean_V,
        mean_W       = mean_W,
    )


def reconstruct_snapshot(pod_result, snapshot_idx, n_modes):
    """Reconstruct a velocity snapshot from the first n_modes POD modes.

    Parameters
    ----------
    pod_result   : dict returned by compute_pod
    snapshot_idx : int   index into the Nt dimension
    n_modes      : int   number of modes to use (clamped to pod_result['n_modes'])

    Returns
    -------
    U_rec, V_rec, W_rec : (ny, nx) arrays  (W_rec is None for 2D data)
        Reconstructed *total* velocity fields (mean + fluctuation).
    """
    modes   = pod_result["modes"]       # (n_modes_stored, ny, nx, Nc)
    coeffs  = pod_result["coeffs"]      # (Nt, n_modes_stored)
    Nc      = pod_result["Nc"]
    mean_U  = pod_result["mean_U"]
    mean_V  = pod_result["mean_V"]
    mean_W  = pod_result["mean_W"]

    n_modes = min(n_modes, pod_result["n_modes"])

    ny, nx = mean_U.shape

    # Sum of a_n(t) * phi_n over the retained modes
    # coeffs[snapshot_idx, :n_modes] : (n_modes,)
    # modes[:n_modes, :, :, c]       : (n_modes, ny, nx)
    c_vec = coeffs[snapshot_idx, :n_modes]            # (n_modes,)

    # Broadcast: (n_modes,1,1,1) * (n_modes,ny,nx,Nc) -> (n_modes,ny,nx,Nc), then sum
    fluct = np.nansum(
        c_vec[:, np.newaxis, np.newaxis, np.newaxis] * modes[:n_modes, :, :, :],
        axis=0
    )   # (ny, nx, Nc)

    U_rec = mean_U + fluct[:, :, 0]
    V_rec = mean_V + fluct[:, :, 1]

    if Nc == 3 and mean_W is not None:
        W_rec = mean_W + fluct[:, :, 2]
    else:
        W_rec = None

    return U_rec, V_rec, W_rec
