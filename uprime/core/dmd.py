"""
core/dmd.py
-----------
Dynamic Mode Decomposition following Schmid (2010).
J. Fluid Mech. 656, 5-28. doi:10.1017/S0022112010001217

For stereo PIV: snapshot matrix is stacked [U; V; W] vertically.
For 2D PIV: stacked [U; V] or single component.
Rank truncation via thin SVD (POD pre-processing) for noise robustness.
"""

import numpy as np


def build_snapshot_matrix(U, V, W, component='stacked', mask=None):
    """
    Build snapshot matrix X of shape [n_space, Nt] from velocity arrays.

    Parameters
    ----------
    U, V : [ny, nx, Nt] -- velocity fluctuations (mean already subtracted)
    W    : [ny, nx, Nt] or None
    component : 'stacked', 'U', 'V', 'W'
    mask : [ny, nx] bool, True=valid -- used to flatten only valid pixels

    Returns
    -------
    X    : [n_space, Nt] snapshot matrix
    n_per_component : int -- number of spatial points per component
                             (used to split stacked modes back into U/V/W)
    """
    ny, nx, Nt = U.shape

    if mask is None:
        mask = np.ones((ny, nx), dtype=bool)

    flat_mask = mask.ravel()  # [ny*nx]

    def _flat(arr):
        # arr: [ny, nx, Nt] -> [n_valid, Nt]
        return arr.reshape(ny * nx, Nt)[flat_mask, :]

    n_per = int(flat_mask.sum())

    if component == 'stacked':
        parts = [_flat(U), _flat(V)]
        if W is not None:
            parts.append(_flat(W))
        X = np.concatenate(parts, axis=0)  # [n_per*n_comp, Nt]
    elif component == 'U':
        X = _flat(U)
    elif component == 'V':
        X = _flat(V)
    elif component == 'W':
        if W is None:
            raise ValueError("W component not available for 2D dataset")
        X = _flat(W)
    else:
        raise ValueError(f"Unknown component: {component}")

    return X, n_per


def compute_dmd(X, rank=None):
    """
    Standard DMD algorithm (Schmid 2010, Tu et al. 2014 exact DMD variant).

    Parameters
    ----------
    X    : [n_space, Nt] snapshot matrix, columns are snapshots
    rank : int or None -- SVD truncation rank. If None, auto-select as
           min(n_space, Nt-1, 50) for noise robustness.

    Returns
    -------
    dict with keys:
        'modes'       : [n_space, r] complex -- DMD mode spatial structures
        'eigenvalues' : [r] complex -- discrete-time eigenvalues mu
        'amplitudes'  : [r] float -- mode amplitudes |b|
        'frequencies' : [r] float -- oscillation frequencies [rad/sample]
                        multiply by fs/(2*pi) for Hz
        'growth_rates': [r] float -- growth/decay rates [per sample]
                        positive = growing, negative = decaying
        'rank'        : int -- actual rank used
    """
    n_space, Nt = X.shape

    # Split into X1 (first Nt-1 snapshots) and X2 (last Nt-1 snapshots)
    X1 = X[:, :-1]   # [n_space, Nt-1]
    X2 = X[:, 1:]    # [n_space, Nt-1]

    # Auto rank selection
    if rank is None:
        rank = min(n_space, Nt - 1, 50)

    # Step 1: Thin SVD of X1
    U_svd, sigma, Vh = np.linalg.svd(X1, full_matrices=False)
    # Truncate to rank r
    r = min(rank, len(sigma))
    U_r     = U_svd[:, :r]          # [n_space, r]
    sigma_r = sigma[:r]              # [r]
    V_r     = Vh[:r, :].conj().T    # [Nt-1, r]

    # Step 2: Project onto POD subspace
    # A_tilde = U_r^H @ X2 @ V_r @ diag(1/sigma_r)
    A_tilde = (U_r.conj().T @ X2) @ V_r @ np.diag(1.0 / sigma_r)

    # Step 3: Eigendecomposition of A_tilde
    eigenvalues, W_eig = np.linalg.eig(A_tilde)

    # Step 4: Reconstruct full spatial modes (exact DMD)
    modes = X2 @ V_r @ np.diag(1.0 / sigma_r) @ W_eig
    # Normalize each mode to unit norm
    norms = np.linalg.norm(modes, axis=0, keepdims=True)
    norms[norms < 1e-12] = 1.0
    modes = modes / norms   # [n_space, r]

    # Step 5: Compute amplitudes via least-squares fit to first snapshot
    b, _, _, _ = np.linalg.lstsq(modes, X[:, 0], rcond=None)
    amplitudes = np.abs(b)   # [r]

    # Step 6: Convert discrete eigenvalues to continuous (per-sample) values
    log_mu = np.log(eigenvalues)
    growth_rates_discrete = log_mu.real   # [r] -- multiply by fs for 1/s
    frequencies_discrete  = log_mu.imag  # [r] -- multiply by fs/(2pi) for Hz

    return {
        'modes'       : modes,
        'eigenvalues' : eigenvalues,
        'amplitudes'  : amplitudes,
        'growth_rates': growth_rates_discrete,
        'frequencies' : frequencies_discrete,
        'rank'        : r,
    }


def scale_to_physical(dmd_result, fs):
    """
    Convert discrete-time DMD results to physical units.

    Parameters
    ----------
    dmd_result : dict from compute_dmd
    fs         : float -- sampling frequency [Hz]

    Returns a copy of dmd_result with added keys:
        'frequencies_hz'   : [r] float -- oscillation frequency [Hz]
        'growth_rates_phys': [r] float -- growth rate [1/s]
    """
    result = dict(dmd_result)
    result['frequencies_hz']    = dmd_result['frequencies'] * fs / (2 * np.pi)
    result['growth_rates_phys'] = dmd_result['growth_rates'] * fs
    return result


def get_mode_components(mode_flat, n_per, n_components, ny, nx, mask):
    """
    Reconstruct spatial mode into [ny, nx] arrays for each component.

    Parameters
    ----------
    mode_flat    : [n_space] complex -- flattened mode vector
    n_per        : int -- spatial points per component
    n_components : int -- 2 for 2D, 3 for stereo
    ny, nx       : int -- grid dimensions
    mask         : [ny, nx] bool -- valid pixel mask

    Returns
    -------
    list of [ny, nx] complex arrays, one per component
    """
    flat_mask = mask.ravel()
    components = []
    for i in range(n_components):
        chunk = mode_flat[i * n_per:(i + 1) * n_per]  # [n_valid] complex
        field = np.full(ny * nx, np.nan, dtype=complex)
        field[flat_mask] = chunk
        components.append(field.reshape(ny, nx))
    return components
