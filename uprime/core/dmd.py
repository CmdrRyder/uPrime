"""
uprime/core/dmd.py
-----------
Dynamic Mode Decomposition following Schmid (2010).
J. Fluid Mech. 656, 5-28. doi:10.1017/S0022112010001217

For stereo PIV: snapshot matrix is stacked [U; V; W] vertically.
For 2D PIV: stacked [U; V] or single component.
Rank truncation via thin SVD (POD pre-processing) for noise robustness.

DMD needs a complete time series at every grid point it uses, but invalid
vectors reach it as NaN that varies from snapshot to snapshot. Gaps are
therefore resolved in build_snapshot_matrix, on the same terms as
core.pod.compute_pod: grid points that are NaN in 10 % or more of the
snapshots are dropped, and the remaining gaps are filled with that point's own
temporal mean. compute_dmd itself refuses non-finite input.
"""

import numpy as np

MAX_NAN_FRAC = 0.10   # a grid point is dropped at or above this NaN fraction


def _components_used(U, V, W, component):
    """Return the arrays that `component` draws on, in stacking order."""
    if component == 'stacked':
        return [U, V] if W is None else [U, V, W]
    if component == 'U':
        return [U]
    if component == 'V':
        return [V]
    if component == 'W':
        if W is None:
            raise ValueError("W component not available for 2D dataset")
        return [W]
    raise ValueError(f"Unknown component: {component}")


def resolve_gaps(fields, mask=None, max_nan_frac=MAX_NAN_FRAC):
    """
    Drop unusable grid points and fill the remaining NaN gaps, as POD does.

    A grid point is kept only if `mask` marks it valid and fewer than
    `max_nan_frac` of its snapshots are NaN, counting a snapshot as NaN if any
    of `fields` is NaN there -- so every component keeps the same set of
    points. Gaps at the surviving points are filled with that point's own
    temporal mean. A field whose mean has already been subtracted is therefore
    filled with (numerically) zero, while a raw velocity field is filled with
    the local mean velocity instead of a false zero.

    Parameters
    ----------
    fields : sequence of [ny, nx, Nt] arrays -- what will enter the decomposition
    mask : [ny, nx] bool, True=valid, or None to start with every point valid
    max_nan_frac : float -- NaN fraction at which a point is dropped.
                   The 0.10 default matches core.pod.compute_pod.

    Returns
    -------
    filled : list of [ny, nx, Nt] arrays -- copies with the gaps filled.
             Values at dropped points are meaningless; mask_out excludes them.
    mask_out : [ny, nx] bool -- the reduced mask, True = point kept
    filled_fraction : float -- fraction of the retained values that were
                      gap-filled; 0.0 for data without gaps
    """
    fields = [np.asarray(f, dtype=float) for f in fields]
    ny, nx, Nt = fields[0].shape

    # A snapshot counts as a gap at a point if any component is missing there.
    gap_any = np.zeros((ny, nx, Nt), dtype=bool)
    nan_per_field = []
    for f in fields:
        g = np.isnan(f)
        nan_per_field.append(g)
        gap_any |= g

    mask_out = np.mean(gap_any, axis=2) < max_nan_frac
    if mask is not None:
        mask_out &= np.asarray(mask, dtype=bool)

    filled = []
    n_fill = 0
    for f, g in zip(fields, nan_per_field):
        if g.any():
            # Point's own temporal mean, computed without warning on points
            # that have no finite sample at all (those are dropped anyway).
            count = (~g).sum(axis=2)
            total = np.where(g, 0.0, f).sum(axis=2)
            mean_t = np.divide(total, count,
                               out=np.zeros((ny, nx), dtype=float),
                               where=count > 0)
            f = f.copy()
            f[g] = np.broadcast_to(mean_t[:, :, np.newaxis], f.shape)[g]
            n_fill += int(g[mask_out, :].sum())
        filled.append(f)

    n_used = int(mask_out.sum()) * Nt * len(fields)
    filled_fraction = (n_fill / n_used) if n_used else 0.0

    return filled, mask_out, filled_fraction


def build_snapshot_matrix(U, V, W, component='stacked', mask=None,
                          max_nan_frac=MAX_NAN_FRAC):
    """
    Build snapshot matrix X of shape [n_space, Nt] from velocity arrays.

    Gaps are resolved first by resolve_gaps, so `mask` is only the starting
    point: grid points that are NaN in `max_nan_frac` or more of the snapshots
    are dropped as well, and the reduced mask comes back in `info`. Pass that
    reduced mask -- not the dataset MASK -- to get_mode_components, or the
    modes will be mapped onto the wrong grid points.

    Parameters
    ----------
    U, V : [ny, nx, Nt] -- velocity, fluctuations or raw (see resolve_gaps)
    W    : [ny, nx, Nt] or None
    component : 'stacked', 'U', 'V', 'W'
    mask : [ny, nx] bool, True=valid, or None for every point valid
    max_nan_frac : float -- passed through to resolve_gaps

    Returns
    -------
    X    : [n_space, Nt] snapshot matrix, free of NaN
    n_per_component : int -- number of spatial points per component
                             (used to split stacked modes back into U/V/W)
    info : dict with keys
        'mask'            : [ny, nx] bool -- the reduced mask actually used
        'filled_fraction' : float -- fraction of values in X that were filled
        'n_components'    : int -- components stacked into X (1, 2 or 3)
    """
    ny, nx, Nt = U.shape

    used = _components_used(U, V, W, component)
    filled, mask_out, filled_fraction = resolve_gaps(
        used, mask=mask, max_nan_frac=max_nan_frac)

    flat_mask = mask_out.ravel()          # [ny*nx]
    n_per = int(flat_mask.sum())

    # each field: [ny, nx, Nt] -> [n_per, Nt]
    X = np.concatenate([f.reshape(ny * nx, Nt)[flat_mask, :] for f in filled],
                       axis=0)           # [n_per*n_comp, Nt]

    info = {
        'mask'           : mask_out,
        'filled_fraction': filled_fraction,
        'n_components'   : len(filled),
    }
    return X, n_per, info


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

    Raises
    ------
    ValueError
        If X holds any NaN or inf. Resolve the gaps first -- normally by
        building X with build_snapshot_matrix, which does it for you.
    """
    X = np.asarray(X)
    if not np.all(np.isfinite(X)):
        n_bad = int(np.count_nonzero(~np.isfinite(X)))
        raise ValueError(
            f"compute_dmd got {n_bad} non-finite value(s) of {X.size} in X. "
            "Build the snapshot matrix with core.dmd.build_snapshot_matrix, "
            "which drops unusable grid points and fills the remaining gaps, "
            "or remove the gaps before calling.")

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
    mask         : [ny, nx] bool -- the SAME reduced mask that
                   build_snapshot_matrix returned in info['mask'], whose
                   True count is n_per. Passing the unreduced dataset MASK
                   scatters the mode across the wrong grid points.

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
