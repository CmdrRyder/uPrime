"""
core/anisotropy.py
------------------
Anisotropy invariant analysis.

Reynolds stress anisotropy tensor:
    bij = Rij / (2k) - delta_ij / 3
    where Rij = <ui' uj'>,  k = 0.5 * trace(Rij)

Invariants of bij:
    II  = bij * bji           (= b_ij b_ji, always negative for turbulence)
    III = bij * bjk * bki

Lumley triangle axes: -II vs III
Barycentric map: RGB coloring based on eigenvalues of bij
    C1c = lambda1 - lambda2       (1-component limit, red)
    C2c = 2*(lambda2 - lambda3)   (2-component limit, green)
    C3c = 3*lambda3 + 1           (3-component/isotropic limit, blue)

Reference: Banerjee et al. (2007), Phys. Fluids
"""

import numpy as np


def compute_reynolds_tensor(U, V, W):
    """
    Compute Reynolds stress tensor components from velocity arrays.

    Parameters
    ----------
    U, V, W : [ny, nx, Nt]  -- W must not be None

    Returns
    -------
    R : dict with keys 'uu','vv','ww','uv','uw','vw'
        each is a [ny, nx] array of time-averaged <ui'uj'>
    k : [ny, nx]  turbulent kinetic energy = 0.5*(uu+vv+ww)
    """
    # Fluctuations = instantaneous - time mean
    mean_u = np.nanmean(U, axis=2, keepdims=True)
    mean_v = np.nanmean(V, axis=2, keepdims=True)
    mean_w = np.nanmean(W, axis=2, keepdims=True)

    up = U - mean_u
    vp = V - mean_v
    wp = W - mean_w

    R = {
        "uu": np.nanmean(up * up, axis=2),
        "vv": np.nanmean(vp * vp, axis=2),
        "ww": np.nanmean(wp * wp, axis=2),
        "uv": np.nanmean(up * vp, axis=2),
        "uw": np.nanmean(up * wp, axis=2),
        "vw": np.nanmean(vp * wp, axis=2),
    }

    k = 0.5 * (R["uu"] + R["vv"] + R["ww"])
    return R, k


def compute_anisotropy_tensor(R, k):
    """
    Compute anisotropy tensor bij = Rij/(2k) - delta_ij/3

    Returns bij as dict with same keys as R, plus diagonal corrections.
    Also returns the full [ny, nx, 3, 3] tensor for eigenvalue computation.
    """
    ny, nx = k.shape

    # Safe division: where k is too small set to NaN
    k_safe = np.where(k > 1e-12, k, np.nan)

    b = np.full((ny, nx, 3, 3), np.nan)

    b[:, :, 0, 0] = R["uu"] / (2 * k_safe) - 1.0/3
    b[:, :, 1, 1] = R["vv"] / (2 * k_safe) - 1.0/3
    b[:, :, 2, 2] = R["ww"] / (2 * k_safe) - 1.0/3
    b[:, :, 0, 1] = R["uv"] / (2 * k_safe)
    b[:, :, 1, 0] = R["uv"] / (2 * k_safe)
    b[:, :, 0, 2] = R["uw"] / (2 * k_safe)
    b[:, :, 2, 0] = R["uw"] / (2 * k_safe)
    b[:, :, 1, 2] = R["vw"] / (2 * k_safe)
    b[:, :, 2, 1] = R["vw"] / (2 * k_safe)

    return b


def compute_invariants(b):
    """
    Compute Lumley invariants II and III from anisotropy tensor b [ny,nx,3,3].

    II  = -0.5 * sum_ij(bij * bji)   (defined positive here, so -II > 0)
    III =  (1/3) * sum_ijk(bij*bjk*bki)

    Returns
    -------
    II  : [ny, nx]  (negative, typically plotted as -II)
    III : [ny, nx]
    """
    ny, nx = b.shape[:2]
    II  = np.full((ny, nx), np.nan)
    III = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            bij = b[i, j]
            if np.any(np.isnan(bij)):
                continue
            II[i, j]  = np.trace(bij @ bij)
            III[i, j] = np.trace(bij @ bij @ bij)

    # Standard Lumley convention: -II (positive) vs III
    return -II, III


def compute_invariants_fast(b):
    """
    Vectorized version of compute_invariants using einsum.
    Much faster than the loop version for large grids.
    """
    # Invariants following paper convention (Eq. 7):
    #   I2 = -0.5 * bij * bji = -0.5 * trace(b^2)   [negative]
    #   I3 = det(bij)
    # Plot -I2 (positive) vs I3
    # Limits from paper:
    #   Isotropic:      I3=0,       -I2=0
    #   1-component:    I3=2/27,    -I2=1/3
    #   2-comp axisym:  I3=-1/108,  -I2=1/12

    ny, nx = b.shape[:2]
    neg_I2 = np.full((ny, nx), np.nan)
    I3     = np.full((ny, nx), np.nan)

    nan_mask = np.any(np.isnan(b), axis=(-2, -1))

    # Vectorized trace(b^2)
    b2     = np.einsum('...ij,...jk->...ik', b, b)
    trb2   = np.einsum('...ii->...', b2)
    neg_I2 = 0.5 * trb2          # -I2 = 0.5*trace(b^2), always >= 0

    # det(bij) computed via eigenvalues (numerically stable for 3x3)
    # For symmetric matrix: det = product of eigenvalues
    for i in range(ny):
        for j in range(nx):
            if nan_mask[i, j]:
                continue
            eigs     = np.linalg.eigvalsh(b[i, j])
            I3[i, j] = np.prod(eigs)

    neg_I2[nan_mask] = np.nan
    I3[nan_mask]     = np.nan

    return neg_I2, I3


def compute_barycentric(b):
    """
    Compute barycentric coordinates (C1c, C2c, C3c) and RGB color
    from anisotropy tensor b [ny, nx, 3, 3].

    Eigenvalues sorted descending: lambda1 >= lambda2 >= lambda3
    C1c = lambda1 - lambda2        (1-component, red)
    C2c = 2*(lambda2 - lambda3)    (2-component, green)
    C3c = 3*lambda3 + 1            (isotropic, blue)

    Reference: Banerjee et al. (2007)
    """
    ny, nx = b.shape[:2]

    C1c = np.full((ny, nx), np.nan)
    C2c = np.full((ny, nx), np.nan)
    C3c = np.full((ny, nx), np.nan)

    for i in range(ny):
        for j in range(nx):
            bij = b[i, j]
            if np.any(np.isnan(bij)):
                continue
            # Eigenvalues of symmetric matrix, sorted descending
            eigs = np.linalg.eigvalsh(bij)   # ascending order
            l3, l2, l1 = sorted(eigs)        # l1 >= l2 >= l3

            C1c[i, j] = l1 - l2
            C2c[i, j] = 2.0 * (l2 - l3)
            C3c[i, j] = 3.0 * l3 + 1.0

    # Clamp to [0, 1] to handle small numerical errors
    C1c = np.clip(C1c, 0, 1)
    C2c = np.clip(C2c, 0, 1)
    C3c = np.clip(C3c, 0, 1)

    # RGB: red=1-component, green=2-component, blue=isotropic
    RGB = np.stack([C1c, C2c, C3c], axis=-1)

    # Normalize each pixel so RGB sums to 1 (barycentric constraint)
    total = RGB.sum(axis=-1, keepdims=True)
    total = np.where(total > 0, total, 1.0)
    RGB   = RGB / total

    return C1c, C2c, C3c, RGB


def points_near_line(x, y, x0, y0, x1, y1):
    """
    Find grid points nearest to a line drawn from (x0,y0) to (x1,y1).

    Strategy: parameterize the line, sample Npts points along it,
    find the nearest grid point to each sample (no interpolation).
    Deduplicate so each grid point appears at most once.

    Returns
    -------
    rows, cols : arrays of grid indices along the line
    dist       : distance along the line for each point (mm)
    """
    # Number of samples along the line -- use max grid dimension
    Npts = max(x.shape[0], x.shape[1]) * 2

    t    = np.linspace(0, 1, Npts)
    xl   = x0 + t * (x1 - x0)
    yl   = y0 + t * (y1 - y0)

    rows = []
    cols = []

    for xi, yi in zip(xl, yl):
        dist2 = (x - xi)**2 + (y - yi)**2
        r, c  = np.unravel_index(np.argmin(dist2), dist2.shape)
        rows.append(r)
        cols.append(c)

    # Deduplicate while preserving order
    seen  = set()
    u_rows, u_cols = [], []
    for r, c in zip(rows, cols):
        if (r, c) not in seen:
            seen.add((r, c))
            u_rows.append(r)
            u_cols.append(c)

    u_rows = np.array(u_rows)
    u_cols = np.array(u_cols)

    # Distance along line from start point
    dist = np.sqrt(
        (x[u_rows, u_cols] - x0)**2 +
        (y[u_rows, u_cols] - y0)**2
    )

    return u_rows, u_cols, dist
