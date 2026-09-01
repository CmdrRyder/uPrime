"""
core/probability.py
--------------------
Probability analysis of velocity fields: per-point flow-direction probability,
probability density functions, binary space-time maps, and quadrant analysis.

Pure NumPy, no Qt. Every function that walks the time axis is **chunked**, because
datasets above 4 GB are memory-mapped (see ``core/loader.py``) and a full-array
boolean temporary would force the whole thing into RAM.

The NaN rule (critical)
-----------------------
``np.nan < 0`` is ``False``, so ``np.nanmean(field < 0, axis=2)`` silently divides
by ``Nt`` and biases probabilities low wherever PIV dropped a vector -- exactly the
near-wall region these maps are read in. Every accumulator here counts valid samples
per point separately and divides by that count, returning ``np.nan`` where the count
is zero. The dataset mask is applied as a boolean AND against ``mask_2d[:, :, None]``
inside the chunk loop; ``core.dataset_utils.get_masked`` is deliberately NOT used
because it copies the whole array.

References
----------
Simpson, R.L. (1989), Turbulent boundary-layer separation, Annu. Rev. Fluid Mech.
    21, 205-234 -- backflow coefficient chi and the detachment states at
    chi = 0.01 (incipient), 0.20 (intermittent transitory), 0.50 (transitory).
Lu, S.S. & Willmarth, W.W. (1973), Measurements of the structure of the Reynolds
    stress in a turbulent boundary layer, JFM 60, 481-511 -- quadrant analysis and
    hole size.
Wallace, J.M. (2016), Quadrant analysis in turbulence research, Annu. Rev. Fluid
    Mech. 48, 131-158.
(Verify DOIs before they go into the manual or a paper.)
"""

import numpy as np

CHUNK_DEFAULT = 200   # frames per block


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _iter_chunks(Nt, chunk):
    """Yield (t0, t1) frame-block bounds covering [0, Nt)."""
    chunk = max(1, int(chunk))
    for t0 in range(0, Nt, chunk):
        yield t0, min(t0 + chunk, Nt)


def _emit(progress_cb, frac):
    if progress_cb is not None:
        progress_cb(int(round(100 * frac)))


def _gather_values(field, region, mask_2d, t0, t1):
    """Return a 1D float array of the valid values in frames [t0, t1) over
    ``region`` (see module docstring / the analysis spec for the region grammar).
    Only the region's spatial extent is read, so this stays memmap-friendly."""
    if region is None:
        blk = np.asarray(field[:, :, t0:t1])
        valid = np.isfinite(blk)
        if mask_2d is not None:
            valid &= mask_2d[:, :, None]
        return blk[valid]

    kind = region[0]
    if kind == "point":
        _, r, c = region
        if mask_2d is not None and not mask_2d[r, c]:
            return np.empty(0, dtype=float)
        blk = np.asarray(field[r, c, t0:t1])
        return blk[np.isfinite(blk)]

    if kind == "index":
        _, rows, cols = region
        blk = np.asarray(field[rows, cols, t0:t1])          # [P, tlen]
        valid = np.isfinite(blk)
        if mask_2d is not None:
            valid &= mask_2d[rows, cols][:, None]
        return blk[valid]

    if kind == "roi":
        _, r0, r1, c0, c1 = region
        blk = np.asarray(field[r0:r1 + 1, c0:c1 + 1, t0:t1])
        valid = np.isfinite(blk)
        if mask_2d is not None:
            valid &= mask_2d[r0:r1 + 1, c0:c1 + 1][:, :, None]
        return blk[valid]

    raise ValueError(f"unknown region spec: {region!r}")


def _region_points(region, ny, nx):
    """Resolve a region to flat (rows, cols) index arrays of the contributing
    grid points. Used by the quadrant joint histogram and hole sweep."""
    if region is None:
        rr, cc = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        return rr.ravel(), cc.ravel()
    kind = region[0]
    if kind == "point":
        _, r, c = region
        return np.array([r]), np.array([c])
    if kind == "index":
        _, rows, cols = region
        return np.asarray(rows), np.asarray(cols)
    if kind == "roi":
        _, r0, r1, c0, c1 = region
        rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1),
                             indexing="ij")
        return rr.ravel(), cc.ravel()
    raise ValueError(f"unknown region spec: {region!r}")


def _roi_bounds(region, ny, nx):
    """Return inclusive (r0, r1, c0, c1, is_roi) for a region.

    ``None`` -> the whole domain. ``('roi', r0, r1, c0, c1)`` -> those bounds,
    clamped to the array so a single row/column or an off-grid drag is safe.
    """
    if region is None:
        return 0, ny - 1, 0, nx - 1, False
    if region[0] == "roi":
        _, a, b, c, d = region
        r0, r1 = int(np.clip(min(a, b), 0, ny - 1)), int(np.clip(max(a, b), 0, ny - 1))
        c0, c1 = int(np.clip(min(c, d), 0, nx - 1)), int(np.clip(max(c, d), 0, nx - 1))
        return r0, r1, c0, c1, True
    raise ValueError(f"region must be None or ('roi', r0, r1, c0, c1); got {region!r}")


# --------------------------------------------------------------------------- #
# 3.2  Flow-direction probability
# --------------------------------------------------------------------------- #

def compute_direction_probability(field, mask_2d, deadband=0.0,
                                  chunk=CHUNK_DEFAULT, progress_cb=None,
                                  region=None):
    """
    Per-point probability that a velocity component is forward or reverse.

    Parameters
    ----------
    field     : [ny, nx, Nt] velocity component (may be a np.memmap)
    mask_2d   : [ny, nx] bool, True = valid, or None
    deadband  : float >= 0, epsilon in the same units as `field`
    chunk     : frames per block
    progress_cb : callable(int 0..100) or None
    region    : None (whole domain) or ('roi', r0, r1, c0, c1) with inclusive
                grid indices. Only the ROI is READ and computed — the block is
                sliced ``field[r0:r1+1, c0:c1+1, t0:t1]`` (a basic slice, so it
                stays memmap-friendly and reads less data).

    Returns
    -------
    dict with keys
        'p_forward'       [ny, nx] float64, NaN where n_valid == 0
        'p_reverse'       [ny, nx]
        'p_indeterminate' [ny, nx]   (all zeros when deadband == 0)
        'n_valid'         [ny, nx] int32
        'deadband'        float
        'region'          the region argument (for the GUI / export header)

    Outputs keep the FULL domain shape. Points **outside** the ROI are not
    computed and are returned as ``np.nan`` (float) / ``0`` (n_valid) — distinct
    from a point that WAS computed but had no valid samples (also NaN, but its
    ``n_valid`` count is a real 0 inside the ROI). The ROI extent is the way a
    reader tells the two NaNs apart.

    Classification (eps = float(deadband)):
        eps == 0.0 : forward q >= 0, reverse q < 0, no indeterminate state.
                     Reduces to the strict sign test bit for bit.
        eps  > 0.0 : forward q > eps, reverse q < -eps, indeterminate |q| <= eps.

    Denominator convention: all three probabilities divide by ``n_valid``, so
    ``p_forward + p_reverse + p_indeterminate == 1`` at every valid point.
    """
    ny, nx, Nt = field.shape
    eps = float(deadband)
    r0, r1, c0, c1, _is_roi = _roi_bounds(region, ny, nx)
    sub = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    sh = (r1 - r0 + 1, c1 - c0 + 1)

    nv = np.zeros(sh, dtype=np.int32)
    nf = np.zeros(sh, dtype=np.int32)
    nr = np.zeros(sh, dtype=np.int32)
    msub = None if mask_2d is None else mask_2d[sub]

    for t0, t1 in _iter_chunks(Nt, chunk):
        blk   = np.asarray(field[r0:r1 + 1, c0:c1 + 1, t0:t1])
        valid = np.isfinite(blk)
        if msub is not None:
            valid &= msub[:, :, None]
        nv += valid.sum(axis=2, dtype=np.int32)
        if eps == 0.0:
            nf += (valid & (blk >= 0.0)).sum(axis=2, dtype=np.int32)
            nr += (valid & (blk < 0.0)).sum(axis=2, dtype=np.int32)
        else:
            nf += (valid & (blk > eps)).sum(axis=2, dtype=np.int32)
            nr += (valid & (blk < -eps)).sum(axis=2, dtype=np.int32)
        _emit(progress_cb, t1 / Nt)

    # Embed the sub-region into full-domain arrays: 0 outside (counts), NaN
    # outside (probabilities), so plotting / x-y grids / export stay full-size.
    n_valid = np.zeros((ny, nx), dtype=np.int32)
    n_fwd = np.zeros((ny, nx), dtype=np.int32)
    n_rev = np.zeros((ny, nx), dtype=np.int32)
    n_valid[sub] = nv
    n_fwd[sub] = nf
    n_rev[sub] = nr

    p_forward = np.full((ny, nx), np.nan, dtype=np.float64)
    p_reverse = np.full((ny, nx), np.nan, dtype=np.float64)
    p_indeterminate = np.full((ny, nx), np.nan, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = nv.astype(np.float64)
        denom[nv == 0] = np.nan
        p_forward[sub] = nf / denom
        p_reverse[sub] = nr / denom
        p_indeterminate[sub] = (nv - nf - nr) / denom

    return {
        "p_forward": p_forward,
        "p_reverse": p_reverse,
        "p_indeterminate": p_indeterminate,
        "n_valid": n_valid,
        "deadband": eps,
        "region": region,
    }


# --------------------------------------------------------------------------- #
# 3.3  Histogram (two passes) + moment statistics
# --------------------------------------------------------------------------- #

def estimate_bin_edges(field, mask_2d, region, nbins=101,
                       stride=10, robust=True):
    """
    Pass 1. Walk every ``stride``-th frame to find the value range.
    robust=True  -> use the 0.1 and 99.9 percentiles, then pad by 2% of the span
    robust=False -> use min/max
    Returns [nbins + 1] edges.
    """
    Nt = field.shape[2]
    stride = max(1, int(stride))
    chunks = []
    for t in range(0, Nt, stride):
        v = _gather_values(field, region, mask_2d, t, t + 1)
        if v.size:
            chunks.append(v)

    if not chunks:
        return np.linspace(0.0, 1.0, nbins + 1)

    vals = np.concatenate(chunks)
    if robust:
        lo, hi = np.percentile(vals, [0.1, 99.9])
    else:
        lo, hi = float(vals.min()), float(vals.max())

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        centre = float(np.nanmean(vals)) if vals.size else 0.0
        lo, hi = centre - 0.5, centre + 0.5

    span = hi - lo
    pad = 0.02 * span if robust else 0.0
    return np.linspace(lo - pad, hi + pad, nbins + 1)


def accumulate_histogram(field, mask_2d, region, bin_edges,
                         chunk=CHUNK_DEFAULT, progress_cb=None):
    """
    Pass 2. Returns (counts [nbins] int64, n_valid int).

    Honours the same NaN and mask rules as ``compute_direction_probability``.
    """
    Nt = field.shape[2]
    nbins = len(bin_edges) - 1
    counts = np.zeros(nbins, dtype=np.int64)
    n_valid = 0

    for t0, t1 in _iter_chunks(Nt, chunk):
        vals = _gather_values(field, region, mask_2d, t0, t1)
        if vals.size:
            c, _ = np.histogram(vals, bins=bin_edges)
            counts += c.astype(np.int64)
            n_valid += int(vals.size)
        _emit(progress_cb, t1 / Nt)

    return counts, n_valid


def histogram_stats(counts, bin_edges):
    """Return dict: n, mean, std, skewness, kurtosis, min, max.

    Moments are computed from the binned distribution using bin centres.
    Kurtosis is the NON-EXCESS definition (Gaussian = 3); label it that way in the
    GUI. ``min``/``max`` are the edges of the occupied bin range.
    """
    counts = np.asarray(counts, dtype=np.float64)
    centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    n = float(counts.sum())
    if n == 0:
        nan = float("nan")
        return {"n": 0, "mean": nan, "std": nan, "skewness": nan,
                "kurtosis": nan, "min": nan, "max": nan}

    mean = float(np.sum(counts * centres) / n)
    var = float(np.sum(counts * (centres - mean) ** 2) / n)
    std = float(np.sqrt(var))
    if std > 0:
        skew = float(np.sum(counts * (centres - mean) ** 3) / n / std ** 3)
        kurt = float(np.sum(counts * (centres - mean) ** 4) / n / std ** 4)
    else:
        skew = float("nan")
        kurt = float("nan")

    occ = np.where(counts > 0)[0]
    vmin = float(bin_edges[occ[0]])
    vmax = float(bin_edges[occ[-1] + 1])

    return {"n": int(n), "mean": mean, "std": std, "skewness": skew,
            "kurtosis": kurt, "min": vmin, "max": vmax}


# --------------------------------------------------------------------------- #
# 3.4  Space-time extraction along a grid-aligned line
# --------------------------------------------------------------------------- #

def extract_space_time(field, x, y, x0, y0, x1, y1,
                       direction, avg_band=0):
    """
    Extract q(s, t) along a grid-aligned line.

    direction : "x" (horizontal line, snaps to nearest row at y0, spans x0..x1)
                "y" (vertical line, snaps to nearest col at x0, spans y0..y1)
    avg_band  : average this many grid lines either side, before any sign test

    The band average is taken on the RAW velocity with ``np.nanmean`` (so a single
    bad line does not poison the slice) BEFORE any sign test -- matching
    ``core/spatiotemporal_spectra.compute_st_spectra``, not a majority vote after
    binarisation. Free-line mode is not supported: rotate the dataset with the
    Transform module first.

    Returns
    -------
    q_st : [Ns, Nt] float32
    s_mm : [Ns] coordinate along the line in mm (x for "x", y for "y")
    info : dict with the snapped row/col index and the actual line coordinate
    """
    ny, nx = x.shape
    band = int(avg_band)

    if direction == "x":
        row0 = int(np.argmin(np.abs(y[:, 0] - y0)))
        r0 = max(0, row0 - band)
        r1 = min(ny - 1, row0 + band)
        x_min, x_max = min(x0, x1), max(x0, x1)
        cols = np.where((x[0, :] >= x_min) & (x[0, :] <= x_max))[0]
        if cols.size == 0:
            cols = np.arange(nx)
        stack = np.stack([np.asarray(field[line, cols, :], dtype=np.float32)
                          for line in range(r0, r1 + 1)], axis=0)   # [nb, Ns, Nt]
        q_st = np.nanmean(stack, axis=0).astype(np.float32)          # [Ns, Nt]
        s_mm = x[0, cols].astype(np.float64)
        info = {"direction": "x", "row": row0, "coord_mm": float(y[row0, 0]),
                "band": band, "n_lines": r1 - r0 + 1}
    elif direction == "y":
        col0 = int(np.argmin(np.abs(x[0, :] - x0)))
        c0 = max(0, col0 - band)
        c1 = min(nx - 1, col0 + band)
        y_min, y_max = min(y0, y1), max(y0, y1)
        rows = np.where((y[:, 0] >= y_min) & (y[:, 0] <= y_max))[0]
        if rows.size == 0:
            rows = np.arange(ny)
        stack = np.stack([np.asarray(field[rows, line, :], dtype=np.float32)
                          for line in range(c0, c1 + 1)], axis=0)   # [nb, Ns, Nt]
        q_st = np.nanmean(stack, axis=0).astype(np.float32)
        s_mm = y[rows, 0].astype(np.float64)
        info = {"direction": "y", "col": col0, "coord_mm": float(x[0, col0]),
                "band": band, "n_lines": c1 - c0 + 1}
    else:
        raise ValueError("direction must be 'x' or 'y' (free lines unsupported)")

    return q_st, s_mm, info


# --------------------------------------------------------------------------- #
# 3.5  Binarise a space-time slice
# --------------------------------------------------------------------------- #

def binarize_space_time(q_st, deadband=0.0):
    """
    Return state [Ns, Nt] int8 using codes chosen so the array can index a
    4-entry ListedColormap directly:

        0 = reverse        (q < -eps, or q < 0 when eps == 0)
        1 = forward        (q >  eps, or q >= 0 when eps == 0)
        2 = indeterminate  (|q| <= eps, only reachable when eps > 0)
        3 = invalid        (NaN)

    Invalid points never fall into the forward bin; this is a data-quality readout
    as much as a physics one.
    """
    q = np.asarray(q_st)
    eps = float(deadband)
    finite = np.isfinite(q)
    state = np.full(q.shape, 3, dtype=np.int8)   # default: invalid

    if eps == 0.0:
        fwd = finite & (q >= 0.0)
        rev = finite & (q < 0.0)
        state[rev] = 0
        state[fwd] = 1
    else:
        rev = finite & (q < -eps)
        fwd = finite & (q > eps)
        ind = finite & (np.abs(q) <= eps)
        state[rev] = 0
        state[fwd] = 1
        state[ind] = 2

    return state


# --------------------------------------------------------------------------- #
# 3.6  Quadrant analysis
# --------------------------------------------------------------------------- #

def _quadrant_means_rms(field_a, field_b, mask_2d, chunk, progress_cb, frac0, frac1,
                        bounds=None):
    """First chunked pass: per-point n_valid, means and rms of (a, b) using only
    samples where BOTH components are finite (and unmasked). When ``bounds`` is
    given only that ROI slice is read; the outputs are full-size with 0 / NaN
    outside."""
    ny, nx, Nt = field_a.shape
    r0, r1, c0, c1 = (0, ny - 1, 0, nx - 1) if bounds is None else bounds
    sub = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    sh = (r1 - r0 + 1, c1 - c0 + 1)
    n = np.zeros(sh, dtype=np.int64)
    sa = np.zeros(sh); sb = np.zeros(sh); saa = np.zeros(sh); sbb = np.zeros(sh)
    msub = None if mask_2d is None else mask_2d[sub]

    for t0, t1 in _iter_chunks(Nt, chunk):
        a = np.asarray(field_a[r0:r1 + 1, c0:c1 + 1, t0:t1], dtype=np.float64)
        b = np.asarray(field_b[r0:r1 + 1, c0:c1 + 1, t0:t1], dtype=np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        if msub is not None:
            valid &= msub[:, :, None]
        a0 = np.where(valid, a, 0.0)
        b0 = np.where(valid, b, 0.0)
        n += valid.sum(axis=2)
        sa += a0.sum(axis=2); sb += b0.sum(axis=2)
        saa += (a0 * a0).sum(axis=2); sbb += (b0 * b0).sum(axis=2)
        _emit(progress_cb, frac0 + (frac1 - frac0) * (t1 / Nt))

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = n.astype(np.float64)
        denom[n == 0] = np.nan
        mean_a_s = sa / denom
        mean_b_s = sb / denom
        a_rms_s = np.sqrt(np.maximum(saa / denom - mean_a_s ** 2, 0.0))
        b_rms_s = np.sqrt(np.maximum(sbb / denom - mean_b_s ** 2, 0.0))

    n_full = np.zeros((ny, nx), dtype=np.int64); n_full[sub] = n
    mean_a = np.full((ny, nx), np.nan); mean_a[sub] = mean_a_s
    mean_b = np.full((ny, nx), np.nan); mean_b[sub] = mean_b_s
    a_rms = np.full((ny, nx), np.nan); a_rms[sub] = a_rms_s
    b_rms = np.full((ny, nx), np.nan); b_rms[sub] = b_rms_s
    return n_full, mean_a, mean_b, a_rms, b_rms


def compute_quadrant(field_a, field_b, mask_2d, hole=0.0,
                     region=None, joint_bins=101,
                     chunk=CHUNK_DEFAULT, progress_cb=None):
    """
    Quadrant analysis of a fluctuation pair (a', b'), conventionally (u', v').

    Quadrant numbering (standard, x streamwise and y wall-normal):
        Q1 : a'>0, b'>0  outward interaction
        Q2 : a'<0, b'>0  ejection
        Q3 : a'<0, b'<0  inward interaction
        Q4 : a'>0, b'<0  sweep

    Hole size H excludes samples with |a'b'| <= H * a_rms * b_rms, where the rms
    values are per-point. H = 0 includes everything. The temporal means and rms
    are found in a first chunked pass; the quadrant sums in a second. The full
    fluctuation array is never materialised -- the cached 2D mean is subtracted
    inside the chunk loop.

    Returns
    -------
    dict with
        'time_frac'    list of 4 [ny, nx]  fraction of valid samples in each quadrant
        'stress_frac'  list of 4 [ny, nx]  <a'b'>_i / <a'b'>  per quadrant
        'joint_hist'   [nb, nb] int64  joint histogram over `region`
        'edges_a', 'edges_b'         [nb + 1] each
        'a_rms', 'b_rms'  [ny, nx]
        'hole'         float
        'region'       the region argument

    ``region`` restricts the whole computation (maps, means/rms and joint
    histogram): only that ROI slice is read, and every ``[ny, nx]`` output keeps
    the full domain shape with ``np.nan`` outside the ROI (time/stress fractions,
    a_rms, b_rms) so downstream plotting and export stay full-size.
    """
    ny, nx, Nt = field_a.shape
    H = float(hole)
    r0, r1, c0, c1, _is_roi = _roi_bounds(region, ny, nx)
    bounds = (r0, r1, c0, c1)
    sub = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    sh = (r1 - r0 + 1, c1 - c0 + 1)

    n_valid, mean_a, mean_b, a_rms, b_rms = _quadrant_means_rms(
        field_a, field_b, mask_2d, chunk, progress_cb, 0.0, 0.45, bounds=bounds)

    thr_s = H * a_rms[sub] * b_rms[sub]          # per-point hole threshold on |a'b'|
    ma = mean_a[sub][:, :, None]
    mb = mean_b[sub][:, :, None]
    msub = None if mask_2d is None else mask_2d[sub]

    q_count = [np.zeros(sh, dtype=np.int64) for _ in range(4)]
    q_stress = [np.zeros(sh) for _ in range(4)]
    tot_stress = np.zeros(sh)

    for t0, t1 in _iter_chunks(Nt, chunk):
        a = np.asarray(field_a[r0:r1 + 1, c0:c1 + 1, t0:t1], dtype=np.float64)
        b = np.asarray(field_b[r0:r1 + 1, c0:c1 + 1, t0:t1], dtype=np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        if msub is not None:
            valid &= msub[:, :, None]
        ap = a - ma
        bp = b - mb
        ab = ap * bp
        tot_stress += np.where(valid, ab, 0.0).sum(axis=2)

        keep = valid & (np.abs(ab) > thr_s[:, :, None])   # hole exclusion
        quads = [
            (ap > 0) & (bp > 0),   # Q1
            (ap < 0) & (bp > 0),   # Q2
            (ap < 0) & (bp < 0),   # Q3
            (ap > 0) & (bp < 0),   # Q4
        ]
        for i, q in enumerate(quads):
            sel = keep & q
            q_count[i] += sel.sum(axis=2)
            q_stress[i] += np.where(sel, ab, 0.0).sum(axis=2)
        _emit(progress_cb, 0.45 + 0.45 * (t1 / Nt))

    time_frac = [np.full((ny, nx), np.nan) for _ in range(4)]
    stress_frac = [np.full((ny, nx), np.nan) for _ in range(4)]
    with np.errstate(invalid="ignore", divide="ignore"):
        nsub = n_valid[sub].astype(np.float64); nsub[n_valid[sub] == 0] = np.nan
        sden = tot_stress.copy(); sden[tot_stress == 0] = np.nan
        for i in range(4):
            time_frac[i][sub] = q_count[i] / nsub
            stress_frac[i][sub] = q_stress[i] / sden

    # Joint histogram over `region` (region points lie inside the ROI, where the
    # embedded means are valid).
    edges_a, edges_b, joint_hist = _quadrant_joint_hist(
        field_a, field_b, mask_2d, mean_a, mean_b, region, joint_bins,
        chunk, progress_cb, 0.90, 1.0)

    return {
        "time_frac": time_frac,
        "stress_frac": stress_frac,
        "joint_hist": joint_hist,
        "edges_a": edges_a,
        "edges_b": edges_b,
        "a_rms": a_rms,
        "b_rms": b_rms,
        "hole": H,
        "region": region,
    }


def _quadrant_joint_hist(field_a, field_b, mask_2d, mean_a, mean_b, region,
                         joint_bins, chunk, progress_cb, frac0, frac1):
    """2D histogram of (a', b') over the region points, symmetric about zero."""
    ny, nx, Nt = field_a.shape
    rows, cols = _region_points(region, ny, nx)
    ma = mean_a[rows, cols][:, None]
    mb = mean_b[rows, cols][:, None]
    mv = None if mask_2d is None else mask_2d[rows, cols][:, None]

    # symmetric edges from a strided estimate of |a'|, |b'|
    amax = bmax = 0.0
    for t in range(0, Nt, max(1, Nt // 20 or 1)):
        a = np.asarray(field_a[rows, cols, t:t + 1], dtype=np.float64) - ma
        b = np.asarray(field_b[rows, cols, t:t + 1], dtype=np.float64) - mb
        good = np.isfinite(a) & np.isfinite(b)
        if mv is not None:
            good &= mv
        if good.any():
            amax = max(amax, float(np.abs(a[good]).max()))
            bmax = max(bmax, float(np.abs(b[good]).max()))
    amax = amax or 1.0
    bmax = bmax or 1.0
    edges_a = np.linspace(-amax, amax, joint_bins + 1)
    edges_b = np.linspace(-bmax, bmax, joint_bins + 1)

    joint = np.zeros((joint_bins, joint_bins), dtype=np.int64)
    for t0, t1 in _iter_chunks(Nt, chunk):
        a = np.asarray(field_a[rows, cols, t0:t1], dtype=np.float64) - ma
        b = np.asarray(field_b[rows, cols, t0:t1], dtype=np.float64) - mb
        good = np.isfinite(a) & np.isfinite(b)
        if mv is not None:
            good &= mv
        if good.any():
            hh, _, _ = np.histogram2d(a[good], b[good], bins=[edges_a, edges_b])
            joint += hh.astype(np.int64)
        _emit(progress_cb, frac0 + (frac1 - frac0) * (t1 / Nt))

    return edges_a, edges_b, joint


def quadrant_hole_sweep(field_a, field_b, mask_2d, region,
                        holes=None, chunk=CHUNK_DEFAULT, progress_cb=None):
    """Return stress_frac [4, nH] averaged over ``region``, for a curve of the
    fractional stress contribution against hole size (Lu & Willmarth 1973).

    The denominator is the total mean covariance over the region (all valid
    samples, no hole), so at H = 0 the four quadrant fractions sum to 1.
    """
    if holes is None:
        holes = np.arange(0, 10.5, 0.5)
    holes = np.asarray(holes, dtype=np.float64)
    nH = len(holes)
    ny, nx, Nt = field_a.shape
    r0, r1, c0, c1, _is_roi = _roi_bounds(region, ny, nx)

    _n, mean_a, mean_b, a_rms, b_rms = _quadrant_means_rms(
        field_a, field_b, mask_2d, chunk, progress_cb, 0.0, 0.4,
        bounds=(r0, r1, c0, c1))

    rows, cols = _region_points(region, ny, nx)
    ma = mean_a[rows, cols][:, None]
    mb = mean_b[rows, cols][:, None]
    rms_prod = (a_rms[rows, cols] * b_rms[rows, cols])[:, None]
    mv = None if mask_2d is None else mask_2d[rows, cols][:, None]

    q_sum = np.zeros((4, nH))
    total = 0.0
    for t0, t1 in _iter_chunks(Nt, chunk):
        a = np.asarray(field_a[rows, cols, t0:t1], dtype=np.float64) - ma
        b = np.asarray(field_b[rows, cols, t0:t1], dtype=np.float64) - mb
        good = np.isfinite(a) & np.isfinite(b)
        if mv is not None:
            good &= mv
        ab = np.where(good, a * b, 0.0)
        total += ab.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(rms_prod > 0, np.abs(ab) / rms_prod, np.inf)
        quads = [(a > 0) & (b > 0), (a < 0) & (b > 0),
                 (a < 0) & (b < 0), (a > 0) & (b < 0)]
        for i, q in enumerate(quads):
            base = good & q
            vals = ab[base]
            rats = ratio[base]
            for h in range(nH):
                incl = rats > holes[h]
                if incl.any():
                    q_sum[i, h] += vals[incl].sum()
        _emit(progress_cb, 0.4 + 0.6 * (t1 / Nt))

    with np.errstate(invalid="ignore", divide="ignore"):
        denom = total if total != 0 else np.nan
        stress_frac = q_sum / denom
    return stress_frac, holes
