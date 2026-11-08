"""Gap handling in DMD, and the filled_fraction both decompositions report.

After the loader fix, invalid vectors reach DMD as NaN that varies from
snapshot to snapshot. DMD needs a complete time series at every point it uses,
so core.dmd.build_snapshot_matrix now resolves the gaps the way POD does:
points that are NaN in 10 % or more of the snapshots are dropped, and the rest
are filled with that point's own temporal mean.

Before the fix, compute_dmd had no gap handling at all and died inside the SVD
on NaN input, while DMDWorker papered over it with np.nan_to_num -- which fills
with literal zero and therefore injected large false spikes whenever
"subtract mean" was off and the matrix held raw velocity.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest

from uprime.core.dmd import build_snapshot_matrix, compute_dmd, scale_to_physical

FS = 1000.0     # sampling frequency [Hz]
F0 = 50.0       # travelling-wave frequency [Hz]
NT = 128
U0 = 5.0        # mean streamwise velocity, so "subtract mean" has work to do
AMP = 1.0
GAP_FRAC = 0.05

BIN_HZ = FS / NT        # 7.8125 Hz -- the "one frequency bin" tolerance


def _travelling_wave(ny=10, nx=16, nt=NT):
    """A clean travelling wave at F0: u = U0 + A cos(kx x - 2 pi F0 t)."""
    x = np.linspace(0.0, 1.0, nx)
    t = np.arange(nt) / FS
    phase = (2 * np.pi * 2.0 * x)[None, :, None] - (2 * np.pi * F0 * t)[None, None, :]
    phase = np.broadcast_to(phase, (ny, nx, nt))
    U = U0 + AMP * np.cos(phase)
    V = AMP * np.sin(phase)
    return np.array(U, dtype=float), np.array(V, dtype=float)


def _punch_gaps(arrays, frac=GAP_FRAC, seed=7):
    """NaN the same randomly chosen vectors in every array, per snapshot."""
    rng = np.random.default_rng(seed)
    gap = rng.random(arrays[0].shape) < frac
    out = []
    for a in arrays:
        a = a.copy()
        a[gap] = np.nan
        out.append(a)
    return out, gap


def _dominant_frequency(U, V, subtract_mean=True, mask=None, rank=12):
    """Frequency [Hz] of the largest-amplitude oscillatory DMD mode."""
    if subtract_mean:
        U = U - np.nanmean(U, axis=2, keepdims=True)
        V = V - np.nanmean(V, axis=2, keepdims=True)

    X, n_per, info = build_snapshot_matrix(U, V, None,
                                           component="stacked", mask=mask)
    result = scale_to_physical(compute_dmd(X, rank=rank), FS)

    f = np.abs(result["frequencies_hz"])
    amp = result["amplitudes"].copy()
    # The mean, when it has not been subtracted, shows up as a zero-frequency
    # mode that would otherwise dominate the amplitude ranking.
    amp[f < 0.5 * BIN_HZ] = -np.inf
    return float(f[int(np.argmax(amp))]), n_per, info


# ---------------------------------------------------------------------------
# 1. DMD runs on gappy data and still finds the right frequency
# ---------------------------------------------------------------------------

def test_dmd_with_gaps_recovers_the_known_frequency():
    U, V = _travelling_wave()
    (Ug, Vg), gap = _punch_gaps([U, V])

    assert np.isnan(Ug).any(), "the test data must actually have gaps"

    freq, n_per, info = _dominant_frequency(Ug, Vg)

    # Runs at all, and the dominant frequency is within one bin of the truth.
    assert freq == pytest.approx(F0, abs=BIN_HZ)
    # The reduced mask and n_per agree, which is what get_mode_components needs.
    assert int(info["mask"].sum()) == n_per
    assert info["n_components"] == 2
    assert info["filled_fraction"] > 0.0


# ---------------------------------------------------------------------------
# 2. "Subtract mean" off no longer fills raw velocity with zero
# ---------------------------------------------------------------------------

def test_dmd_without_mean_subtraction_matches_the_subtracted_case():
    """Gaps must be filled with the point's own mean, not with zero velocity.

    Filling raw velocity with 0 puts a -U0 spike at every gap, which is what
    np.nan_to_num used to do in DMDWorker. Filling with the point's temporal
    mean leaves the oscillation untouched, so both settings agree.
    """
    U, V = _travelling_wave()
    (Ug, Vg), _gap = _punch_gaps([U, V])

    f_sub, _, _ = _dominant_frequency(Ug, Vg, subtract_mean=True)
    f_raw, _, _ = _dominant_frequency(Ug, Vg, subtract_mean=False)

    assert f_sub == pytest.approx(F0, abs=BIN_HZ)
    assert f_raw == pytest.approx(f_sub, abs=BIN_HZ)


def test_dmd_worker_leaves_gap_handling_to_the_core():
    """The worker must pass the reduced mask and filled_fraction on."""
    pytest.importorskip("PyQt6.QtCore")
    from uprime.core.workers import DMDWorker

    U, V = _travelling_wave()
    (Ug, Vg), _gap = _punch_gaps([U, V])
    mask = np.ones(Ug.shape[:2], dtype=bool)

    for subtract_mean in (True, False):
        worker = DMDWorker(Ug, Vg, None, mask=mask, component="stacked",
                           n_modes=12, subtract_mean=subtract_mean, fs=FS)
        result = worker.compute()

        assert int(result["mask"].sum()) == result["n_per"]
        assert result["filled_fraction"] > 0.0
        assert np.all(np.isfinite(result["frequencies_hz"]))

        f = np.abs(result["frequencies_hz"])
        amp = result["amplitudes"].copy()
        amp[f < 0.5 * BIN_HZ] = -np.inf
        assert float(f[int(np.argmax(amp))]) == pytest.approx(F0, abs=BIN_HZ)


# ---------------------------------------------------------------------------
# 3. compute_dmd refuses non-finite input instead of dying in the SVD
# ---------------------------------------------------------------------------

def test_compute_dmd_rejects_nan():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 30))
    X[5, 7] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        compute_dmd(X, rank=5)


def test_compute_dmd_rejects_inf():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 30))
    X[2, 2] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        compute_dmd(X, rank=5)


# ---------------------------------------------------------------------------
# 5. filled_fraction reports what was actually filled
# ---------------------------------------------------------------------------

def test_dmd_filled_fraction_zero_without_gaps_and_matches_with_gaps():
    U, V = _travelling_wave()

    _X, _n_per, clean = build_snapshot_matrix(U, V, None, component="stacked")
    assert clean["filled_fraction"] == 0.0

    (Ug, Vg), gap = _punch_gaps([U, V])
    _X, _n_per, gappy = build_snapshot_matrix(Ug, Vg, None, component="stacked")

    # Close to the injected fraction. Points that lost 10 % or more of their
    # snapshots are dropped, so the retained points are slightly cleaner than
    # the series as a whole and the reported value sits just below GAP_FRAC.
    assert gappy["filled_fraction"] == pytest.approx(GAP_FRAC, abs=0.01)
    assert gappy["filled_fraction"] <= float(gap.mean()) + 1e-12


def test_pod_filled_fraction_zero_without_gaps_and_matches_with_gaps():
    from uprime.core.pod import compute_pod

    U, V = _travelling_wave()

    clean = compute_pod(U, V, None, n_modes=5)
    assert clean["filled_fraction"] == 0.0

    (Ug, Vg), _gap = _punch_gaps([U, V])
    gappy = compute_pod(Ug, Vg, None, n_modes=5)
    assert gappy["filled_fraction"] == pytest.approx(GAP_FRAC, abs=0.01)


def test_gappy_points_are_dropped_from_the_mask():
    """A point missing most of its snapshots must not reach the decomposition."""
    U, V = _travelling_wave()
    Ug, Vg = U.copy(), V.copy()

    dead = (4, 5)          # NaN in half the snapshots -> dropped
    survivor = (2, 3)      # NaN in one snapshot only   -> kept
    Ug[dead[0], dead[1], ::2] = np.nan
    Vg[dead[0], dead[1], ::2] = np.nan
    Ug[survivor[0], survivor[1], 0] = np.nan
    Vg[survivor[0], survivor[1], 0] = np.nan

    _X, n_per, info = build_snapshot_matrix(Ug, Vg, None, component="stacked")

    assert not bool(info["mask"][dead])
    assert bool(info["mask"][survivor])
    assert n_per == Ug.shape[0] * Ug.shape[1] - 1
