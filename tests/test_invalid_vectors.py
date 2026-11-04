"""Invalid-vector handling in the loaders.

The rule under test, for every loader:

1. A vector that is invalid in a snapshot is NaN in that snapshot.
2. MASK marks a grid point invalid only if it is invalid in EVERY snapshot.
3. valid, valid_frac and MASK_LOADED stay consistent with MASK.

Before the fix, the .dat loader kept only the first snapshot's isValid column,
so invalid vectors in every later snapshot were read as real data (typically
0.0), and the .mat loader blanked a point for the whole series if it was
invalid in any single snapshot.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import contextlib
import gc
import io

import numpy as np
import pytest

from uprime.core.dataset_utils import get_masked
from uprime.core.loader import cleanup_memmap, load_dataset

U0 = 5.0        # mean streamwise velocity [m/s]
SIGMA = 0.5     # so that <u'u'> = 0.25 (m/s)^2
INVALID_WRITTEN_AS = 0.0

_VARS = ('"x [mm]", "y [mm]", "velocity u [m/s]", '
         '"velocity v [m/s]", "velocity w [m/s]", "isValid"')


def _write_series(folder, U, V, W, VALID, x1d, y1d):
    """Write one DaVis-style Tecplot .dat per snapshot.

    Invalid vectors are written as INVALID_WRITTEN_AS with isValid = 0, which
    is what a real DaVis export does.
    """
    os.makedirs(folder, exist_ok=True)
    ny, nx, nt = U.shape
    paths = []
    for t in range(nt):
        p = os.path.join(folder, f"snap_{t:04d}.dat")
        u = np.where(VALID[:, :, t], U[:, :, t], INVALID_WRITTEN_AS)
        v = np.where(VALID[:, :, t], V[:, :, t], INVALID_WRITTEN_AS)
        w = np.where(VALID[:, :, t], W[:, :, t], INVALID_WRITTEN_AS)
        with open(p, "w", encoding="utf-8") as f:
            f.write('TITLE = "synthetic"\n')
            f.write(f"VARIABLES = {_VARS}\n")
            f.write(f'ZONE T="frame {t}", I={nx}, J={ny}, F=POINT\n')
            # Rows top-to-bottom (y descending), as DaVis writes them.
            # load_grid flips rows only when y descends, while load_single_file
            # flips unconditionally -- an ascending-y file would come back with
            # the data upside down relative to the grid.
            for r in range(ny - 1, -1, -1):
                for c in range(nx):
                    f.write(f"{x1d[c]:.6f} {y1d[r]:.6f} "
                            f"{u[r, c]:.6f} {v[r, c]:.6f} {w[r, c]:.6f} "
                            f"{1 if VALID[r, c, t] else 0}\n")
        paths.append(p)
    return paths


def _field(ny=10, nx=12, nt=120, seed=4):
    rng = np.random.default_rng(seed)
    U = U0 + SIGMA * rng.standard_normal((ny, nx, nt))
    V = SIGMA * rng.standard_normal((ny, nx, nt))
    W = SIGMA * rng.standard_normal((ny, nx, nt))
    x1d = np.linspace(0.0, 11.0, nx)
    y1d = np.linspace(0.0, 9.0, ny)
    return U, V, W, x1d, y1d


def _load_quiet(files):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_dataset(files)


# ---------------------------------------------------------------------------
# 1. Invalid vectors become NaN, and the statistics are right
# ---------------------------------------------------------------------------

def test_dat_invalid_vectors_become_nan_and_do_not_bias_statistics(tmp_path):
    """Each snapshot's own isValid column must be honoured.

    Before the fix the invalid vectors of every snapshot after the first were
    read as 0.0 real data, pulling the mean down and inflating <u'u'>.
    """
    U, V, W, x1d, y1d = _field()
    rng = np.random.default_rng(99)
    VALID = rng.random(U.shape) >= 0.05
    VALID[:, :, 0] = True          # first snapshot fully valid, as in the bug

    files = _write_series(str(tmp_path / "case"), U, V, W, VALID, x1d, y1d)
    ds = _load_quiet(files)

    Um = get_masked(ds, "U")

    # NaN exactly where the data said invalid -- not one point more or fewer.
    assert np.array_equal(np.isnan(Um), ~VALID)

    # The sentinel value must not have survived anywhere.
    assert not np.any(Um[~VALID] == INVALID_WRITTEN_AS)

    # Statistics match the underlying field, not the zero-padded one.
    assert np.nanmean(Um) == pytest.approx(U0, abs=0.02)
    assert np.nanmean(np.nanvar(Um, axis=2)) == pytest.approx(SIGMA ** 2, abs=0.02)

    # V and W get the same treatment.
    assert np.array_equal(np.isnan(get_masked(ds, "V")), ~VALID)
    assert np.array_equal(np.isnan(get_masked(ds, "W")), ~VALID)


# ---------------------------------------------------------------------------
# 2. MASK marks a point invalid only if it is invalid in every snapshot
# ---------------------------------------------------------------------------

def test_mask_requires_invalid_in_every_snapshot(tmp_path):
    U, V, W, x1d, y1d = _field(nt=30)
    VALID = np.ones(U.shape, dtype=bool)

    sometimes = (2, 3)      # invalid in some snapshots, including the first
    always = (6, 8)         # invalid in every snapshot
    VALID[sometimes[0], sometimes[1], :5] = False
    VALID[always[0], always[1], :] = False

    files = _write_series(str(tmp_path / "case"), U, V, W, VALID, x1d, y1d)
    ds = _load_quiet(files)

    # Invalid in snapshot 0 but valid later -> still a usable grid point.
    assert bool(ds["MASK"][sometimes])
    # Invalid everywhere -> masked out.
    assert not bool(ds["MASK"][always])

    # Everything else untouched.
    expected = np.ones(VALID.shape[:2], dtype=bool)
    expected[always] = False
    assert np.array_equal(ds["MASK"], expected)

    # The companion keys agree with MASK.
    assert np.array_equal(ds["valid"], ds["MASK"])
    assert np.array_equal(ds["MASK_LOADED"], ds["MASK"])
    assert np.array_equal(ds["valid_frac"], ds["MASK"].astype(np.float32))

    # Per-snapshot NaN is still applied at the sometimes-invalid point.
    Um = get_masked(ds, "U")
    assert np.all(np.isnan(Um[sometimes[0], sometimes[1], :5]))
    assert np.all(np.isfinite(Um[sometimes[0], sometimes[1], 5:]))


# ---------------------------------------------------------------------------
# 3. .mat files with a 3-D isValid, under both mask conventions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("convention", ["valid", "invalid"])
def test_mat_3d_isvalid_both_conventions(tmp_path, convention):
    scipy_io = pytest.importorskip("scipy.io")
    from uprime.core.mat_loader import (auto_detect_mapping, detect_variables,
                                        load_mat_dataset)

    ny, nx, nt = 9, 11, 40           # ny != nx so a transpose would be caught
    rng = np.random.default_rng(11)
    U = U0 + SIGMA * rng.standard_normal((ny, nx, nt))
    V = SIGMA * rng.standard_normal((ny, nx, nt))

    valid = np.ones((ny, nx, nt), dtype=bool)
    sometimes, always = (2, 3), (6, 8)
    valid[sometimes[0], sometimes[1], :5] = False
    valid[always[0], always[1], :] = False

    # Under the "invalid" convention the stored variable means the opposite.
    stored = valid if convention == "valid" else ~valid

    path = str(tmp_path / "case.mat")
    scipy_io.savemat(path, {
        "x": np.linspace(0.0, 11.0, nx),
        "y": np.linspace(0.0, 9.0, ny),
        "u": np.where(valid, U, INVALID_WRITTEN_AS),
        "v": np.where(valid, V, INVALID_WRITTEN_AS),
        "isValid": stored.astype(np.float64),
    })

    var_info, _is_v73 = detect_variables(path)
    mapping, complete = auto_detect_mapping(var_info)
    assert complete, mapping

    with contextlib.redirect_stdout(io.StringIO()):
        ds, _status = load_mat_dataset(path, mapping, mask_convention=convention)

    assert ds["U"].shape == (ny, nx, nt)

    Um = get_masked(ds, "U")
    # Rule 1: NaN exactly in the snapshots where the vector was invalid.
    assert np.array_equal(np.isnan(Um), ~valid)
    # Rule 2: a point invalid in only some snapshots survives in MASK.
    assert bool(ds["MASK"][sometimes])
    assert not bool(ds["MASK"][always])
    # Rule 3
    assert np.array_equal(ds["valid"], ds["MASK"])
    assert np.array_equal(ds["MASK_LOADED"], ds["MASK"])


# ---------------------------------------------------------------------------
# 4. The memory-mapped branch follows the same rule
# ---------------------------------------------------------------------------

def test_memmap_branch_applies_the_same_rule(tmp_path, monkeypatch):
    """Force the memmap path by lowering SIZE_THRESHOLD, not by making a big file."""
    from uprime.core import loader as loader_mod

    monkeypatch.setattr(loader_mod, "SIZE_THRESHOLD", 0)

    U, V, W, x1d, y1d = _field(ny=8, nx=9, nt=40)
    rng = np.random.default_rng(5)
    VALID = rng.random(U.shape) >= 0.05
    VALID[:, :, 0] = True
    always = (3, 4)
    VALID[always[0], always[1], :] = False

    files = _write_series(str(tmp_path / "case"), U, V, W, VALID, x1d, y1d)
    ds = _load_quiet(files)

    try:
        # Confirm we really exercised the memmap path.
        assert ds["_memmap_path"] is not None
        assert isinstance(ds["U"], np.memmap)

        Um = get_masked(ds, "U")
        assert np.array_equal(np.isnan(Um), ~VALID)
        assert not bool(ds["MASK"][always])

        expected = np.ones(VALID.shape[:2], dtype=bool)
        expected[always] = False
        assert np.array_equal(ds["MASK"], expected)
        assert np.array_equal(ds["valid"], ds["MASK"])
    finally:
        # Drop every reference to the memmaps before deleting the files;
        # Windows refuses to remove a file that is still mapped.
        ds["U"] = ds["V"] = ds["W"] = None
        gc.collect()
        cleanup_memmap(ds)
