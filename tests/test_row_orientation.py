"""Grid orientation in the .dat loader.

uPrime's convention is row 0 = smallest y, column 0 = smallest x. load_grid
decides whether the file needs a row flip, a column flip, or both, and records
each decision in header["flip_rows"] / header["flip_cols"] so that
load_single_file reorders the data exactly as the grid was reordered.

Before the fix, load_single_file always flipped rows and never flipped
columns, whatever the grid did. DaVis writes y descending and x ascending, so
the two cancelled out on real exports and nothing ever showed; a file written
either axis the other way round loaded mirrored against its own x/y -- in the
row case silently, since no summary statistic changes under a flip.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import contextlib
import io

import numpy as np
import pytest

from uprime.core.loader import (load_dataset, load_grid, load_single_file,
                                parse_header)


def _vars(with_vort):
    cols = ['"x [mm]"', '"y [mm]"', '"velocity u [m/s]"', '"velocity v [m/s]"']
    if with_vort:
        cols.append('"Vorticity [1/s]"')
    cols.append('"isValid"')
    return ", ".join(cols)


def _write_series(folder, U, V, x1d, y1d, desc_y=False, desc_x=False,
                  vort=None, valid=None):
    """Write a .dat series in either row and column order, same physical field."""
    os.makedirs(folder, exist_ok=True)
    ny, nx, nt = U.shape
    rows = range(ny - 1, -1, -1) if desc_y else range(ny)
    cols = range(nx - 1, -1, -1) if desc_x else range(nx)
    paths = []
    for t in range(nt):
        p = os.path.join(folder, f"snap_{t:04d}.dat")
        with open(p, "w", encoding="utf-8") as f:
            f.write('TITLE = "synthetic"\n')
            f.write(f"VARIABLES = {_vars(vort is not None)}\n")
            f.write(f'ZONE T="frame {t}", I={nx}, J={ny}, F=POINT\n')
            for r in rows:
                for c in cols:
                    fields = [f"{x1d[c]:.6f}", f"{y1d[r]:.6f}",
                              f"{U[r, c, t]:.6f}", f"{V[r, c, t]:.6f}"]
                    if vort is not None:
                        fields.append(f"{vort[r, c, t]:.6f}")
                    ok = 1 if (valid is None or valid[r, c, t]) else 0
                    fields.append(str(ok))
                    f.write(" ".join(fields) + "\n")
        paths.append(p)
    return paths


def _field(ny=6, nx=7, nt=3):
    """A field that varies with both row and column, so any flip shows up."""
    y1d = np.linspace(0.0, 5.0, ny)
    x1d = np.linspace(0.0, 6.0, nx)
    U = np.empty((ny, nx, nt))
    V = np.empty((ny, nx, nt))
    vort = np.empty((ny, nx, nt))
    for t in range(nt):
        U[:, :, t] = (np.arange(ny)[:, None] + 1.0) * 10.0 + t
        V[:, :, t] = np.arange(nx)[None, :] * 0.5 - t
        vort[:, :, t] = (np.arange(ny)[:, None] * 100.0
                         + np.arange(nx)[None, :] + t)
    return U, V, vort, x1d, y1d


def _load_quiet(files):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_dataset(files)


# Every combination of row and column order, all describing the same field.
ORDERS = [
    pytest.param(False, False, id="y_asc_x_asc"),
    pytest.param(True, False, id="y_desc_x_asc"),
    pytest.param(False, True, id="y_asc_x_desc"),
    pytest.param(True, True, id="both_reversed"),
]


@pytest.mark.parametrize("desc_y,desc_x", ORDERS)
def test_every_row_and_column_order_loads_identically(tmp_path, desc_y, desc_x):
    U, V, vort, x1d, y1d = _field()

    ref = _load_quiet(_write_series(str(tmp_path / "ref"), U, V, x1d, y1d))
    got = _load_quiet(_write_series(str(tmp_path / "got"), U, V, x1d, y1d,
                                    desc_y=desc_y, desc_x=desc_x))

    for key in ("x", "y", "U", "V", "MASK"):
        np.testing.assert_allclose(np.asarray(got[key], dtype=float),
                                   np.asarray(ref[key], dtype=float),
                                   rtol=0, atol=1e-5, err_msg=key)

    # And it matches the field as written, in uPrime's convention.
    np.testing.assert_allclose(got["U"], U, rtol=0, atol=1e-5)
    np.testing.assert_allclose(got["V"], V, rtol=0, atol=1e-5)
    np.testing.assert_allclose(got["x"][0, :], x1d, rtol=0, atol=1e-9)
    np.testing.assert_allclose(got["y"][:, 0], y1d, rtol=0, atol=1e-9)
    assert got["y"][0, 0] < got["y"][-1, 0]
    assert got["x"][0, 0] < got["x"][0, -1]


@pytest.mark.parametrize("desc_y,desc_x", ORDERS)
def test_vorticity_and_validity_flip_with_the_velocities(tmp_path, desc_y,
                                                         desc_x):
    U, V, vort, x1d, y1d = _field()

    valid = np.ones(U.shape, dtype=bool)
    valid[4, 5, :] = False          # invalid in every snapshot
    valid[1, 2, :1] = False         # invalid in the first snapshot only

    files = _write_series(str(tmp_path / "case"), U, V, x1d, y1d,
                          desc_y=desc_y, desc_x=desc_x, vort=vort, valid=valid)
    header = parse_header(files[0])
    _x, _y = load_grid(files[0], header)
    u, v, w, got_valid, got_vort = load_single_file(files[0], header)

    assert w is None
    np.testing.assert_allclose(u, np.where(valid[:, :, 0], U[:, :, 0], np.nan),
                               rtol=0, atol=1e-5)
    np.testing.assert_allclose(got_vort,
                               np.where(valid[:, :, 0], vort[:, :, 0], np.nan),
                               rtol=0, atol=1e-4)
    assert np.array_equal(got_valid, valid[:, :, 0])

    ds = _load_quiet(files)
    expected_mask = np.ones(U.shape[:2], dtype=bool)
    expected_mask[4, 5] = False
    assert np.array_equal(ds["MASK"], expected_mask)


@pytest.mark.parametrize("desc_y,desc_x", ORDERS)
def test_load_grid_records_both_decisions(tmp_path, desc_y, desc_x):
    U, V, _vort, x1d, y1d = _field()
    files = _write_series(str(tmp_path / "case"), U, V, x1d, y1d,
                          desc_y=desc_y, desc_x=desc_x)

    header = parse_header(files[0])
    assert "flip_rows" not in header      # set by load_grid, not the header
    assert "flip_cols" not in header
    _x, y = load_grid(files[0], header)

    assert header["flip_rows"] is desc_y
    assert header["flip_cols"] is desc_x
    assert y[0, 0] < y[-1, 0]


def test_flags_default_to_the_pre_existing_behaviour(tmp_path):
    """A caller that skips load_grid gets rows flipped and columns not.

    That is exactly what load_single_file did before the flags existed, so
    such a caller is unaffected by this change.
    """
    U, V, _vort, x1d, y1d = _field()
    files = _write_series(str(tmp_path / "davis"), U, V, x1d, y1d,
                          desc_y=True, desc_x=False)

    header = parse_header(files[0])
    assert "flip_rows" not in header and "flip_cols" not in header
    u, _v, _w, _valid, _vort = load_single_file(files[0], header)

    # Written y-descending, x-ascending: the defaults give the right answer.
    np.testing.assert_allclose(u, U[:, :, 0], rtol=0, atol=1e-5)


def test_single_row_and_single_column_grids_are_not_flipped(tmp_path):
    """A degenerate axis has no order to detect; neither flag may be set."""
    U, V, _vort, _x, _y = _field(ny=1, nx=5)
    x1d = np.linspace(0.0, 4.0, 5)
    y1d = np.array([2.0])
    files = _write_series(str(tmp_path / "row"), U, V, x1d, y1d)

    header = parse_header(files[0])
    load_grid(files[0], header)
    assert header["flip_rows"] is False
    assert header["flip_cols"] is False
