"""Row orientation in the .dat loader.

load_grid flips rows only when y descends, so that row 0 always holds the
smallest y. load_single_file used to flip unconditionally. DaVis writes y
descending, so the two cancelled out on real files, but a .dat written with y
ascending loaded with the velocities mirrored vertically relative to the grid
-- silently, since every summary statistic is unchanged by a row flip.

load_grid now records its decision in header["flip_rows"] and load_single_file
follows it, so both row orders load to the same field.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import contextlib
import io

import numpy as np

from uprime.core.loader import load_dataset, load_grid, parse_header

_VARS = ('"x [mm]", "y [mm]", "velocity u [m/s]", '
         '"velocity v [m/s]", "isValid"')


def _write_series(folder, U, V, x1d, y1d, descending):
    """Write a .dat series with rows in either y order, same physical field."""
    os.makedirs(folder, exist_ok=True)
    ny, nx, nt = U.shape
    rows = range(ny - 1, -1, -1) if descending else range(ny)
    paths = []
    for t in range(nt):
        p = os.path.join(folder, f"snap_{t:04d}.dat")
        with open(p, "w", encoding="utf-8") as f:
            f.write('TITLE = "synthetic"\n')
            f.write(f"VARIABLES = {_VARS}\n")
            f.write(f'ZONE T="frame {t}", I={nx}, J={ny}, F=POINT\n')
            for r in rows:
                for c in range(nx):
                    f.write(f"{x1d[c]:.6f} {y1d[r]:.6f} "
                            f"{U[r, c, t]:.6f} {V[r, c, t]:.6f} 1\n")
        paths.append(p)
    return paths


def _field(ny=6, nx=7, nt=3):
    """A field that varies strongly with row, so a flip cannot go unnoticed."""
    y1d = np.linspace(0.0, 5.0, ny)
    x1d = np.linspace(0.0, 6.0, nx)
    # u depends on the row index only -> mirroring changes it everywhere.
    U = np.empty((ny, nx, nt))
    V = np.empty((ny, nx, nt))
    for t in range(nt):
        U[:, :, t] = (np.arange(ny)[:, None] + 1.0) * 10.0 + t
        V[:, :, t] = np.arange(nx)[None, :] * 0.5 - t
    return U, V, x1d, y1d


def _load_quiet(files):
    with contextlib.redirect_stdout(io.StringIO()):
        return load_dataset(files)


def test_ascending_and_descending_y_load_identically(tmp_path):
    U, V, x1d, y1d = _field()

    desc = _write_series(str(tmp_path / "desc"), U, V, x1d, y1d, descending=True)
    asc = _write_series(str(tmp_path / "asc"), U, V, x1d, y1d, descending=False)

    ds_desc = _load_quiet(desc)
    ds_asc = _load_quiet(asc)

    # Element by element, both orders give the same grid and the same data.
    assert np.array_equal(ds_desc["y"], ds_asc["y"])
    assert np.array_equal(ds_desc["x"], ds_asc["x"])
    np.testing.assert_allclose(ds_asc["U"], ds_desc["U"], rtol=0, atol=1e-5)
    np.testing.assert_allclose(ds_asc["V"], ds_desc["V"], rtol=0, atol=1e-5)

    # And both match the field that was written, with row 0 = smallest y.
    np.testing.assert_allclose(ds_asc["U"], U, rtol=0, atol=1e-5)
    assert ds_asc["y"][0, 0] < ds_asc["y"][-1, 0]


def test_load_grid_records_its_flip_decision(tmp_path):
    U, V, x1d, y1d = _field()

    for descending in (True, False):
        folder = str(tmp_path / f"d{int(descending)}")
        files = _write_series(folder, U, V, x1d, y1d, descending=descending)

        header = parse_header(files[0])
        assert "flip_rows" not in header      # set by load_grid, not the header
        _x, y = load_grid(files[0], header)

        assert header["flip_rows"] is descending
        assert y[0, 0] < y[-1, 0]             # convention holds either way


def test_load_single_file_defaults_to_flipping(tmp_path):
    """A caller that skips load_grid keeps the old unconditional flip."""
    from uprime.core.loader import load_single_file

    U, V, x1d, y1d = _field()
    files = _write_series(str(tmp_path / "desc"), U, V, x1d, y1d, descending=True)

    header = parse_header(files[0])
    assert "flip_rows" not in header
    u, _v, _w, _valid, _vort = load_single_file(files[0], header)

    # Rows were written y-descending, so the default flip gives row 0 = min y.
    np.testing.assert_allclose(u, U[:, :, 0], rtol=0, atol=1e-5)
