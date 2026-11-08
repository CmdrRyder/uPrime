"""Grid orientation in the .mat loader.

uPrime's convention is row 0 = smallest y and column 0 = smallest x.
core.loader.load_grid and core.davis_io both enforce it; mat_loader used to
take the file's own order as given. A .mat written the other way round
therefore loaded mirrored relative to a .dat or .vc7 case -- self-consistent,
so nothing ever warned, but not comparable.

Each test writes the same physical field in several row and column orders and
asserts they all load to identical arrays.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import contextlib
import io

import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from uprime.core.mat_loader import (auto_detect_mapping, detect_variables,
                                    load_mat_dataset)

NY, NX, NT = 9, 11, 6          # NY != NX so a transpose cannot hide


def _field():
    """A field that varies strongly along y, so a row flip cannot go unnoticed."""
    y1d = np.linspace(0.0, 8.0, NY)
    x1d = np.linspace(0.0, 10.0, NX)
    U = np.empty((NY, NX, NT))
    V = np.empty((NY, NX, NT))
    W = np.empty((NY, NX, NT))
    for t in range(NT):
        U[:, :, t] = (np.arange(NY)[:, None] + 1.0) * 10.0 + t
        V[:, :, t] = np.arange(NX)[None, :] * 0.5 - t
        W[:, :, t] = (np.arange(NY)[:, None] + 1.0) * 0.1
    return U, V, W, x1d, y1d


def _reorder(arr, desc_y, desc_x):
    """Reverse the row and/or column axis as the file is to be written."""
    sl = (slice(None, None, -1) if desc_y else slice(None),
          slice(None, None, -1) if desc_x else slice(None))
    return arr[sl + (Ellipsis,)]


# Every combination of row and column order, all describing the same field.
ORDERS = [
    pytest.param(False, False, id="y_asc_x_asc"),
    pytest.param(True, False, id="y_desc_x_asc"),
    pytest.param(False, True, id="y_asc_x_desc"),
    pytest.param(True, True, id="both_reversed"),
]
WRITERS = ["classic", "v73"]


def _write_classic(path, U, V, W, x1d, y1d, desc_y=False, desc_x=False,
                   valid=None):
    """Write a classic (v5) .mat, rows and columns in either order."""
    y2d = np.tile(y1d.reshape(-1, 1), (1, NX))
    x2d = np.tile(x1d, (NY, 1))
    out = {
        "x": _reorder(x2d, desc_y, desc_x),
        "y": _reorder(y2d, desc_y, desc_x),
        "u": _reorder(U, desc_y, desc_x),
        "v": _reorder(V, desc_y, desc_x),
        "w": _reorder(W, desc_y, desc_x),
    }
    if valid is not None:
        out["isValid"] = _reorder(valid, desc_y, desc_x).astype(np.float64)
    scipy_io.savemat(path, out)
    return path


def _write_v73(path, U, V, W, x1d, y1d, desc_y=False, desc_x=False,
               valid=None):
    """Write a MATLAB v7.3 file, so mat_loader takes its chunked h5py path.

    v7.3 is HDF5 with a 512-byte userblock. scipy reads the version from bytes
    124:128 and raises NotImplementedError for major == 2, which is exactly how
    mat_loader decides to use h5py. HDF5 stores the reverse of the MATLAB
    dimension order, so every array is transposed on the way in.
    """
    h5py = pytest.importorskip("h5py")

    y2d = np.tile(y1d.reshape(-1, 1), (1, NX))
    x2d = np.tile(x1d, (NY, 1))
    R = lambda a: _reorder(a, desc_y, desc_x)
    out = {
        "x": R(x2d).T,                           # (Ny,Nx) -> (Nx,Ny)
        "y": R(y2d).T,
        "u": R(U).transpose(2, 1, 0),            # (Ny,Nx,Nt) -> (Nt,Nx,Ny)
        "v": R(V).transpose(2, 1, 0),
        "w": R(W).transpose(2, 1, 0),
    }
    if valid is not None:
        v = R(valid)
        out["isValid"] = (v.transpose(2, 1, 0) if v.ndim == 3 else v.T)

    with h5py.File(path, "w", userblock_size=512) as fh:
        for name, arr in out.items():
            fh.create_dataset(name, data=np.ascontiguousarray(arr, dtype=np.float64))

    desc = b"MATLAB 7.3 MAT-file, Platform: TEST, Created by: uPrime test suite"
    with open(path, "r+b") as fh:
        # 0-115 description, 116-123 subsystem offset, 124-127 version + "IM"
        fh.write(desc.ljust(116, b" ") + b" " * 8 + bytes([0x00, 0x02]) + b"IM")
    return path


def _load(path, force_isvalid=None, **kw):
    """Load a .mat through the normal auto-detected mapping.

    `force_isvalid` names the validity variable explicitly. auto_detect_mapping
    only matches a strictly 3-D isValid, so a static 2-D mask is reachable only
    the way the GUI reaches it: the user picks it in the load dialog.
    """
    var_info, _is_v73 = detect_variables(path)
    mapping, complete = auto_detect_mapping(var_info)
    assert complete, mapping
    if force_isvalid is not None:
        mapping = dict(mapping, isValid=force_isvalid)
    with contextlib.redirect_stdout(io.StringIO()):
        ds, _status = load_mat_dataset(path, mapping, **kw)
    return ds


def _assert_same(ds_ref, ds_got, U, V, W, x1d, y1d):
    """Both orders give identical arrays, in uPrime's convention."""
    for key in ("x", "y", "U", "V", "W", "MASK", "valid", "MASK_LOADED"):
        a, b = ds_ref[key], ds_got[key]
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), key

    # And they match the field as written: row 0 = min y, column 0 = min x.
    np.testing.assert_allclose(ds_got["U"], U, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ds_got["V"], V, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ds_got["W"], W, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ds_got["y"][:, 0], y1d, rtol=0, atol=1e-9)
    np.testing.assert_allclose(ds_got["x"][0, :], x1d, rtol=0, atol=1e-9)
    assert ds_got["y"][0, 0] < ds_got["y"][-1, 0]
    assert ds_got["x"][0, 0] < ds_got["x"][0, -1]


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("desc_y,desc_x", ORDERS)
def test_every_row_and_column_order_loads_identically(tmp_path, writer,
                                                      desc_y, desc_x):
    U, V, W, x1d, y1d = _field()
    write = _write_classic if writer == "classic" else _write_v73

    ref = _load(write(str(tmp_path / "ref.mat"), U, V, W, x1d, y1d))
    got = _load(write(str(tmp_path / "got.mat"), U, V, W, x1d, y1d,
                      desc_y=desc_y, desc_x=desc_x))

    _assert_same(ref, got, U, V, W, x1d, y1d)
    assert got["header"]["flip_rows"] is desc_y
    assert got["header"]["flip_cols"] is desc_x


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("desc_y,desc_x", ORDERS)
def test_three_d_isvalid_flips_with_the_data(tmp_path, writer, desc_y, desc_x):
    """The per-snapshot isValid path must flip too, or NaN lands on wrong points."""
    U, V, W, x1d, y1d = _field()
    write = _write_classic if writer == "classic" else _write_v73

    valid = np.ones((NY, NX, NT), dtype=bool)
    sometimes, always = (2, 3), (6, 8)
    valid[sometimes[0], sometimes[1], :2] = False
    valid[always[0], always[1], :] = False

    ref = _load(write(str(tmp_path / "ref.mat"), U, V, W, x1d, y1d, valid=valid))
    got = _load(write(str(tmp_path / "got.mat"), U, V, W, x1d, y1d,
                      desc_y=desc_y, desc_x=desc_x, valid=valid))

    for key in ("U", "V", "W", "MASK", "x", "y"):
        assert np.array_equal(np.asarray(ref[key]), np.asarray(got[key]),
                              equal_nan=True), key

    # NaN sits on exactly the points the field was written with.
    assert np.array_equal(np.isnan(got["U"]), ~valid)
    assert bool(got["MASK"][sometimes])
    assert not bool(got["MASK"][always])


@pytest.mark.parametrize("writer", WRITERS)
@pytest.mark.parametrize("desc_y,desc_x", ORDERS)
def test_static_two_d_isvalid_flips_with_the_data(tmp_path, writer, desc_y,
                                                  desc_x):
    """A 2-D isValid bypasses _apply_frame, so it needs its own flip."""
    U, V, W, x1d, y1d = _field()
    write = _write_classic if writer == "classic" else _write_v73

    valid = np.ones((NY, NX), dtype=bool)
    valid[6, 8] = False
    valid[1, 2] = False

    ref = _load(write(str(tmp_path / "ref.mat"), U, V, W, x1d, y1d, valid=valid),
                force_isvalid="isValid")
    got = _load(write(str(tmp_path / "got.mat"), U, V, W, x1d, y1d,
                      desc_y=desc_y, desc_x=desc_x, valid=valid),
                force_isvalid="isValid")

    assert np.array_equal(ref["MASK"], got["MASK"])
    assert np.array_equal(got["MASK"], valid)
    assert np.array_equal(np.asarray(ref["U"]), np.asarray(got["U"]),
                          equal_nan=True)


def test_invalid_mask_convention_still_flips(tmp_path):
    """mask_convention='invalid' inverts per snapshot; the flip must survive it."""
    U, V, W, x1d, y1d = _field()

    valid = np.ones((NY, NX, NT), dtype=bool)
    valid[6, 8, :] = False
    valid[2, 3, :2] = False
    stored = ~valid              # the file stores "invalid", not "valid"

    ref_p = _write_classic(str(tmp_path / "ref.mat"), U, V, W, x1d, y1d,
                           valid=stored)
    got_p = _write_classic(str(tmp_path / "got.mat"), U, V, W, x1d, y1d,
                           desc_y=True, desc_x=True, valid=stored)

    ref = _load(ref_p, mask_convention="invalid")
    got = _load(got_p, mask_convention="invalid")

    assert np.array_equal(np.asarray(ref["U"]), np.asarray(got["U"]),
                          equal_nan=True)
    assert np.array_equal(np.isnan(got["U"]), ~valid)
    assert not bool(got["MASK"][6, 8])
    assert bool(got["MASK"][2, 3])


@pytest.mark.parametrize("desc_y,desc_x", ORDERS[1:])
def test_flip_does_not_copy_the_velocity_arrays(tmp_path, desc_y, desc_x):
    """Every flip must be a view: an HDF5 case can be many GB.

    If this ever fails because someone wrapped the flip in
    np.ascontiguousarray, the fix is not to relax the test: that doubles peak
    memory on exactly the files this loader exists to handle.
    """
    U, V, W, x1d, y1d = _field()
    path = _write_classic(str(tmp_path / "got.mat"), U, V, W, x1d, y1d,
                          desc_y=desc_y, desc_x=desc_x)
    ds = _load(path)

    assert ds["header"]["flip_rows"] is desc_y
    assert ds["header"]["flip_cols"] is desc_x
    for key in ("U", "V", "W", "x", "y"):
        arr = ds[key]
        # A reversed slice keeps its parent alive through .base and owns no
        # data of its own; a copy would have base None and own its buffer.
        assert arr.base is not None, key
        assert not arr.flags["OWNDATA"], key
