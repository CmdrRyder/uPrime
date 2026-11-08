"""Row orientation in the .mat loader.

uPrime's convention is row 0 = smallest y. core.loader.load_grid and
core.davis_io both enforce it; mat_loader used to take the file's own row
order as given. A .mat written with y descending therefore loaded upside down
relative to a .dat or .vc7 case -- self-consistent, so nothing ever warned,
but not comparable.

Each test writes the same physical field twice, once with the rows ascending
in y and once descending, and asserts the two load to identical arrays.
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


def _rows(arr, descending):
    """Reverse the row axis when the file is to be written y-descending."""
    return arr[::-1, ...] if descending else arr


def _write_classic(path, U, V, W, x1d, y1d, descending, valid=None):
    """Write a classic (v5) .mat, rows in either y order."""
    y2d = np.tile(y1d.reshape(-1, 1), (1, NX))
    x2d = np.tile(x1d, (NY, 1))
    out = {
        "x": _rows(x2d, descending),
        "y": _rows(y2d, descending),
        "u": _rows(U, descending),
        "v": _rows(V, descending),
        "w": _rows(W, descending),
    }
    if valid is not None:
        out["isValid"] = _rows(valid, descending).astype(np.float64)
    scipy_io.savemat(path, out)
    return path


def _write_v73(path, U, V, W, x1d, y1d, descending, valid=None):
    """Write a MATLAB v7.3 file, so mat_loader takes its chunked h5py path.

    v7.3 is HDF5 with a 512-byte userblock. scipy reads the version from bytes
    124:128 and raises NotImplementedError for major == 2, which is exactly how
    mat_loader decides to use h5py. HDF5 stores the reverse of the MATLAB
    dimension order, so every array is transposed on the way in.
    """
    h5py = pytest.importorskip("h5py")

    y2d = np.tile(y1d.reshape(-1, 1), (1, NX))
    x2d = np.tile(x1d, (NY, 1))
    out = {
        "x": _rows(x2d, descending).T,                   # (Ny,Nx) -> (Nx,Ny)
        "y": _rows(y2d, descending).T,
        "u": _rows(U, descending).transpose(2, 1, 0),    # (Ny,Nx,Nt) -> (Nt,Nx,Ny)
        "v": _rows(V, descending).transpose(2, 1, 0),
        "w": _rows(W, descending).transpose(2, 1, 0),
    }
    if valid is not None:
        v = _rows(valid, descending)
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


def _assert_same(ds_asc, ds_desc, U, V, W, y1d):
    """Both row orders give identical arrays, in uPrime's convention."""
    for key in ("x", "y", "U", "V", "W", "MASK", "valid", "MASK_LOADED"):
        a, b = ds_asc[key], ds_desc[key]
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), key

    # And they match the field as written, with row 0 = smallest y.
    np.testing.assert_allclose(ds_desc["U"], U, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ds_desc["V"], V, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ds_desc["W"], W, rtol=0, atol=1e-5)
    np.testing.assert_allclose(ds_desc["y"][:, 0], y1d, rtol=0, atol=1e-9)
    assert ds_desc["y"][0, 0] < ds_desc["y"][-1, 0]
    assert ds_asc["y"][0, 0] < ds_asc["y"][-1, 0]


@pytest.mark.parametrize("writer", ["classic", "v73"])
def test_ascending_and_descending_y_load_identically(tmp_path, writer):
    U, V, W, x1d, y1d = _field()
    write = _write_classic if writer == "classic" else _write_v73

    asc = _load(write(str(tmp_path / "asc.mat"), U, V, W, x1d, y1d, False))
    desc = _load(write(str(tmp_path / "desc.mat"), U, V, W, x1d, y1d, True))

    _assert_same(asc, desc, U, V, W, y1d)
    assert asc["header"]["flip_rows"] is False
    assert desc["header"]["flip_rows"] is True


@pytest.mark.parametrize("writer", ["classic", "v73"])
def test_three_d_isvalid_flips_with_the_data(tmp_path, writer):
    """The per-snapshot isValid path must flip too, or NaN lands on wrong rows."""
    U, V, W, x1d, y1d = _field()
    write = _write_classic if writer == "classic" else _write_v73

    valid = np.ones((NY, NX, NT), dtype=bool)
    sometimes, always = (2, 3), (6, 8)
    valid[sometimes[0], sometimes[1], :2] = False
    valid[always[0], always[1], :] = False

    asc = _load(write(str(tmp_path / "asc.mat"), U, V, W, x1d, y1d, False, valid))
    desc = _load(write(str(tmp_path / "desc.mat"), U, V, W, x1d, y1d, True, valid))

    for key in ("U", "V", "W", "MASK", "y"):
        assert np.array_equal(np.asarray(asc[key]), np.asarray(desc[key]),
                              equal_nan=True), key

    # NaN sits on exactly the rows the field was written with.
    assert np.array_equal(np.isnan(desc["U"]), ~valid)
    assert bool(desc["MASK"][sometimes])
    assert not bool(desc["MASK"][always])


@pytest.mark.parametrize("writer", ["classic", "v73"])
def test_static_two_d_isvalid_flips_with_the_data(tmp_path, writer):
    """A 2-D isValid bypasses _apply_frame, so it needs its own flip."""
    U, V, W, x1d, y1d = _field()
    write = _write_classic if writer == "classic" else _write_v73

    valid = np.ones((NY, NX), dtype=bool)
    valid[6, 8] = False
    valid[1, 2] = False

    asc = _load(write(str(tmp_path / "asc.mat"), U, V, W, x1d, y1d, False, valid),
                force_isvalid="isValid")
    desc = _load(write(str(tmp_path / "desc.mat"), U, V, W, x1d, y1d, True, valid),
                 force_isvalid="isValid")

    assert np.array_equal(asc["MASK"], desc["MASK"])
    assert np.array_equal(desc["MASK"], valid)
    assert np.array_equal(np.asarray(asc["U"]), np.asarray(desc["U"]),
                          equal_nan=True)


def test_invalid_mask_convention_still_flips(tmp_path):
    """mask_convention='invalid' inverts per snapshot; the flip must survive it."""
    U, V, W, x1d, y1d = _field()

    valid = np.ones((NY, NX, NT), dtype=bool)
    valid[6, 8, :] = False
    valid[2, 3, :2] = False
    stored = ~valid              # the file stores "invalid", not "valid"

    asc_p = _write_classic(str(tmp_path / "asc.mat"), U, V, W, x1d, y1d,
                           False, stored)
    desc_p = _write_classic(str(tmp_path / "desc.mat"), U, V, W, x1d, y1d,
                            True, stored)

    asc = _load(asc_p, mask_convention="invalid")
    desc = _load(desc_p, mask_convention="invalid")

    assert np.array_equal(np.asarray(asc["U"]), np.asarray(desc["U"]),
                          equal_nan=True)
    assert np.array_equal(np.isnan(desc["U"]), ~valid)
    assert not bool(desc["MASK"][6, 8])
    assert bool(desc["MASK"][2, 3])


def test_flip_does_not_copy_the_velocity_arrays(tmp_path):
    """The flip must be a view: an HDF5 case can be many GB."""
    U, V, W, x1d, y1d = _field()
    path = _write_classic(str(tmp_path / "desc.mat"), U, V, W, x1d, y1d, True)
    ds = _load(path)

    assert ds["header"]["flip_rows"] is True
    for key in ("U", "V", "W", "x", "y"):
        arr = ds[key]
        # A reversed slice keeps its parent alive through .base and owns no
        # data of its own; a copy would have base None and own its buffer.
        assert arr.base is not None, key
        assert not arr.flags["OWNDATA"], key
