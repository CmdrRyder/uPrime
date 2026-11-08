"""In-place writes must reach the dataset when its arrays are strided views.

Since the loaders normalise grid order with reversed slices, a `.mat` whose
file stores y descending or x descending arrives with `x`, `y`, `U`, `V` and
`W` as negative-stride views rather than contiguous arrays. Several routines
write into those arrays in place:

    transform.apply_rotation   dataset["x"][:] = ...,  U[:, :, t] = ...
    transform.apply_shift      dataset["x"] -= ...
    transform.apply_mirror_x   dataset["x"] *= -1, dataset["U"] *= -1
    transform.apply_mirror_y   dataset["y"] *= -1, dataset["V"] *= -1

Strided assignment reaches the underlying buffer at the right positions, so
these are correct -- but the property is easy to break. Writing through
`reshape()` or `ravel()` would not: on a non-contiguous array both return a
*copy*, so the write would land on a temporary and be silently discarded. No
such site exists today; these tests exist so that one cannot be added quietly.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import contextlib
import io

import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from uprime.core import transform
from uprime.core.mat_loader import (auto_detect_mapping, detect_variables,
                                    load_mat_dataset)

NY, NX, NT = 9, 11, 6


def _field():
    y1d = np.linspace(0.0, 8.0, NY)
    x1d = np.linspace(0.0, 10.0, NX)
    Xg, Yg = np.meshgrid(x1d, y1d)
    rng = np.random.default_rng(3)
    U = 3.0 + 0.1 * Yg[:, :, None] + 0.2 * rng.standard_normal((NY, NX, NT))
    V = 0.5 + 0.1 * Xg[:, :, None] + 0.2 * rng.standard_normal((NY, NX, NT))
    W = 0.1 * rng.standard_normal((NY, NX, NT))
    return Xg, Yg, U, V, W


def _write(path, Xg, Yg, U, V, W, desc_y, desc_x):
    sl = (slice(None, None, -1) if desc_y else slice(None),
          slice(None, None, -1) if desc_x else slice(None))

    def R(a):
        return a[sl + (Ellipsis,)]

    scipy_io.savemat(path, {"x": R(Xg), "y": R(Yg),
                            "u": R(U), "v": R(V), "w": R(W)})
    return path


def _load(path):
    var_info, _is_v73 = detect_variables(path)
    mapping, ok = auto_detect_mapping(var_info)
    assert ok, mapping
    with contextlib.redirect_stdout(io.StringIO()):
        ds, _status = load_mat_dataset(path, mapping)
    return ds


def _pair(tmp_path):
    """The same field loaded from an ascending file and a both-descending one."""
    Xg, Yg, U, V, W = _field()
    ref = _load(_write(str(tmp_path / "ref.mat"), Xg, Yg, U, V, W, False, False))
    got = _load(_write(str(tmp_path / "got.mat"), Xg, Yg, U, V, W, True, True))
    return ref, got


def _assert_same(ref, got, keys=("x", "y", "U", "V", "W", "MASK")):
    for key in keys:
        a, b = ref.get(key), got.get(key)
        if a is None or b is None:
            assert a is None and b is None, key
            continue
        assert np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True), key


def test_the_descending_file_really_is_a_strided_view(tmp_path):
    """Guard the premise: if this stops being a view, the rest proves nothing."""
    _ref, got = _pair(tmp_path)

    assert got["header"]["flip_rows"] is True
    assert got["header"]["flip_cols"] is True
    for key in ("x", "y", "U", "V", "W"):
        arr = got[key]
        assert arr.base is not None, key
        assert arr.strides[0] < 0 and arr.strides[1] < 0, key


def test_shift_writes_through_the_view(tmp_path):
    ref, got = _pair(tmp_path)
    x_before = np.array(got["x"], copy=True)

    transform.apply_shift(ref, 1.5, -2.5)
    transform.apply_shift(got, 1.5, -2.5)

    # The write actually landed, and landed on the right elements.
    assert not np.allclose(got["x"], x_before)
    np.testing.assert_allclose(got["x"], x_before - 1.5, rtol=0, atol=1e-5)
    _assert_same(ref, got)


@pytest.mark.parametrize("fn", ["apply_mirror_x", "apply_mirror_y"])
def test_mirror_writes_through_the_view(tmp_path, fn):
    ref, got = _pair(tmp_path)
    mirror = getattr(transform, fn)

    mirror(ref)
    mirror(got)

    _assert_same(ref, got)
    # Mirroring must leave the convention intact on both.
    for ds in (ref, got):
        assert ds["x"][0, 0] < ds["x"][0, -1]
        assert ds["y"][0, 0] < ds["y"][-1, 0]
        # valid stays aliased to MASK, as the rest of the GUI assumes.
        assert ds["valid"] is ds["MASK"]


def test_rotation_writes_through_the_view(tmp_path):
    """apply_rotation writes x/y with `[:] =` and U/V snapshot by snapshot."""
    ref, got = _pair(tmp_path)
    x_before = np.array(got["x"], copy=True)
    u_before = np.array(got["U"], copy=True)

    transform.apply_rotation(ref, 5.0)
    transform.apply_rotation(got, 5.0)

    # The velocities were genuinely modified in place, not silently skipped.
    assert not np.allclose(np.nan_to_num(got["U"]), np.nan_to_num(u_before))
    # x and y are rewritten too, but with the same axis-aligned grid they
    # started from -- a rotation resamples the data onto the original grid
    # rather than moving the grid, so an unchanged x here is correct.
    np.testing.assert_allclose(got["x"], x_before, rtol=0, atol=1e-5)
    _assert_same(ref, got)


def test_transforms_compose_identically(tmp_path):
    """A realistic sequence: the two datasets must not drift apart."""
    ref, got = _pair(tmp_path)

    for ds in (ref, got):
        transform.apply_shift(ds, 0.5, 0.25)
        transform.apply_mirror_x(ds)
        transform.apply_rotation(ds, 3.0)
        transform.apply_mirror_y(ds)

    _assert_same(ref, got)
    assert (transform.transform_status_string(ref)
            == transform.transform_status_string(got))


def test_reshape_on_a_strided_view_would_not_write_through(tmp_path):
    """Document why reshape/ravel are banned for write-through on these arrays.

    This is a property of NumPy, not of uPrime, pinned here so the reasoning
    in the module docstring is checkable rather than asserted.
    """
    _ref, got = _pair(tmp_path)
    arr = got["x"]
    assert arr.base is not None

    flat = arr.reshape(-1)
    flat[0] = -12345.0
    assert got["x"][0, 0] != -12345.0        # the write went to a copy

    # Both forms the code actually uses reach the data.
    arr[0, 0] = -12345.0
    assert got["x"][0, 0] == -12345.0

    filler = np.full(arr.shape, 7.0)
    arr[:] = filler                          # as apply_rotation writes x and y
    assert np.array_equal(np.asarray(got["x"]), filler)
