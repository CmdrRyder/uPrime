import os
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
import sys


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def dataset():
    ny, nx, Nt = 15, 20, 50
    np.random.seed(42)
    x = np.tile(np.linspace(0, 10, nx), (ny, 1)).astype(np.float32)
    y = np.tile(np.linspace(0, 7.5, ny), (nx, 1)).T.astype(np.float32)
    U_mean = 2.0 * (1 - ((y - 3.75) / 3.75) ** 2)
    U = (U_mean[:, :, np.newaxis] +
         0.1 * np.random.randn(ny, nx, Nt)).astype(np.float32)
    V = (0.05 * np.random.randn(ny, nx, Nt)).astype(np.float32)
    W = None
    mask = np.ones((ny, nx), dtype=bool)
    mask[0, :] = False
    mask[-1, :] = False
    return {
        "U": U, "V": V, "W": W,
        "x": x, "y": y,
        "MASK": mask,
        "MASK_LOADED": mask.copy(),
        "mask_active": True,
        "_memmap_path": None,
        "files": [f"B{i:05d}.dat" for i in range(Nt)],
        "is_stereo": False,
        "Nt": Nt, "nx": nx, "ny": ny,
    }


@pytest.fixture
def dataset_stereo(dataset):
    """Stereo version with W component."""
    ds = dict(dataset)
    ny, nx, Nt = ds["U"].shape
    ds["W"] = (0.02 * np.random.randn(ny, nx, Nt)).astype(np.float32)
    ds["is_stereo"] = True
    return ds


# ---------------------------------------------------------------------------
# Core module tests
# ---------------------------------------------------------------------------

class TestDatasetUtils:
    def test_get_masked_applies_mask(self, dataset):
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        assert np.all(np.isnan(U[0, :, :]))
        assert np.all(np.isnan(U[-1, :, :]))
        assert not np.any(np.isnan(U[1:-1, :, :]))

    def test_get_masked_inactive(self, dataset):
        from core.dataset_utils import get_masked
        dataset["mask_active"] = False
        U = get_masked(dataset, "U")
        assert not np.any(np.isnan(U))
        dataset["mask_active"] = True

    def test_get_masked_none(self, dataset):
        from core.dataset_utils import get_masked
        W = get_masked(dataset, "W")
        assert W is None


class TestTKEBudget:
    def test_budget_runs(self, dataset):
        from core.tke_budget import compute_tke_budget
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        result = compute_tke_budget(U, V, None,
                                    dataset["x"], dataset["y"],
                                    mask=dataset["MASK"])
        assert "k" in result
        assert "P" in result
        assert "C" in result
        assert "D" in result
        assert "R" in result

    def test_k_positive(self, dataset):
        from core.tke_budget import compute_tke_budget
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        result = compute_tke_budget(U, V, None,
                                    dataset["x"], dataset["y"],
                                    mask=dataset["MASK"])
        k = result["k"]
        assert np.all(k[np.isfinite(k)] >= 0), "TKE must be non-negative"

    def test_k_units_order_of_magnitude(self, dataset):
        """k should be O(u'^2) ~ 0.01 m^2/s^2 for 10% turbulence intensity."""
        from core.tke_budget import compute_tke_budget
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        result = compute_tke_budget(U, V, None,
                                    dataset["x"], dataset["y"],
                                    mask=dataset["MASK"])
        k_max = np.nanmax(result["k"])
        assert 1e-6 < k_max < 100, f"k_max={k_max} is outside expected range"


class TestCorrelation:
    def test_spatial_correlation_point_returns_one_at_zero(self, dataset):
        from core.two_point_corr import compute_spatial_correlation_point
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        # Returns (R_norm, R_x, R_y)
        R_norm, R_x, R_y = compute_spatial_correlation_point(
            U, V, None, 7, 10, component='uu')
        assert abs(R_norm[7, 10] - 1.0) < 0.05

    def test_spatial_correlation_bounded(self, dataset):
        from core.two_point_corr import compute_spatial_correlation_point
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        R_norm, R_x, R_y = compute_spatial_correlation_point(
            U, V, None, 7, 10, component='uu')
        finite = R_norm[np.isfinite(R_norm)]
        assert np.all(np.abs(finite) <= 1.05), "Correlation must be in [-1, 1]"


class TestVortexID:
    def test_gradients_shape(self, dataset):
        from core.vortex_id import compute_gradients
        from core.dataset_utils import get_masked
        U_mean = np.nanmean(get_masked(dataset, "U"), axis=2)
        V_mean = np.nanmean(get_masked(dataset, "V"), axis=2)
        grads = compute_gradients(U_mean, V_mean,
                                  dataset["x"], dataset["y"],
                                  dataset["MASK"])
        ny, nx = dataset["ny"], dataset["nx"]
        for key in ("dudx", "dudy", "dvdx", "dvdy"):
            assert grads[key].shape == (ny, nx), f"{key} wrong shape"

    def test_vortex_fields_keys(self, dataset):
        from core.vortex_id import compute_gradients, compute_vortex_fields
        from core.dataset_utils import get_masked
        U_mean = np.nanmean(get_masked(dataset, "U"), axis=2)
        V_mean = np.nanmean(get_masked(dataset, "V"), axis=2)
        grads = compute_gradients(U_mean, V_mean,
                                  dataset["x"], dataset["y"],
                                  dataset["MASK"])
        fields = compute_vortex_fields(grads)
        for key in ("omega", "Q", "lambda_ci", "lambda2"):
            assert key in fields, f"Missing field: {key}"

    def test_lambda_ci_not_all_nan(self, dataset):
        from core.vortex_id import compute_gradients, compute_vortex_fields
        from core.dataset_utils import get_masked
        U_mean = np.nanmean(get_masked(dataset, "U"), axis=2)
        V_mean = np.nanmean(get_masked(dataset, "V"), axis=2)
        grads = compute_gradients(U_mean, V_mean,
                                  dataset["x"], dataset["y"],
                                  dataset["MASK"])
        fields = compute_vortex_fields(grads)
        lci = fields["lambda_ci"]
        finite = lci[np.isfinite(lci)]
        assert finite.size > 0, "lambda_ci is all NaN"

    def test_detect_vortices_returns_list(self, dataset):
        from core.vortex_id import (compute_gradients, compute_vortex_fields,
                                    detect_vortices)
        from core.dataset_utils import get_masked
        U_mean = np.nanmean(get_masked(dataset, "U"), axis=2)
        V_mean = np.nanmean(get_masked(dataset, "V"), axis=2)
        grads = compute_gradients(U_mean, V_mean,
                                  dataset["x"], dataset["y"],
                                  dataset["MASK"])
        fields = compute_vortex_fields(grads)
        vortices = detect_vortices(fields["Q"], fields["omega"],
                                   dataset["x"], dataset["y"],
                                   threshold=0.0,
                                   sign_filter="all",
                                   min_area_mm2=0.01)
        assert isinstance(vortices, list)
        if len(vortices) > 0:
            v = vortices[0]
            for key in ("id", "x_center", "y_center", "area_mm2",
                        "sign", "circulation", "aspect_ratio"):
                assert key in v, f"Missing key in vortex dict: {key}"


class TestPOD:
    # compute_pod(U, V, W, n_modes) -- no mask param
    # returns: energy_frac, modes shape (n_modes, ny, nx, Nc), eigenvalues

    def test_pod_runs_and_energy_sums_to_one(self, dataset):
        from core.pod import compute_pod
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        result = compute_pod(U, V, None, n_modes=10)
        assert "energy_frac" in result
        assert "modes" in result
        assert "eigenvalues" in result
        total = result["energy_frac"].sum()
        assert abs(total - 1.0) < 0.01, f"energy_frac sums to {total}, expected 1.0"

    def test_pod_mode_shapes(self, dataset):
        from core.pod import compute_pod
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        result = compute_pod(U, V, None, n_modes=5)
        ny, nx = dataset["ny"], dataset["nx"]
        # modes shape: (n_modes, ny, nx, Nc)
        assert result["modes"].shape[0] == 5
        assert result["modes"].shape[1:3] == (ny, nx)

    def test_pod_energy_descending(self, dataset):
        from core.pod import compute_pod
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        result = compute_pod(U, V, None, n_modes=10)
        ef = result["energy_frac"]
        assert np.all(np.diff(ef) <= 1e-10), "POD energies must be descending"


class TestDMD:
    def test_dmd_runs(self, dataset):
        from core.dmd import build_snapshot_matrix, compute_dmd, scale_to_physical
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U").astype(float)
        V = get_masked(dataset, "V").astype(float)
        U -= np.nanmean(U, axis=2, keepdims=True)
        V -= np.nanmean(V, axis=2, keepdims=True)
        X, n_per = build_snapshot_matrix(U, V, None,
                                         component="stacked",
                                         mask=dataset["MASK"])
        X = np.nan_to_num(X, nan=0.0)
        result = compute_dmd(X, rank=10)
        result = scale_to_physical(result, fs=1000.0)
        assert "modes" in result
        assert "frequencies_hz" in result
        assert "growth_rates_phys" in result
        assert "amplitudes" in result
        assert result["modes"].shape[1] == result["rank"]

    def test_dmd_frequencies_finite(self, dataset):
        from core.dmd import build_snapshot_matrix, compute_dmd, scale_to_physical
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U").astype(float)
        V = get_masked(dataset, "V").astype(float)
        U -= np.nanmean(U, axis=2, keepdims=True)
        V -= np.nanmean(V, axis=2, keepdims=True)
        X, _ = build_snapshot_matrix(U, V, None,
                                     component="stacked",
                                     mask=dataset["MASK"])
        X = np.nan_to_num(X, nan=0.0)
        result = scale_to_physical(compute_dmd(X, rank=10), fs=1000.0)
        assert np.all(np.isfinite(result["frequencies_hz"]))
        assert np.all(np.isfinite(result["growth_rates_phys"]))


class TestLoader:
    def test_estimate_dataset_size(self):
        from core.loader import estimate_dataset_size
        header = {"ny": 50, "nx": 40, "is_stereo": False}
        size = estimate_dataset_size(["f"] * 100, header, stride=1)
        expected = 50 * 40 * 100 * 2 * 4
        assert size == expected

    def test_estimate_dataset_size_stride(self):
        from core.loader import estimate_dataset_size
        header = {"ny": 50, "nx": 40, "is_stereo": False}
        size_stride1 = estimate_dataset_size(["f"] * 100, header, stride=1)
        size_stride2 = estimate_dataset_size(["f"] * 100, header, stride=2)
        assert size_stride2 == size_stride1 // 2


# ---------------------------------------------------------------------------
# GUI smoke tests
# ---------------------------------------------------------------------------

class TestGUISmoke:
    def test_main_window_opens(self, qapp):
        from gui.main_window import MainWindow
        win = MainWindow()
        win.show()
        qapp.processEvents()
        assert win.isVisible()
        win.close()

    def test_reynolds_window_opens(self, qapp, dataset):
        from gui.reynolds_window import ReynoldsWindow
        win = ReynoldsWindow(dataset)
        win.show()
        qapp.processEvents()
        assert win.isVisible()
        win.close()

    def test_tke_window_opens(self, qapp, dataset):
        from gui.tke_budget_window import TKEBudgetWindow
        win = TKEBudgetWindow(dataset)
        win.show()
        qapp.processEvents()
        assert win.isVisible()
        win.close()

    def test_correlation_window_opens(self, qapp, dataset):
        from gui.correlation_window import CorrelationWindow
        win = CorrelationWindow(dataset)
        win.show()
        qapp.processEvents()
        assert win.isVisible()
        win.close()

    def test_pod_window_opens(self, qapp, dataset):
        from gui.pod_window import PODWindow
        win = PODWindow(dataset)
        win.show()
        qapp.processEvents()
        assert win.isVisible()
        win.close()

    def test_vortex_window_opens(self, qapp, dataset):
        from gui.vortex_window import VortexWindow
        win = VortexWindow(dataset)
        win.show()
        qapp.processEvents()
        assert win.isVisible()
        win.close()

    def test_dmd_window_does_not_crash(self, qapp, dataset):
        """DMD window should open or gracefully reject the dataset -- no crash."""
        from gui.dmd_window import DmdWindow
        try:
            win = DmdWindow(dataset, fs=1000.0)
            qapp.processEvents()
            win.close()
        except SystemExit:
            pass

    def test_spatial_spectra_window_opens(self, qapp, dataset):
        from gui.spatial_spectra_window import SpatialSpectraWindow
        win = SpatialSpectraWindow(dataset, is_time_resolved=False, fs=1000.0)
        win.show()
        qapp.processEvents()
        assert win.isVisible()
        win.close()


# ---------------------------------------------------------------------------
# Numerical regression tests
# ---------------------------------------------------------------------------

class TestNumericalRegression:
    def test_tke_equals_half_variance_sum(self, dataset):
        """k = 0.5*(uu + vv) for 2D2C data."""
        from core.tke_budget import compute_tke_budget
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        result = compute_tke_budget(U, V, None,
                                    dataset["x"], dataset["y"],
                                    mask=dataset["MASK"])
        mean_U = np.nanmean(U, axis=2)
        mean_V = np.nanmean(V, axis=2)
        up = U - mean_U[:, :, np.newaxis]
        vp = V - mean_V[:, :, np.newaxis]
        uu = np.nanmean(up ** 2, axis=2)
        vv = np.nanmean(vp ** 2, axis=2)
        k_expected = 0.5 * (uu + vv)
        k_actual = result["k"]
        valid = np.isfinite(k_expected) & np.isfinite(k_actual)
        np.testing.assert_allclose(k_actual[valid], k_expected[valid],
                                   rtol=1e-4,
                                   err_msg="TKE does not equal 0.5*(uu+vv)")

    def test_pod_reconstruction_improves_with_modes(self, dataset):
        """Adding more POD modes should capture more energy."""
        from core.pod import compute_pod
        from core.dataset_utils import get_masked
        U = get_masked(dataset, "U")
        V = get_masked(dataset, "V")
        result = compute_pod(U, V, None, n_modes=20)
        ef = result["energy_frac"]
        err_1mode = 1.0 - ef[:1].sum()
        err_10mode = 1.0 - ef[:10].sum()
        assert err_10mode < err_1mode, "More modes should capture more energy"

    def test_spatial_gradient_units(self, dataset):
        """dU/dx should be O(100) 1/s for U~2 m/s over x~10 mm."""
        from core.vortex_id import compute_gradients
        from core.dataset_utils import get_masked
        U_mean = np.nanmean(get_masked(dataset, "U"), axis=2)
        V_mean = np.nanmean(get_masked(dataset, "V"), axis=2)
        grads = compute_gradients(U_mean, V_mean,
                                  dataset["x"], dataset["y"],
                                  dataset["MASK"])
        max_grad = np.nanmax(np.abs(grads["dudx"]))
        assert max_grad < 10000, (
            f"dU/dx max = {max_grad:.1f} 1/s -- suspiciously large, "
            "check mm-to-m conversion")
        assert max_grad > 0.1, (
            f"dU/dx max = {max_grad:.1f} 1/s -- suspiciously small")
