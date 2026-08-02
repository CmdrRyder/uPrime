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


# ---------------------------------------------------------------------------
# Compare Cases reader (core/case_io.py)
# ---------------------------------------------------------------------------

_LINE_CSV = (
    "# uPrime Export\n"
    "# Generated : 2026-01-01 00:00:00\n"
    "# ==================================================\n"
    "# Analysis                 : Reynolds Stress - Line Profile\n"
    "# Snapshots                : 50\n"
    "# ==================================================\n"
    "dist_mm,x_mm,y_mm,mean_uu,std_uu\n"
    "0.0,0.0,2.0,1.0,0.10\n"
    "1.0,1.0,2.0,1.5,0.20\n"
    "2.0,2.0,2.0,2.0,0.30\n"
)

_TECPLOT_DAT = (
    "# uPrime Export\n"
    "# ==================================================\n"
    "# Analysis                 : Reynolds Stresses - All Components\n"
    "# ==================================================\n"
    'TITLE = "field.dat"\n'
    'VARIABLES = "x [mm]", "y [mm]", "<u\'u\'>"\n'
    'ZONE T="uPrime Export", I=2, J=2, F=POINT\n'
    "0.0 0.0 1.0\n0.0 1.0 2.0\n1.0 0.0 3.0\n1.0 1.0 4.0\n"
)


def _write_velocity_dat(path, nx, ny, scale=1.0, nan_at=None):
    """Write a synthetic Tecplot POINT velocity .dat (U, V, W), y top-to-bottom."""
    xs = np.linspace(0, nx - 1, nx)
    ys = np.linspace(0, ny - 1, ny)
    lines = [
        "# Analysis                 : Mean Velocity Field",
        'TITLE = "v"',
        'VARIABLES = "x [mm]", "y [mm]", "U [m/s]", "V [m/s]", "W [m/s]"',
        f'ZONE T="z", I={nx}, J={ny}, F=POINT',
    ]
    for j in range(ny - 1, -1, -1):          # top (max y) to bottom
        for i in range(nx):
            val = j * nx + i
            u = "nan" if (nan_at is not None and nan_at == (j, i)) \
                else f"{scale * val:.4e}"
            lines.append(f"{xs[i]:.4e} {ys[j]:.4e} {u} "
                         f"{-scale * val:.4e} {0.5 * scale:.4e}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class TestCaseIO:
    def test_parse_line_csv_header_and_series(self, tmp_path):
        from core.case_io import read_case_file
        p = tmp_path / "rey_line.csv"
        p.write_text(_LINE_CSV, encoding="utf-8")

        rec = read_case_file(str(p))
        assert rec.error is None
        assert rec.is_2d is False
        assert rec.source_module == "Reynolds Stress - Line Profile"
        assert rec.x_type == "arc_length"
        assert rec.x_label == "s"
        assert rec.x_units == "mm"
        assert rec.x_col == "dist_mm"

        # One CaseSeries per numeric non-x column; x_mm / y_mm excluded.
        # mean_uu -> canonical R_uu; std_uu falls back to its raw label.
        keys = {s.quantity_key for s in rec.series}
        assert keys == {"R_uu", "std_uu"}

        s = next(sr for sr in rec.series if sr.quantity_key == "R_uu")
        # Full CaseSeries field set
        for fld in ("source_file", "source_module", "quantity_key", "x_type",
                    "x_label", "x_units", "x_data", "y_label", "y_units",
                    "y_data", "um", "label", "style", "enabled"):
            assert hasattr(s, fld)
        assert s.source_file == "rey_line.csv"
        assert s.quantity_key == "R_uu"
        assert s.y_units == "m²/s²"
        assert s.label == "rey_line"          # default label = filename stem
        assert s.enabled is True
        assert s.um is None
        assert set(s.style) == {"color", "linestyle", "marker"}
        np.testing.assert_allclose(s.x_data, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(s.y_data, [1.0, 1.5, 2.0])

    def test_quantity_key_detection(self):
        from core.case_io import detect_quantity_key
        assert detect_quantity_key("mean_u") == "U"
        assert detect_quantity_key("mean_v") == "V"
        assert detect_quantity_key("mean_w") == "W"
        assert detect_quantity_key("mean_uu") == "R_uu"
        assert detect_quantity_key("mean_vw") == "R_vw"
        assert detect_quantity_key("mean_wv") == "R_vw"   # order-normalized
        assert detect_quantity_key("<u'u'>") == "R_uu"
        assert detect_quantity_key("mean_k") == "TKE"
        # fallbacks keep the raw label
        assert detect_quantity_key("mean_P") == "mean_P"
        assert detect_quantity_key("std_uu") == "std_uu"
        assert detect_quantity_key("PSD_u_m2s2_per_Hz") == "PSD_u_m2s2_per_Hz"

    def test_frequency_detection_and_um(self, tmp_path):
        from core.case_io import read_case_file
        content = (
            "# uPrime Export\n"
            "# ==================================================\n"
            "# Analysis                 : Temporal Spectral Analysis (Welch)\n"
            "# U_m                      : 1.500000 m/s\n"
            "# ==================================================\n"
            "frequency_Hz,PSD_u_m2s2_per_Hz\n"
            "1.0,10.0\n2.0,5.0\n3.0,2.5\n"
        )
        p = tmp_path / "spec.csv"
        p.write_text(content, encoding="utf-8")
        rec = read_case_file(str(p))
        assert rec.x_type == "frequency"
        assert rec.x_label == "f" and rec.x_units == "Hz"
        assert len(rec.series) == 1
        s = rec.series[0]
        assert s.quantity_key == "PSD_u_m2s2_per_Hz"
        assert s.x_type == "frequency"
        assert s.um == 1.5
        assert s.y_units == "m²/s²/Hz"

    def test_tecplot_dat_flagged_2d_with_metadata(self, tmp_path):
        from core.case_io import read_case_file
        p = tmp_path / "field.dat"
        p.write_text(_TECPLOT_DAT, encoding="utf-8")
        rec = read_case_file(str(p))
        assert rec.is_2d is True
        assert rec.series == []
        # 2D metadata is still parsed (module + quantity names)
        assert rec.source_module == "Reynolds Stresses - All Components"
        assert rec.twod_quantities == ["<u'u'>"]

    def test_unknown_x_then_manual_override(self, tmp_path):
        from core.case_io import read_case_file
        content = "col_a,col_b\n0.0,1.0\n1.0,2.0\n2.0,3.0\n"
        p = tmp_path / "custom.csv"
        p.write_text(content, encoding="utf-8")
        rec = read_case_file(str(p))
        assert rec.x_type == "unknown"
        assert rec.series == []

        series = rec.set_x_column("col_a")
        assert rec.x_type == "custom:col_a"
        assert [s.quantity_key for s in series] == ["col_b"]
        np.testing.assert_allclose(series[0].x_data, [0.0, 1.0, 2.0])

    def test_nan_safe_blank_cells(self, tmp_path):
        from core.case_io import read_case_file
        content = (
            "# Analysis                 : Mean Velocity - Line Profile\n"
            "dist_mm,x_mm,y_mm,mean_u\n"
            "0.0,0.0,0.0,1.0\n"
            "1.0,1.0,0.0,\n"          # blank y -> NaN, not filled
            "2.0,2.0,0.0,3.0\n"
        )
        p = tmp_path / "mean.csv"
        p.write_text(content, encoding="utf-8")
        rec = read_case_file(str(p))
        s = next(sr for sr in rec.series if sr.quantity_key == "U")
        assert np.isnan(s.y_data[1])
        assert np.isfinite(s.y_data[0]) and np.isfinite(s.y_data[2])

    def test_load_2d_velocity_field(self, tmp_path):
        from core.case_io import read_case_file
        nx, ny = 4, 3
        p = tmp_path / "vel.dat"
        _write_velocity_dat(p, nx, ny, scale=1.0, nan_at=(ny - 1, 0))

        rec = read_case_file(str(p))
        assert rec.is_2d is True
        assert rec.source_module == "Mean Velocity Field"
        assert rec.twod_components == ["U", "V", "W"]
        assert rec.field_loaded is False           # lazy

        rec.load_field()
        assert rec.field_loaded is True
        np.testing.assert_allclose(rec.x, [0, 1, 2, 3])
        np.testing.assert_allclose(rec.y, [0, 1, 2])   # ascending
        assert set(rec.components) == {"U", "V", "W"}
        assert rec.components["U"].shape == (ny, nx)
        assert rec.components["U"].dtype == np.float32
        # NaN preserved at the masked top-left point (not filled)
        assert np.isnan(rec.components["U"][ny - 1, 0])
        assert np.isnan(rec.components["U"]).sum() == 1
        assert np.isfinite(rec.components["V"]).all()
        assert "U" in rec.value_ranges

    def test_extract_all_components_routing(self, qapp, tmp_path):
        from gui.compare_window import CompareWindow
        pA = tmp_path / "caseA.dat"
        pB = tmp_path / "caseB.dat"
        _write_velocity_dat(pA, 8, 5, scale=1.0)
        _write_velocity_dat(pB, 8, 5, scale=2.0)

        w = CompareWindow()
        w._load_paths([str(pA), str(pB)])
        f2 = w.field2d
        f2.var2d.setCurrentIndex(f2.var2d.findData("U"))
        f2.rb_line.setChecked(True)
        f2._on_mode_changed()
        f2.line_case.setCurrentIndex(0)
        f2._draw_single_field()
        # manual horizontal line across the mid-row
        f2.rb_manual.setChecked(True)
        f2.sp_x0.setValue(0.0); f2.sp_y0.setValue(2.0)
        f2.sp_x1.setValue(7.0); f2.sp_y1.setValue(2.0)
        f2._on_manual_plot()
        assert f2._selection is not None
        f2.chk_apply_all.setChecked(True)
        f2._on_extract_plot()

        # One line -> U, V, W profiles from BOTH cases (apply-to-all)
        assert len(w._extracted) == 6
        assert sorted({s.quantity_key for s in w._extracted}) == ["U", "V", "W"]
        assert all(s.source_kind == "extracted" for s in w._extracted)
        assert all(s.x_type == "arc_length" for s in w._extracted)
        labels = {s.label for s in w._extracted}
        assert "caseA_U" in labels and "caseB_W" in labels
        # "Plot" auto-switches to the 1D tab and to an extracted component
        assert w.tabs.currentIndex() == 0
        assert w._current_variable() in ("U", "V", "W")
        # extracted series land under the matching 1D variable
        w.var_combo.setCurrentIndex(w.var_combo.findData("U"))
        assert len(w._current_series()) == 2

    def test_extract_save_writes_merged_csv(self, qapp, tmp_path, monkeypatch):
        from gui.compare_window import CompareWindow
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        pA = tmp_path / "caseA.dat"
        _write_velocity_dat(pA, 8, 5, scale=1.0)

        w = CompareWindow()
        w._load_paths([str(pA)])
        f2 = w.field2d
        f2.var2d.setCurrentIndex(f2.var2d.findData("U"))
        f2.rb_line.setChecked(True)
        f2._on_mode_changed()
        f2.line_case.setCurrentIndex(0)
        f2._draw_single_field()
        f2.rb_manual.setChecked(True)
        f2.sp_x0.setValue(0.0); f2.sp_y0.setValue(2.0)
        f2.sp_x1.setValue(7.0); f2.sp_y1.setValue(2.0)
        f2._on_manual_plot()

        out = tmp_path / "saved.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName",
                            staticmethod(lambda *a, **k: (str(out), "CSV")))
        monkeypatch.setattr(QMessageBox, "information",
                            staticmethod(lambda *a, **k: None))
        f2._on_extract_save()

        assert out.exists()
        assert len(w._extracted) == 0          # Save must NOT touch the basket
        lines = out.read_text(encoding="utf-8").splitlines()
        # same merged-CSV header/format as Export Merged CSV
        assert lines[0] == "case,quantity,x_label,x_value,y_label,y_value"
        assert any(ln.startswith("caseA_U,U,") for ln in lines[1:])


# ---------------------------------------------------------------------------
# Optional DaVis loader (core/davis_io.py) — skips cleanly without lvpyio
# ---------------------------------------------------------------------------

def _write_davis_vc7(path, ny, nx, scale=1.0, stereo=False, mask_corner=True):
    """Write a synthetic single-field DaVis vector buffer (.vc7) via lvpyio."""
    import lvpyio as lv
    dt = lv.vec3c if stereo else lv.vec2c
    a = np.ma.zeros((ny, nx), dtype=dt)
    a["u"] = scale
    a["v"] = -scale
    if stereo:
        a["w"] = 0.5 * scale
    a.mask = np.zeros((ny, nx), dtype=[(n, bool) for n in dt.names])
    if mask_corner:
        for n in dt.names:
            a[n].mask[0, 0] = True
    lv.write_buffer(a, str(path))


class TestDavisIO:
    def test_has_lvpyio_flag_importable(self):
        # Import must always succeed (guarded); the flag reflects availability.
        from core.davis_io import HAS_LVPYIO
        assert isinstance(HAS_LVPYIO, bool)

    def test_load_davis_2c_contract(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis
        ny, nx, nt = 4, 5, 3
        paths = []
        for k in range(nt):
            p = tmp_path / f"B{k:04d}.vc7"
            _write_davis_vc7(p, ny, nx, scale=float(k + 1))
            paths.append(str(p))
        ds = load_davis(paths)

        # identical contract to core.loader.load_dataset
        for key in ("x", "y", "U", "V", "W", "MASK", "MASK_LOADED", "valid",
                    "valid_frac", "mask_active", "is_stereo", "Nt", "nx", "ny",
                    "dx", "dy", "files", "header", "_memmap_path"):
            assert key in ds
        assert ds["x"].shape == (ny, nx)          # 2D coords
        assert ds["U"].shape == (ny, nx, nt)
        assert ds["U"].dtype == np.float32
        assert ds["W"] is None                    # 2D2C
        assert ds["is_stereo"] is False
        assert ds["Nt"] == nt and ds["nx"] == nx and ds["ny"] == ny
        assert ds["MASK"].dtype == bool
        # uPrime convention: True = valid; the single masked vector is invalid
        assert (~ds["MASK"]).sum() == 1
        # invalid vector propagated to NaN in every snapshot
        for i in range(nt):
            assert np.isnan(ds["U"][:, :, i]).sum() == 1
        # y ascending (row 0 = smallest y), orientation corrected from DaVis
        assert ds["y"][0, 0] <= ds["y"][-1, 0]
        assert ds["dx"] > 0 and ds["dy"] > 0

    def test_load_davis_3c_is_stereo(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis
        paths = []
        for k in range(2):
            p = tmp_path / f"B{k:04d}.vc7"
            _write_davis_vc7(p, 4, 5, scale=float(k + 1), stereo=True)
            paths.append(str(p))
        ds = load_davis(paths)
        assert ds["is_stereo"] is True
        assert ds["W"] is not None and ds["W"].shape == (4, 5, 2)

    def test_reject_set_selection(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis
        # .set is no longer supported — rejected by extension (no read needed).
        with pytest.raises(ValueError) as exc:
            load_davis(str(tmp_path / "recording.set"))
        assert ".vc7" in str(exc.value)           # message points to .vc7/.vec

    def test_reject_image_extension(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis
        with pytest.raises(ValueError):
            load_davis(str(tmp_path / "frame.im7"))

    def test_load_multi_vc7_stacks_and_orders(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis, davis_snapshot_count
        ny, nx = 4, 5
        # Deliberately provide the files out of order with names that would sort
        # WRONG lexically if not zero-padded-aware; each file's scalar = index.
        specs = [("B0010.vc7", 10.0), ("B0002.vc7", 2.0), ("B0001.vc7", 1.0)]
        paths = []
        for name, scale in specs:
            p = tmp_path / name
            _write_davis_vc7(p, ny, nx, scale=scale)
            paths.append(str(p))

        assert davis_snapshot_count(paths) == 3
        ds = load_davis(paths)
        assert ds["Nt"] == 3
        assert ds["U"].shape == (ny, nx, 3)
        assert ds["U"].dtype == np.float32
        assert ds["is_stereo"] is False
        # natural order B0001, B0002, B0010 -> U values 1, 2, 10
        vals = [float(ds["U"][1, 1, i]) for i in range(3)]
        assert vals == [1.0, 2.0, 10.0]
        assert [os.path.basename(f) for f in ds["files"]] == \
            ["B0001.vc7", "B0002.vc7", "B0010.vc7"]
        # per-snapshot mask preserved (one invalid vector each)
        for i in range(3):
            assert np.isnan(ds["U"][:, :, i]).sum() == 1

    def test_multi_vc7_subset(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis
        paths = []
        for k in range(4):
            p = tmp_path / f"B{k:04d}.vc7"
            _write_davis_vc7(p, 3, 3, scale=float(k))
            paths.append(str(p))
        ds = load_davis(paths, subset=[0, 2])
        assert ds["Nt"] == 2

    def test_multi_vc7_grid_mismatch_named(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis
        good1 = tmp_path / "B0001.vc7"; _write_davis_vc7(good1, 4, 5, scale=1.0)
        good2 = tmp_path / "B0002.vc7"; _write_davis_vc7(good2, 4, 5, scale=2.0)
        bad   = tmp_path / "B0003.vc7"; _write_davis_vc7(bad, 6, 5, scale=3.0)
        with pytest.raises(ValueError) as exc:
            load_davis([str(good1), str(good2), str(bad)])
        assert "B0003.vc7" in str(exc.value)      # names the offending file

    def test_reject_single_vc7(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis
        p = tmp_path / "B0001.vc7"
        _write_davis_vc7(p, 3, 3)
        with pytest.raises(ValueError):           # lone .vc7 = single snapshot
            load_davis([str(p)])

    def test_reject_set_mixed_with_vc7(self, tmp_path):
        pytest.importorskip("lvpyio")
        from core.davis_io import load_davis
        # Any selection containing a .set is rejected (no .set support at all).
        v = tmp_path / "B0001.vc7"
        _write_davis_vc7(v, 4, 5)
        with pytest.raises(ValueError):
            load_davis([str(tmp_path / "recording.set"), str(v)])
