"""
core/workers.py
---------------
Reusable QThread worker base class and one worker per heavy computation.
"""

from PyQt6.QtCore import QThread, pyqtSignal
import numpy as np
import traceback


class BaseWorker(QThread):
    """Base class for all uPrime background workers."""
    finished = pyqtSignal(object)   # emits result dict on success
    error    = pyqtSignal(str)      # emits traceback string on failure
    progress = pyqtSignal(int)      # emits 0-100

    def run(self):
        try:
            result = self.compute()
            self.finished.emit(result)
        except Exception:
            self.error.emit(traceback.format_exc())

    def compute(self):
        raise NotImplementedError


class PODWorker(BaseWorker):
    def __init__(self, U, V, W, n_modes):
        super().__init__()
        self.U, self.V, self.W = U, V, W
        self.n_modes = n_modes

    def compute(self):
        from core.pod import compute_pod
        return compute_pod(self.U, self.V, self.W, n_modes=self.n_modes)


class DMDWorker(BaseWorker):
    def __init__(self, U, V, W, mask, component, n_modes, subtract_mean, fs):
        super().__init__()
        self.U, self.V, self.W = U, V, W
        self.mask = mask
        self.component = component
        self.n_modes = n_modes
        self.subtract_mean = subtract_mean
        self.fs = fs

    def compute(self):
        from core.dmd import build_snapshot_matrix, compute_dmd, scale_to_physical
        U = self.U.copy()
        V = self.V.copy()
        W = self.W.copy() if self.W is not None else None
        if self.subtract_mean:
            U -= np.nanmean(U, axis=2, keepdims=True)
            V -= np.nanmean(V, axis=2, keepdims=True)
            if W is not None:
                W -= np.nanmean(W, axis=2, keepdims=True)
        X, n_per = build_snapshot_matrix(U, V, W,
                                          component=self.component,
                                          mask=self.mask)
        X = np.nan_to_num(X, nan=0.0)
        rank = min(X.shape[0], X.shape[1] - 1, self.n_modes)
        result = compute_dmd(X, rank=rank)
        result = scale_to_physical(result, self.fs)
        result['n_per'] = n_per
        result['n_components'] = (
            (3 if W is not None else 2) if self.component == 'stacked' else 1
        )
        return result


class TKEBudgetWorker(BaseWorker):
    def __init__(self, U, V, W, x, y, mask, smooth_kernel, compute_dkdt):
        super().__init__()
        self.U, self.V, self.W = U, V, W
        self.x, self.y = x, y
        self.mask = mask
        self.smooth_kernel = smooth_kernel
        self.compute_dkdt = compute_dkdt

    def compute(self):
        from core.tke_budget import compute_tke_budget
        return compute_tke_budget(self.U, self.V, self.W,
                                   self.x, self.y,
                                   mask=self.mask,
                                   smooth_kernel=self.smooth_kernel,
                                   compute_dkdt=self.compute_dkdt)


class CorrelationWorker(BaseWorker):
    def __init__(self, U, V, W, x, y, mode, ref_row, ref_col,
                 component, use_kernel, max_lag_frac, dt,
                 roi_coords=None):
        super().__init__()
        self.U, self.V, self.W = U, V, W
        self.x, self.y = x, y
        self.mode = mode
        self.ref_row = ref_row
        self.ref_col = ref_col
        self.component = component
        self.use_kernel = use_kernel
        self.max_lag_frac = max_lag_frac
        self.dt = dt
        self.roi_coords = roi_coords

    def compute(self):
        from core.two_point_corr import (
            compute_spatial_correlation_point,
            compute_spatial_correlation_roi,
            compute_temporal_correlation,
        )
        if self.mode == 'spatial_point':
            R_norm, R_x, R_y = compute_spatial_correlation_point(
                self.U, self.V, self.W,
                self.ref_row, self.ref_col,
                component=self.component,
                use_kernel=self.use_kernel)
            return {'mode': 'spatial_point',
                    'R_norm': R_norm, 'R_x': R_x, 'R_y': R_y}

        elif self.mode == 'spatial_roi':
            x0, x1, y0, y1 = self.roi_coords
            dx_arr, R_x, dy_arr, R_y, _Lx, _Ly = compute_spatial_correlation_roi(
                self.U, self.V, self.W,
                self.x, self.y,
                x0, x1, y0, y1,
                component=self.component)
            return {'mode': 'spatial_roi',
                    'dx_arr': dx_arr, 'R_x': R_x,
                    'dy_arr': dy_arr, 'R_y': R_y}

        else:  # temporal
            roi_kw  = {}
            ref_row = self.ref_row
            ref_col = self.ref_col
            use_k   = self.use_kernel
            if self.roi_coords is not None:
                roi_kw  = dict(roi_coords=self.roi_coords,
                               x=self.x, y=self.y)
                ref_row = ref_col = 0
                use_k   = False
            R_tau, lags = compute_temporal_correlation(
                self.U, self.V, self.W,
                ref_row, ref_col,
                component=self.component,
                use_kernel=use_k,
                max_lag_fraction=self.max_lag_frac,
                **roi_kw)
            return {'mode': 'temporal', 'R_tau': R_tau, 'lags': lags}


class SpectraWorker(BaseWorker):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def compute(self):
        from core.spectra import compute_spectra
        return compute_spectra(**self.kwargs)


class VortexWorker(BaseWorker):
    """U, V are 2D arrays (pre-averaged or single frame)."""
    def __init__(self, U, V, x, y, mask, field_key, S):
        super().__init__()
        self.U, self.V = U, V
        self.x, self.y = x, y
        self.mask = mask
        self.field_key = field_key
        self.S = S

    def compute(self):
        from core.vortex_id import (compute_gradients, compute_vortex_fields,
                                     compute_gamma)
        U, V = self.U, self.V
        if self.field_key == 'gamma1':
            g1, g2 = compute_gamma(U, V, self.x, self.y, S=self.S)
            grads  = compute_gradients(U, V, self.x, self.y, self.mask)
            omega  = compute_vortex_fields(grads)['omega']
            return dict(field=g1, gamma2=g2, omega=omega)
        else:
            grads  = compute_gradients(U, V, self.x, self.y, self.mask)
            fields = compute_vortex_fields(grads)
            return dict(field=fields[self.field_key],
                        omega=fields['omega'], gamma2=None)


class ReynoldsWorker(BaseWorker):
    def __init__(self, U, V, W):
        super().__init__()
        self.U, self.V, self.W = U, V, W

    def compute(self):
        from core.reynolds_stress import (compute_reynolds_stresses,
                                           compute_reynolds_stress_std)
        stresses, k = compute_reynolds_stresses(self.U, self.V, self.W)
        std = compute_reynolds_stress_std(self.U, self.V, self.W)
        return dict(stresses=stresses, k=k, std=std)
