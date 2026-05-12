"""
core/spectra.py
---------------
Thin dispatcher for SpectraWorker.  Each branch calls the appropriate
computation function and returns a plain dict consumed by the GUI result handler.
"""

import numpy as np


def compute_spectra(**kw):
    tab = kw['tab']

    # ------------------------------------------------------------------
    # Spatial  E(k) along a line
    # ------------------------------------------------------------------
    if tab == 'spatial':
        from core.spatial_spectra import spatial_psd_line
        k, psds = spatial_psd_line(
            kw['U'], kw['V'], kw['W'],
            kw['x'], kw['y'],
            kw['x0'], kw['y0'], kw['x1'], kw['y1'],
            kw['direction'], kw['avg_band'],
            kw['nperseg'], kw['noverlap'], kw['subtract'])
        return {'tab': 'spatial', 'k': k, 'psds': psds,
                'direction': kw['direction']}

    # ------------------------------------------------------------------
    # 3-D spatial  E(k) via FFT inside an ROI
    # ------------------------------------------------------------------
    elif tab == '3d_spatial':
        from core.spatial_spectra_fft import (compute_spectra_from_fluctuations,
                                               subtract_temporal_mean)
        U_roi = kw['U_roi']
        V_roi = kw['V_roi']
        W_roi = kw['W_roi']
        Lx    = kw['Lx']
        Ly    = kw['Ly']

        ny_roi, nx_roi, nt = U_roi.shape
        nan_mask = np.isnan(U_roi) | np.isnan(V_roi)
        if W_roi is not None:
            nan_mask |= np.isnan(W_roi)
        n_nan    = int(np.sum(nan_mask))
        total    = ny_roi * nx_roi * nt
        mask_pct = 100.0 * n_nan / total if total > 0 else 0.0

        print(f"[FFT Spectra] ROI: x=[{kw['roi']['x0']:.1f}, {kw['roi']['x1']:.1f}] mm  "
              f"y=[{kw['roi']['y0']:.1f}, {kw['roi']['y1']:.1f}] mm")
        print(f"[FFT Spectra] Grid: {ny_roi}×{nx_roi} × {nt} snapshots = {total} pts")
        n_valid = total - n_nan
        print(f"[FFT Spectra] Valid: {n_valid}/{total}  "
              f"({100-mask_pct:.1f}% valid, {mask_pct:.1f}% masked)")

        if n_valid < total * 0.5:
            raise ValueError(
                f"{mask_pct:.1f}% of ROI points are masked/NaN — too few valid "
                "vectors. Select a region with more valid data.")

        if n_nan > 0:
            U_roi = U_roi.copy()
            V_roi = V_roi.copy()
            U_roi[nan_mask] = 0.0
            V_roi[nan_mask] = 0.0
            if W_roi is not None:
                W_roi = W_roi.copy()
                W_roi[nan_mask] = 0.0

        # Upcast float16 ROI to float32 before pyfftw (float16 not supported).
        if U_roi.dtype == np.float16:
            U_roi = U_roi.astype(np.float32, copy=False)
            V_roi = V_roi.astype(np.float32, copy=False)
            if W_roi is not None:
                W_roi = W_roi.astype(np.float32, copy=False)

        if W_roi is None:
            W_roi = np.zeros_like(U_roi)

        U_4d = U_roi.reshape(1, ny_roi, nx_roi, nt)
        V_4d = V_roi.reshape(1, ny_roi, nx_roi, nt)
        W_4d = W_roi.reshape(1, ny_roi, nx_roi, nt)
        U_f, V_f, W_f = subtract_temporal_mean(U_4d, V_4d, W_4d)
        result = compute_spectra_from_fluctuations(U_f, V_f, W_f, Lx, Ly, 1.0)
        return {'tab': '3d_spatial', 'result': result,
                'roi': kw['roi'], 'mask_pct': mask_pct}

    # ------------------------------------------------------------------
    # Temporal  E(f) at a point or rectangle
    # ------------------------------------------------------------------
    elif tab == 'temporal':
        from core.spectral import psd_at_point, psd_in_region
        sel_type = kw['sel_type']
        if sel_type == 'temp_point':
            freq, psds = psd_at_point(
                kw['U'], kw['V'], kw['W'],
                kw['row'], kw['col'],
                kw['fs'], kw['nperseg'], kw['noverlap'])
            return {'tab': 'temporal', 'type': 'point',
                    'freq': freq, 'psds': psds, 'title': kw['title']}
        else:
            freq, psds, npts = psd_in_region(
                kw['U'], kw['V'], kw['W'],
                kw['x'], kw['y'],
                kw['x0'], kw['x1'], kw['y0'], kw['y1'],
                kw['fs'], kw['nperseg'], kw['noverlap'])
            return {'tab': 'temporal', 'type': 'rect',
                    'freq': freq, 'psds': psds,
                    'title': f"Rectangle avg ({npts} pts)"}

    # ------------------------------------------------------------------
    # Spatiotemporal  E(k, f)
    # ------------------------------------------------------------------
    elif tab == 'st':
        from core.spatiotemporal_spectra import compute_st_spectra
        k, f, psds = compute_st_spectra(
            kw['U'], kw['V'], kw['W'],
            kw['x'], kw['y'],
            kw['x0'], kw['y0'], kw['x1'], kw['y1'],
            kw['direction'], kw['avg'], kw['fs'])
        return {'tab': 'st', 'k': k, 'f': f, 'psds': psds,
                'direction': kw['direction']}

    raise ValueError(f"Unknown spectra tab: {tab!r}")
