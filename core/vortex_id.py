"""
core/vortex_id.py
-----------------
Vortex identification algorithms for uPrime.

Provides:
  compute_gradients      -- 2D velocity gradient tensor
  compute_vortex_fields  -- omega, Q, lambda_ci, lambda2
  compute_gamma          -- Gamma1, Gamma2 (Graftieaux et al. 2001)
  detect_vortices        -- threshold + blob analysis
  compute_spatial_probability -- detection probability map
  export_vortex_csv      -- CSV export of vortex table
"""

import csv
import numpy as np
from scipy import ndimage

_FONT_AX   = 9
_FONT_TICK = 8


# ---------------------------------------------------------------------------
# 1. Gradients
# ---------------------------------------------------------------------------

def compute_gradients(U, V, x, y, mask):
    """
    Compute 2D velocity gradients from U, V fields.

    Parameters
    ----------
    U, V : [ny, nx] -- velocity in m/s, NaN at invalid pixels (already masked)
    x, y : [ny, nx] -- coordinates in mm
    mask : [ny, nx] -- bool, True = valid. Used to NaN gradient boundaries.

    Returns
    -------
    dict with keys: dudx, dudy, dvdx, dvdy -- all [ny, nx] in 1/s
    """
    dx_m = float(np.abs(x[0, 1] - x[0, 0])) / 1000.0
    dy_m = float(np.abs(y[1, 0] - y[0, 0])) / 1000.0

    dudy, dudx = np.gradient(U, dy_m, dx_m)
    dvdy, dvdx = np.gradient(V, dy_m, dx_m)

    inv = ~mask
    for arr in (dudx, dudy, dvdx, dvdy):
        arr[inv] = np.nan

    for arr in (dudx, dudy, dvdx, dvdy):
        arr[0,  :] = np.nan
        arr[-1, :] = np.nan
        arr[:,  0] = np.nan
        arr[:, -1] = np.nan

    return dict(dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy)


# ---------------------------------------------------------------------------
# 2. Gradient-based scalar fields
# ---------------------------------------------------------------------------

def compute_vortex_fields(grads):
    """
    Compute omega, Q, lambda_ci, lambda2 from gradient dict.

    Returns dict with keys: 'omega', 'Q', 'lambda_ci', 'lambda2'
    """
    dudx = grads['dudx']
    dudy = grads['dudy']
    dvdx = grads['dvdx']
    dvdy = grads['dvdy']

    omega = dvdx - dudy

    Q = -0.5 * (dudx**2 + dvdy**2 + 2.0 * dudy * dvdx)

    discriminant = (dudx - dvdy)**2 + 4.0 * dudy * dvdx
    lambda_ci = np.where(discriminant < 0,
                         np.sqrt(np.maximum(-discriminant, 0)) / 2.0,
                         0.0)
    lambda_ci = lambda_ci * np.sign(omega)

    s11 = dudx
    s12 = 0.5 * (dudy + dvdx)
    s22 = dvdy
    o12 = 0.5 * (dvdx - dudy)

    m11 = s11**2 + s12**2 - o12**2
    m12 = s11 * s12 + s12 * s22
    m22 = s12**2 + s22**2 - o12**2

    trace_half = 0.5 * (m11 + m22)
    disc       = np.sqrt(np.maximum((m11 - m22)**2 + 4.0 * m12**2, 0))
    lambda2    = trace_half - 0.5 * disc

    return dict(omega=omega, Q=Q, lambda_ci=lambda_ci, lambda2=lambda2)


# ---------------------------------------------------------------------------
# 3. Gamma1 and Gamma2 (Graftieaux et al. 2001)
# ---------------------------------------------------------------------------

def compute_gamma(U, V, x, y, S=2):
    """
    Compute Gamma1 and Gamma2 following Graftieaux et al. (2001).
    MST 12(9), 1422. doi:10.1088/0957-0233/12/9/309

    Parameters
    ----------
    U, V : [ny, nx] -- velocity in m/s, NaN at invalid pixels
    x, y : [ny, nx] -- coordinates in mm
    S    : int -- neighborhood half-width in grid points (default 2 -> 5x5 kernel)

    Returns
    -------
    gamma1 : [ny, nx] -- vortex core indicator, range [-1, +1]
    gamma2 : [ny, nx] -- vortex boundary indicator, range [-1, +1]
    """
    ny, nx = U.shape
    gamma1 = np.full((ny, nx), np.nan)
    gamma2 = np.full((ny, nx), np.nan)

    for j in range(S, ny - S):
        for i in range(S, nx - S):
            U_nb = U[j-S:j+S+1, i-S:i+S+1].ravel()
            V_nb = V[j-S:j+S+1, i-S:i+S+1].ravel()
            x_nb = x[j-S:j+S+1, i-S:i+S+1].ravel()
            y_nb = y[j-S:j+S+1, i-S:i+S+1].ravel()

            valid = np.isfinite(U_nb) & np.isfinite(V_nb)
            if valid.sum() < 0.75 * len(U_nb):
                continue

            cx = x[j, i]
            cy = y[j, i]
            PM_x = x_nb - cx
            PM_y = y_nb - cy

            r = np.sqrt(PM_x**2 + PM_y**2)
            nonzero = (r > 1e-12) & valid

            if nonzero.sum() < 4:
                continue

            PM_x = PM_x[nonzero]
            PM_y = PM_y[nonzero]
            r    = r[nonzero]
            U_m  = U_nb[nonzero]
            V_m  = V_nb[nonzero]

            # Gamma1
            U_mag = np.sqrt(U_m**2 + V_m**2)
            denom1 = r * U_mag
            valid1 = denom1 > 1e-12
            if valid1.sum() < 4:
                continue
            cross1 = PM_x[valid1] * V_m[valid1] - PM_y[valid1] * U_m[valid1]
            gamma1[j, i] = np.mean(cross1 / denom1[valid1])

            # Gamma2
            U_mean_nb = np.nanmean(U_nb[valid])
            V_mean_nb = np.nanmean(V_nb[valid])
            dU_m = U_m - U_mean_nb
            dV_m = V_m - V_mean_nb
            dU_mag = np.sqrt(dU_m**2 + dV_m**2)
            denom2 = r * dU_mag
            valid2 = denom2 > 1e-12
            if valid2.sum() < 4:
                continue
            cross2 = PM_x[valid2] * dV_m[valid2] - PM_y[valid2] * dU_m[valid2]
            gamma2[j, i] = np.mean(cross2 / denom2[valid2])

    return gamma1, gamma2


# ---------------------------------------------------------------------------
# 4. Vortex detection (thresholding + blob analysis)
# ---------------------------------------------------------------------------

def detect_vortices(field, omega, x, y, threshold, sign_filter='all',
                    min_area_mm2=1.0, frame=0):
    """
    Threshold scalar field, label blobs, compute per-blob properties.

    Parameters
    ----------
    field        : [ny, nx] -- scalar field to threshold
    omega        : [ny, nx] -- vorticity field (used for circulation)
    x, y         : [ny, nx] -- coordinates in mm
    threshold    : float -- absolute threshold value
    sign_filter  : 'all', 'positive', 'negative'
    min_area_mm2 : float -- minimum blob area in mm^2
    frame        : int -- frame index (0 for mean field)

    Returns
    -------
    list of dicts, one per detected vortex
    """
    dx_mm = float(np.abs(x[0, 1] - x[0, 0]))
    dy_mm = float(np.abs(y[1, 0] - y[0, 0]))
    pixel_area_mm2 = dx_mm * dy_mm
    dx_m = dx_mm / 1000.0
    dy_m = dy_mm / 1000.0

    if sign_filter == 'positive':
        blob_mask = field > threshold
    elif sign_filter == 'negative':
        blob_mask = field < -threshold
    else:
        blob_mask = np.abs(field) > threshold

    blob_mask = blob_mask & np.isfinite(field)

    labeled, n_blobs = ndimage.label(blob_mask)
    vortices = []

    for blob_id in range(1, n_blobs + 1):
        this_blob = labeled == blob_id
        area_mm2  = this_blob.sum() * pixel_area_mm2

        if area_mm2 < min_area_mm2:
            continue

        cy_idx, cx_idx = ndimage.center_of_mass(this_blob)
        j0 = int(np.clip(np.round(cy_idx), 0, x.shape[0] - 1))
        i0 = int(np.clip(np.round(cx_idx), 0, x.shape[1] - 1))
        x_center = float(x[j0, i0])
        y_center = float(y[j0, i0])

        omega_blob = omega[this_blob]
        omega_blob = omega_blob[np.isfinite(omega_blob)]
        sign = int(np.sign(np.nanmean(omega_blob))) if len(omega_blob) > 0 else 0
        circulation = float(np.nansum(omega_blob) * dx_m * dy_m)

        slices = ndimage.find_objects(this_blob.astype(int))[0]
        h = slices[0].stop - slices[0].start
        w = slices[1].stop - slices[1].start
        aspect_ratio = float(max(h, w)) / float(max(min(h, w), 1))

        vortices.append({
            'id'          : blob_id,
            'x_center'    : x_center,
            'y_center'    : y_center,
            'area_mm2'    : area_mm2,
            'sign'        : sign,
            'circulation' : circulation,
            'aspect_ratio': aspect_ratio,
            'frame'       : frame,
        })

    return vortices


# ---------------------------------------------------------------------------
# 5. Spatial probability map
# ---------------------------------------------------------------------------

def compute_spatial_probability(field, threshold, sign_filter='all'):
    """
    For a single [ny, nx] field return binary detection map (0 or 1).
    For stacked [ny, nx, Nt] fields return fraction of frames detected.
    """
    if field.ndim == 2:
        if sign_filter == 'positive':
            return (field > threshold).astype(float)
        elif sign_filter == 'negative':
            return (field < -threshold).astype(float)
        else:
            return (np.abs(field) > threshold).astype(float)
    else:
        if sign_filter == 'positive':
            return np.nanmean(field > threshold, axis=2)
        elif sign_filter == 'negative':
            return np.nanmean(field < -threshold, axis=2)
        else:
            return np.nanmean(np.abs(field) > threshold, axis=2)


# ---------------------------------------------------------------------------
# 6. Export
# ---------------------------------------------------------------------------

def export_vortex_csv(vortex_list, filepath):
    """Save per-vortex properties to CSV."""
    if not vortex_list:
        return
    fieldnames = ['id', 'x_center_mm', 'y_center_mm', 'area_mm2',
                  'sign', 'circulation_m2s', 'aspect_ratio', 'frame']
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for v in vortex_list:
            writer.writerow({
                'id'              : v['id'],
                'x_center_mm'     : f"{v['x_center']:.4f}",
                'y_center_mm'     : f"{v['y_center']:.4f}",
                'area_mm2'        : f"{v['area_mm2']:.4f}",
                'sign'            : v['sign'],
                'circulation_m2s' : f"{v['circulation']:.6f}",
                'aspect_ratio'    : f"{v['aspect_ratio']:.3f}",
                'frame'           : v['frame'],
            })
