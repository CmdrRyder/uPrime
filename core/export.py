"""
core/export.py
--------------
Export analysis results to file.

2D fields  -> Tecplot ASCII .dat
Line profiles -> CSV
Spectra    -> CSV

All exports include a settings header block.
"""

import os
import numpy as np
from datetime import datetime


def _settings_header(info_dict, comment="#"):
    """Build a settings comment block for export files."""
    lines = [
        f"{comment} uPrime Export",
        f"{comment} Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"{comment} {'='*50}",
    ]
    for k, v in info_dict.items():
        lines.append(f"{comment} {k:<25}: {v}")
    lines.append(f"{comment} {'='*50}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 2D field export -- Tecplot ASCII
# --------------------------------------------------------------------------- #

def export_2d_tecplot(filepath, x, y, fields, field_names, settings_info):
    """
    Export a 2D field to Tecplot ASCII .dat format.

    Parameters
    ----------
    filepath     : output file path (.dat)
    x, y         : [ny, nx] coordinate arrays in mm
    fields       : list of [ny, nx] arrays (mean values only)
    field_names  : list of strings, one per field
    settings_info: dict of settings to write in header
    """
    ny, nx = x.shape

    # Build variable list
    var_names = ["x [mm]", "y [mm]"] + field_names
    var_str   = ", ".join(f'"{v}"' for v in var_names)

    with open(filepath, "w", encoding="utf-8") as f:
        # Settings header as comments
        f.write(_settings_header(settings_info, comment="#"))
        f.write("\n")

        # Tecplot header
        f.write(f'TITLE = "{os.path.basename(filepath)}"\n')
        f.write(f"VARIABLES = {var_str}\n")
        f.write(f'ZONE T="uPrime Export", I={nx}, J={ny}, F=POINT\n')

        # Data -- row by row (top to bottom to match DaVis convention)
        # Flip back: our arrays are bottom-to-top, Tecplot expects top-to-bottom
        x_out = x[::-1, :]
        y_out = y[::-1, :]
        fields_out = [f[::-1, :] for f in fields]

        for j in range(ny):
            for i in range(nx):
                vals = [x_out[j, i], y_out[j, i]] + [
                    fld[j, i] if np.isfinite(fld[j, i]) else 0.0
                    for fld in fields_out
                ]
                f.write(" ".join(f"{v:.6g}" for v in vals) + "\n")


# --------------------------------------------------------------------------- #
# Line profile export -- CSV
# --------------------------------------------------------------------------- #

def export_line_csv(filepath, dist, xpts, ypts,
                    mean_dict, std_dict, settings_info):
    """
    Export line profile data to CSV.

    Parameters
    ----------
    filepath   : output file path (.csv)
    dist       : 1D array, distance along line [mm]
    xpts, ypts : 1D arrays, coordinates of points
    mean_dict  : dict {name: 1D mean array}
    std_dict   : dict {name: 1D std array}  (same keys as mean_dict)
    settings_info: dict of settings
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(_settings_header(settings_info, comment="#"))
        f.write("\n")

        # Column headers
        cols = ["dist_mm", "x_mm", "y_mm"]
        for name in mean_dict:
            cols.append(f"mean_{name}")
            if name in std_dict and std_dict[name] is not None:
                cols.append(f"std_{name}")
        f.write(",".join(cols) + "\n")

        # Data rows
        n = len(dist)
        for i in range(n):
            row = [dist[i], xpts[i], ypts[i]]
            for name, mean_arr in mean_dict.items():
                row.append(mean_arr[i] if np.isfinite(mean_arr[i]) else "")
                if name in std_dict and std_dict[name] is not None:
                    row.append(std_dict[name][i] if np.isfinite(std_dict[name][i]) else "")
            f.write(",".join(str(v) for v in row) + "\n")


# --------------------------------------------------------------------------- #
# Spectra export -- CSV
# --------------------------------------------------------------------------- #

def export_spectra_csv(filepath, freq, psd_dict, settings_info):
    """
    Export PSD data to CSV.

    Parameters
    ----------
    filepath    : output file path (.csv)
    freq        : 1D frequency array [Hz]
    psd_dict    : dict {component: 1D PSD array}
    settings_info: dict of settings
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(_settings_header(settings_info, comment="#"))
        f.write("\n")

        # Column headers
        comps = [k for k, v in psd_dict.items() if v is not None]
        cols  = ["frequency_Hz"] + [f"PSD_{c}_m2s2_per_Hz" for c in comps]
        f.write(",".join(cols) + "\n")

        # Data rows -- skip DC (freq=0)
        mask = freq > 0
        f_out = freq[mask]
        for i in range(len(f_out)):
            row = [f_out[i]]
            for c in comps:
                v = psd_dict[c][mask][i]
                row.append(v if np.isfinite(v) else "")
            f.write(",".join(str(v) for v in row) + "\n")
