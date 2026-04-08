"""
core/loader.py
Reads DaVis Tecplot-format .dat files into NumPy arrays.
One file = one snapshot. Supports 2D and Stereo PIV.
Invalid vectors (isValid == 0) are set to NaN.
"""

import os
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_header(filepath):
    header = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if re.match(r"^-?\d", stripped):
                break
            lines.append(stripped)

    n_skip = len(lines)

    var_line = ""
    for l in lines:
        if l.upper().startswith("VARIABLES"):
            var_line = l
            break

    variables = re.findall(r'"([^"]+)"', var_line)
    header["variables"] = variables

    zone_line = ""
    for l in lines:
        if l.upper().startswith("ZONE"):
            zone_line = l
            break

    nx_match = re.search(r"\bI\s*=\s*(\d+)", zone_line, re.IGNORECASE)
    ny_match = re.search(r"\bJ\s*=\s*(\d+)", zone_line, re.IGNORECASE)

    if not nx_match or not ny_match:
        raise ValueError(f"Could not find I= or J= in ZONE line of {filepath}")

    header["nx"] = int(nx_match.group(1))
    header["ny"] = int(ny_match.group(1))
    header["n_skip"] = n_skip

    # --- Column detection ---
    # Strategy: require variable name to START with "velocity" so that
    # "Acceleration u", "Acceleration v" are never mistaken for velocity.
    # This handles any DaVis export regardless of how many extra columns exist.

    col_x     = None
    col_y     = None
    col_u     = None
    col_v     = None
    col_w     = None
    col_vort  = None
    col_valid = None

    for i, var in enumerate(variables):
        vl = var.lower().strip()

        if vl.startswith("x ") or vl == "x":
            col_x = i
        elif vl.startswith("y ") or vl == "y":
            col_y = i
        elif vl.startswith("velocity u") or vl.startswith("velocity  u"):
            col_u = i
        elif vl.startswith("velocity v") or vl.startswith("velocity  v"):
            col_v = i
        elif vl.startswith("velocity w") or vl.startswith("velocity  w"):
            col_w = i
        elif "vorticity" in vl and col_vort is None:
            col_vort = i
        elif "isvalid" in vl:
            col_valid = i

    # Hard fallbacks if detection still fails
    if col_x     is None: col_x     = 0
    if col_y     is None: col_y     = 1
    if col_u     is None: col_u     = 2
    if col_v     is None: col_v     = 3
    if col_valid is None: col_valid = len(variables) - 1

    header["col_x"]     = col_x
    header["col_y"]     = col_y
    header["col_u"]     = col_u
    header["col_v"]     = col_v
    header["col_w"]     = col_w
    header["col_vort"]  = col_vort
    header["col_valid"] = col_valid
    header["is_stereo"] = col_w is not None
    header["has_vort"]  = col_vort is not None

    return header


def load_grid(filepath, header):
    nx = header["nx"]
    ny = header["ny"]
    n  = nx * ny

    raw = np.loadtxt(
        filepath,
        skiprows=header["n_skip"],
        usecols=[header["col_x"], header["col_y"]],
        max_rows=n,
        encoding="utf-8"
    )

    x = raw[:, 0].reshape(ny, nx)
    y = raw[:, 1].reshape(ny, nx)

    # DaVis stores rows top-to-bottom; flip so row 0 = smallest y
    if y[0, 0] > y[-1, 0]:
        x = x[::-1, :]
        y = y[::-1, :]

    return x, y


def load_single_file(filepath, header):
    nx = header["nx"]
    ny = header["ny"]
    n  = nx * ny

    cols_needed = [header["col_u"], header["col_v"]]
    if header["is_stereo"]:
        cols_needed.append(header["col_w"])
    if header.get("col_vort") is not None:
        cols_needed.append(header["col_vort"])
    cols_needed.append(header["col_valid"])
    cols_needed = sorted(set(cols_needed))

    raw = np.loadtxt(
        filepath,
        skiprows=header["n_skip"],
        usecols=cols_needed,
        max_rows=n,
        encoding="utf-8"
    )

    col_map = {c: i for i, c in enumerate(cols_needed)}

    u     = raw[:, col_map[header["col_u"]]].reshape(ny, nx)
    v     = raw[:, col_map[header["col_v"]]].reshape(ny, nx)
    valid = raw[:, col_map[header["col_valid"]]].reshape(ny, nx).astype(bool)

    if header["is_stereo"]:
        w = raw[:, col_map[header["col_w"]]].reshape(ny, nx)
    else:
        w = None

    if header.get("col_vort") is not None and header["col_vort"] in col_map:
        vort = raw[:, col_map[header["col_vort"]]].reshape(ny, nx)
    else:
        vort = None

    # Flip rows to match grid orientation (bottom to top)
    u     = u[::-1, :]
    v     = v[::-1, :]
    valid = valid[::-1, :]
    if w    is not None: w    = w[::-1, :]
    if vort is not None: vort = vort[::-1, :]

    # Mask invalid vectors
    u[~valid] = np.nan
    v[~valid] = np.nan
    if w    is not None: w[~valid]    = np.nan
    if vort is not None: vort[~valid] = np.nan

    return u, v, w, valid, vort


def _read_single_file(args):
    """Module-level worker for ThreadPoolExecutor (must be picklable)."""
    idx, filepath, header = args
    try:
        u, v, w, valid, vort = load_single_file(filepath, header)
        u     = u.astype(np.float32)
        v     = v.astype(np.float32)
        valid = valid.astype(bool)
        if w    is not None: w    = w.astype(np.float32)
        if vort is not None: vort = vort.astype(np.float32)
        return idx, u, v, w, valid, vort, None
    except Exception as e:
        return idx, None, None, None, None, None, str(e)


def load_dataset(file_list, progress_callback=None):
    if not file_list:
        raise ValueError("No files provided.")

    file_list = sorted(file_list)
    Nt = len(file_list)

    # --- Step 1: read first file to detect grid and metadata ---
    header    = parse_header(file_list[0])
    nx        = header["nx"]
    ny        = header["ny"]
    is_stereo = header["is_stereo"]
    has_vort  = header.get("has_vort", False)

    x, y = load_grid(file_list[0], header)

    # --- Step 2: pre-allocate output arrays as float32 ---
    U     = np.empty((ny, nx, Nt), dtype=np.float32)
    V     = np.empty((ny, nx, Nt), dtype=np.float32)
    W     = np.empty((ny, nx, Nt), dtype=np.float32) if is_stereo else None
    VALID = np.empty((ny, nx, Nt), dtype=bool)
    VORT  = np.empty((ny, nx, Nt), dtype=np.float32) if has_vort else None

    # Fill with NaN / False so skipped files are safe
    U[:]     = np.nan
    V[:]     = np.nan
    VALID[:] = False
    if W    is not None: W[:]    = np.nan
    if VORT is not None: VORT[:] = np.nan

    # --- Steps 3 & 4: parallel read with ThreadPoolExecutor ---
    args_list   = [(i, fp, header) for i, fp in enumerate(file_list)]
    report_every = max(1, Nt // 100)
    n_done       = 0
    max_workers  = min(8, os.cpu_count() or 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_read_single_file, args): args[0]
                   for args in args_list}

        for future in as_completed(futures):
            idx, u, v, w, valid, vort, err = future.result()

            if err is not None:
                print(f"Warning: skipping file {idx} ({os.path.basename(file_list[idx])}): {err}")
            else:
                U[:, :, idx]     = u
                V[:, :, idx]     = v
                VALID[:, :, idx] = valid
                if W    is not None and w    is not None: W[:, :, idx]    = w
                if VORT is not None and vort is not None: VORT[:, :, idx] = vort

            n_done += 1

            # --- Step 5: progress callback ---
            if progress_callback and (n_done % report_every == 0 or n_done == Nt):
                progress_callback(int(100 * n_done / Nt))

    # --- Step 6: if W is all-NaN, discard it ---
    if W is not None and np.all(np.isnan(W)):
        W         = None
        is_stereo = False

    # --- Step 7: pre-compute valid_frac ---
    valid_frac = np.mean(VALID.astype(np.float32), axis=2)

    # --- Step 8: return dataset ---
    return {
        "x": x, "y": y,
        "U": U, "V": V, "W": W,
        "vort": VORT,
        "valid": VALID,
        "valid_frac": valid_frac,
        "is_stereo": is_stereo,
        "has_vort": has_vort,
        "Nt": Nt, "nx": nx, "ny": ny,
        "files": file_list,
        "header": header,
    }
