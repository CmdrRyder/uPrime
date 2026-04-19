"""
core/loader.py
Reads DaVis Tecplot-format .dat files into NumPy arrays.
One file = one snapshot. Supports 2D and Stereo PIV.
Invalid vectors (isValid == 0) are set to NaN.
"""

import os
import re
import tempfile
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


SIZE_THRESHOLD = 4 * 1024 ** 3   # 4 GB in bytes


def estimate_dataset_size(file_list, header, stride=1):
    """Return estimated memory in bytes for the dataset (float32)."""
    ny = header["ny"]
    nx = header["nx"]
    n_components = 3 if header["is_stereo"] else 2
    n_files = len(file_list) // max(1, stride)
    return ny * nx * n_files * n_components * 4


def cleanup_memmap(dataset):
    """Delete memmap temp files created during load_dataset, if any."""
    path = dataset.get("_memmap_path")
    if not path:
        return
    for suffix in ("_U", "_V", "_W"):
        try:
            os.remove(path + suffix)
        except FileNotFoundError:
            pass
    dataset["_memmap_path"] = None


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

    # Hard fallbacks for position/velocity columns if detection still fails.
    # col_valid intentionally has NO fallback: if the column is absent we treat
    # all vectors as valid rather than misreading an unrelated data column.
    if col_x is None: col_x = 0
    if col_y is None: col_y = 1
    if col_u is None: col_u = 2
    if col_v is None: col_v = 3

    header["col_x"]     = col_x
    header["col_y"]     = col_y
    header["col_u"]     = col_u
    header["col_v"]     = col_v
    header["col_w"]     = col_w
    header["col_vort"]  = col_vort
    header["col_valid"] = col_valid          # may be None if absent in file
    header["has_valid"] = col_valid is not None
    header["is_stereo"] = col_w is not None
    header["has_vort"]  = col_vort is not None

    # --- Unit extraction ---
    x_var = variables[col_x] if col_x < len(variables) else ""
    u_var = variables[col_u] if col_u < len(variables) else ""

    x_match = re.search(r'\[([^\]]+)\]', x_var)
    if x_match:
        unit_str = x_match.group(1)
        if 'mm' in unit_str:
            x_unit = 'mm'
        elif 'm' in unit_str:
            x_unit = 'm'
        else:
            x_unit = 'mm'
    else:
        x_unit = 'mm'

    u_match = re.search(r'\[([^\]]+)\]', u_var)
    if u_match:
        unit_str = u_match.group(1)
        if 'mm/s' in unit_str:
            vel_unit = 'mm/s'
        elif 'm/s' in unit_str:
            vel_unit = 'm/s'
        else:
            vel_unit = 'm/s'
    else:
        vel_unit = 'm/s'

    header["x_unit"]    = x_unit
    header["vel_unit"]  = vel_unit
    header["xy_to_mm"]  = 1000.0 if x_unit == 'm' else 1.0
    header["vel_to_ms"] = 0.001  if vel_unit == 'mm/s' else 1.0

    print(f"[loader] Units detected: xy={x_unit} (scale x{header['xy_to_mm']}), vel={vel_unit} (scale x{header['vel_to_ms']})")

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

    x = x * header["xy_to_mm"]
    y = y * header["xy_to_mm"]

    return x, y


def load_single_file(filepath, header):
    nx = header["nx"]
    ny = header["ny"]
    n  = nx * ny

    has_valid = header.get("has_valid", True)
    col_valid = header.get("col_valid")   # None when column is absent

    cols_needed = [header["col_u"], header["col_v"]]
    if header["is_stereo"]:
        cols_needed.append(header["col_w"])
    if header.get("col_vort") is not None:
        cols_needed.append(header["col_vort"])
    if has_valid and col_valid is not None:
        cols_needed.append(col_valid)
    cols_needed = sorted(set(cols_needed))

    raw = np.loadtxt(
        filepath,
        skiprows=header["n_skip"],
        usecols=cols_needed,
        max_rows=n,
        encoding="utf-8"
    )

    col_map = {c: i for i, c in enumerate(cols_needed)}

    u = raw[:, col_map[header["col_u"]]].reshape(ny, nx)
    v = raw[:, col_map[header["col_v"]]].reshape(ny, nx)

    if has_valid and col_valid is not None and col_valid in col_map:
        valid = raw[:, col_map[col_valid]].reshape(ny, nx).astype(bool)
    else:
        # isValid column absent — treat every vector as valid
        valid = np.ones((ny, nx), dtype=bool)

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

    u = u * header["vel_to_ms"]
    v = v * header["vel_to_ms"]
    if w is not None: w = w * header["vel_to_ms"]

    # Apply valid mask to vorticity only; U/V/W stay raw (mask is stored separately)
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

    # --- Step 2: decide between in-memory and memory-mapped storage ---
    est_size   = estimate_dataset_size(file_list, header)
    use_memmap = est_size > SIZE_THRESHOLD

    mask_2d  = None
    VORT     = None
    tmp_path = None

    if use_memmap:
        print(f"[loader] Dataset estimated at {est_size / 1024**3:.1f} GB — "
              "using memory-mapped arrays.")
        tmp_dir  = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"uprime_memmap_{os.getpid()}.bin")
        shape    = (ny, nx, Nt)
        U = np.memmap(tmp_path + '_U', dtype='float32', mode='w+', shape=shape)
        V = np.memmap(tmp_path + '_V', dtype='float32', mode='w+', shape=shape)
        W = np.memmap(tmp_path + '_W', dtype='float32', mode='w+', shape=shape) \
            if is_stereo else None
        U[:] = np.nan
        V[:] = np.nan
        if W is not None:
            W[:] = np.nan

        # Sequential load into memmap
        report_every = max(1, Nt // 100)
        for i, fpath in enumerate(file_list):
            try:
                u, v, w, valid, _vort = load_single_file(fpath, header)
                U[:, :, i] = u.astype(np.float32)
                V[:, :, i] = v.astype(np.float32)
                if W is not None and w is not None:
                    W[:, :, i] = w.astype(np.float32)
                if i == 0:
                    mask_2d = valid
            except Exception as e:
                print(f"Warning: skipping file {i} "
                      f"({os.path.basename(fpath)}): {e}")
            if progress_callback and (i % report_every == 0 or i == Nt - 1):
                progress_callback(int(100 * (i + 1) / Nt))

        U.flush()
        V.flush()
        if W is not None:
            W.flush()

    else:
        # --- In-memory: pre-allocate and parallel read ---
        U    = np.empty((ny, nx, Nt), dtype=np.float32)
        V    = np.empty((ny, nx, Nt), dtype=np.float32)
        W    = np.empty((ny, nx, Nt), dtype=np.float32) if is_stereo else None
        VORT = np.empty((ny, nx, Nt), dtype=np.float32) if has_vort  else None
        U[:] = np.nan
        V[:] = np.nan
        if W    is not None: W[:]    = np.nan
        if VORT is not None: VORT[:] = np.nan

        args_list    = [(i, fp, header) for i, fp in enumerate(file_list)]
        report_every = max(1, Nt // 100)
        n_done       = 0
        max_workers  = min(8, os.cpu_count() or 1)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_read_single_file, args): args[0]
                       for args in args_list}

            for future in as_completed(futures):
                idx, u, v, w, valid, vort, err = future.result()

                if err is not None:
                    print(f"Warning: skipping file {idx} "
                          f"({os.path.basename(file_list[idx])}): {err}")
                else:
                    U[:, :, idx] = u
                    V[:, :, idx] = v
                    if W    is not None and w    is not None: W[:, :, idx]    = w
                    if VORT is not None and vort is not None: VORT[:, :, idx] = vort
                    if idx == 0:
                        mask_2d = valid

                n_done += 1
                if progress_callback and (n_done % report_every == 0 or n_done == Nt):
                    progress_callback(int(100 * n_done / Nt))

    # --- Finalise ---
    if W is not None and np.all(np.isnan(W)):
        if use_memmap:
            try:
                os.remove(tmp_path + '_W')
            except FileNotFoundError:
                pass
        W         = None
        is_stereo = False

    if mask_2d is None:
        mask_2d = np.ones((ny, nx), dtype=bool)

    valid_frac = mask_2d.astype(np.float32)

    return {
        "x": x, "y": y,
        "U": U, "V": V, "W": W,
        "vort": VORT,
        "valid":          mask_2d,
        "valid_frac":     valid_frac,
        "MASK":           mask_2d,
        "MASK_LOADED":    mask_2d.copy(),
        "mask_active":    True,
        "is_stereo":      is_stereo,
        "has_vort":       has_vort,
        "Nt": Nt, "nx": nx, "ny": ny,
        "files":          file_list,
        "header":         header,
        "_memmap_path":   tmp_path,
    }
