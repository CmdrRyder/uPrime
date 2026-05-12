import numpy as np


def write_tecplot_mean(ds, filepath):
    """Write mean velocity field to a Tecplot ASCII .dat file."""
    for key in ("x", "y", "U", "V"):
        if key not in ds:
            raise KeyError(f"Dataset is missing required key '{key}'")

    x = ds["x"]
    y = ds["y"]
    ny, nx = x.shape

    U = np.nanmean(ds["U"], axis=-1)
    V = np.nanmean(ds["V"], axis=-1)

    has_w = "W" in ds and ds["W"] is not None
    if has_w:
        W = np.nanmean(ds["W"], axis=-1)
        MAG = np.sqrt(U**2 + V**2 + W**2)
        variables = "X Y U V W MAG"
        data_arrays = (x, y, U, V, W, MAG)
    else:
        MAG = np.sqrt(U**2 + V**2)
        variables = "X Y U V MAG"
        data_arrays = (x, y, U, V, MAG)

    flat = [a.ravel(order="C") for a in data_arrays]
    n_points = flat[0].shape[0]

    with open(filepath, "w", encoding="utf-8") as f:
        f.write('TITLE = "uPrime Mean Velocity Export"\n')
        f.write(f"VARIABLES = {variables}\n")
        f.write(f'ZONE T="Zone 1", I={nx}, J={ny}, DATAPACKING=POINT\n')
        for i in range(n_points):
            f.write(" ".join(f"{a[i]:.6f}" for a in flat) + "\n")
