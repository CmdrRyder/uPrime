"""
core/dataset_utils.py
---------------------
Utilities for accessing dataset arrays through the non-destructive mask system.
"""

import numpy as np


def get_masked(dataset, key):
    """
    Return dataset[key] with invalid points set to NaN if mask is active.
    Returns a copy; never modifies the original array.

    The output dtype matches the stored array (float16, float32, or float64).
    All floating-point dtypes — including float16 — support IEEE 754 NaN.

    Parameters
    ----------
    dataset : dict   -- the dataset dict returned by load_dataset
    key     : str    -- "U", "V", or "W"

    Returns
    -------
    out : ndarray [ny, nx, Nt], same dtype as dataset[key], or None
    """
    field = dataset[key]
    if field is None:
        return None
    if dataset.get("mask_active", True):
        mask = dataset["MASK"]          # [ny, nx], True = valid
        out  = field.copy()
        out[~mask, :] = np.nan          # broadcast across all Nt frames
        return out
    return field.copy()
