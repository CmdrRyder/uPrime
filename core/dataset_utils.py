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

    Parameters
    ----------
    dataset : dict   -- the dataset dict returned by load_dataset
    key     : str    -- "U", "V", or "W"

    Returns
    -------
    out : ndarray [ny, nx, Nt] float64, or None if dataset[key] is None
    """
    field = dataset[key]
    if field is None:
        return None
    if dataset.get("mask_active", True):
        mask = dataset["MASK"]          # [ny, nx], True = valid
        out  = field.astype(float, copy=True)
        out[~mask, :] = np.nan          # broadcast across all Nt frames
        return out
    return field.astype(float, copy=True)
