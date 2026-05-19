# SPDX-License-Identifier: LGPL-3.0-or-later
"""Minimum pairwise distance check for frame validity filtering."""

from __future__ import (
    annotations,
)

import numpy as np


def compute_min_pair_dist_single(
    coord: np.ndarray,
    box: np.ndarray | None,
    atype: np.ndarray,
) -> float:
    """Compute the minimum pairwise atomic distance for a single frame.

    Parameters
    ----------
    coord : np.ndarray
        Atomic coordinates, flattened with shape (natoms * 3,)
        or reshaped as (natoms, 3).
    box : np.ndarray or None
        Box vectors with shape (9,) for PBC, or None for non-PBC.
    atype : np.ndarray
        Atom types with shape (natoms,). Virtual atoms (type < 0)
        are excluded from the distance check.

    Returns
    -------
    float
        Minimum pairwise distance. Returns inf if fewer than 2
        real atoms exist.
    """
    coord = coord.reshape(-1, 3)

    # === Step 1. Filter out virtual atoms ===
    real_mask = atype.ravel() >= 0
    real_coord = coord[real_mask]
    n_real = real_coord.shape[0]
    if n_real < 2:
        return float("inf")

    # === Step 2. Compute pairwise displacement vectors ===
    diff = real_coord[np.newaxis, :, :] - real_coord[:, np.newaxis, :]

    # === Step 3. Apply minimum image convention for PBC ===
    if box is not None:
        cell = box.reshape(3, 3)
        inv_cell = np.linalg.inv(cell)
        frac_diff = diff @ inv_cell
        frac_diff -= np.round(frac_diff)
        diff = frac_diff @ cell

    # === Step 4. Compute distances and exclude self-pairs ===
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, np.inf)

    return float(np.sqrt(dist_sq.min()))
