# SPDX-License-Identifier: LGPL-3.0-or-later
"""Minimum pairwise distance check for frame validity filtering."""

from __future__ import (
    annotations,
)

import numpy as np

_MIN_PAIR_DIST_BLOCK_PAIRS = 262_144


def compute_min_pair_dist_single(
    coord: np.ndarray,
    box: np.ndarray | None,
    atype: np.ndarray,
    stop_below: float | None = None,
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
    stop_below : float or None
        Optional early-stop threshold. If a block has any pair closer
        than this value, the block minimum is returned immediately.

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

    # === Step 2. Prepare minimum image convention for PBC ===
    if box is not None:
        cell = box.reshape(3, 3)
        inv_cell = np.linalg.inv(cell)
    else:
        cell = None
        inv_cell = None

    # === Step 3. Compute distances in bounded row blocks ===
    block_size = max(1, min(n_real, _MIN_PAIR_DIST_BLOCK_PAIRS // n_real))
    min_dist_sq = float("inf")
    stop_dist_sq = (
        float(stop_below) * float(stop_below)
        if stop_below is not None and stop_below > 0.0
        else None
    )
    for start in range(0, n_real, block_size):
        stop = min(start + block_size, n_real)
        diff = real_coord[np.newaxis, :, :] - real_coord[start:stop, np.newaxis, :]

        if cell is not None and inv_cell is not None:
            frac_diff = diff @ inv_cell
            frac_diff -= np.round(frac_diff)
            diff = frac_diff @ cell

        dist_sq = np.sum(diff * diff, axis=-1)
        rows = np.arange(stop - start, dtype=np.int64)
        dist_sq[rows, start + rows] = np.inf
        min_dist_sq = min(min_dist_sq, float(dist_sq.min()))
        if min_dist_sq == 0.0 or (
            stop_dist_sq is not None and min_dist_sq < stop_dist_sq
        ):
            break

    return float(np.sqrt(min_dist_sq))
