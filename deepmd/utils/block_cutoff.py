# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-block cutoff radii of message-passing descriptors."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import (
        Sequence,
    )


def resolve_block_cutoffs(rcut: float | Sequence[float], n_blocks: int) -> list[float]:
    """Return the cutoff radius of every message-passing block.

    A descriptor that stacks ``n_blocks`` message-passing blocks accepts one
    cutoff for all of them or one cutoff per block. The neighbor graph is built
    at the largest entry; whether the entries must be ordered is a property of
    the descriptor that consumes them.

    Parameters
    ----------
    rcut : float or Sequence[float]
        A cutoff radius in Å shared by every block, or one radius per block.
    n_blocks : int
        Number of message-passing blocks.

    Returns
    -------
    list[float]
        Cutoff radius of every block in Å, with length ``n_blocks``.

    Raises
    ------
    ValueError
        If ``n_blocks`` is not a positive integer, if a sequence does not hold
        exactly ``n_blocks`` entries, or if a cutoff is not positive.
    """
    if isinstance(n_blocks, bool) or int(n_blocks) != n_blocks or n_blocks < 1:
        raise ValueError(f"`n_blocks` must be a positive integer, got {n_blocks!r}")
    n_blocks = int(n_blocks)
    if np.isscalar(rcut):
        cutoffs = [float(rcut)] * n_blocks
    else:
        cutoffs = [float(radius) for radius in rcut]
        if len(cutoffs) != n_blocks:
            raise ValueError(
                f"`rcut` lists {len(cutoffs)} cutoff(s), but `n_blocks` is "
                f"{n_blocks}; give one cutoff per block or a single cutoff for all"
            )
    if not all(radius > 0.0 for radius in cutoffs):
        raise ValueError(f"Every cutoff in `rcut` must be positive, got {rcut!r}")
    return cutoffs
