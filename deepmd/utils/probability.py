# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sampling probabilities shared by NPY and LMDB without importing loaders."""

from collections.abc import (
    Sequence,
)

import numpy as np


def prob_sys_size_ext(
    keywords: str, nsystems: int, nbatch: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Distribute normalized block weights in proportion to system sizes.

    ``keywords`` has the form ``prob_sys_size;start:end:weight;...``, with
    exclusive end indices. ``nbatch`` supplies one size per system: batch
    counts for NPY data or frame counts for LMDB data. Callers must supply
    nonempty blocks with positive total size and positive total block weight.
    Systems outside the specified blocks retain zero probability.
    """
    block_str = keywords.split(";")[1:]
    block_stt = []
    block_end = []
    block_weights = []
    for ii in block_str:
        stt = int(ii.split(":")[0])
        end = int(ii.split(":")[1])
        weight = float(ii.split(":")[2])
        assert weight >= 0, "the weight of a block should be no less than 0"
        block_stt.append(stt)
        block_end.append(end)
        block_weights.append(weight)
    nblocks = len(block_str)
    block_probs = np.array(block_weights) / np.sum(block_weights)
    sys_probs = np.zeros([nsystems], dtype=np.float64)
    for ii in range(nblocks):
        nbatch_block = nbatch[block_stt[ii] : block_end[ii]]
        tmp_prob = [float(i) for i in nbatch_block] / np.sum(nbatch_block)
        sys_probs[block_stt[ii] : block_end[ii]] = tmp_prob * block_probs[ii]
    return sys_probs
