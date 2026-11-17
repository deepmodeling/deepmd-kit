# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sampling probabilities shared by data formats and training backends."""

from collections.abc import (
    Sequence,
)

import numpy as np

from deepmd.utils import random as dp_random


def process_sys_probs(
    sys_probs: Sequence[float], nbatch: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Fill unspecified probabilities in proportion to the batch counts.

    Non-negative entries specify absolute probabilities. Negative entries
    share the remaining probability in proportion to their batch counts.
    A fully assigned probability vector does not depend on batch counts.
    """
    probabilities = np.asarray(sys_probs)
    if not np.issubdtype(probabilities.dtype, np.floating):
        probabilities = probabilities.astype(np.float64)
    if probabilities.ndim != 1:
        raise ValueError("Sampling probabilities must be one-dimensional.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Sampling probabilities must be finite.")
    assigned = probabilities >= 0
    assigned_sum = float(probabilities[assigned].sum())
    tolerance = max(1e-8, float(np.finfo(probabilities.dtype).eps))
    if assigned_sum > 1.0 + tolerance:
        raise ValueError("The sum of assigned probabilities must not exceed 1.")
    result = np.where(assigned, probabilities, 0.0)
    remaining = max(0.0, 1.0 - assigned_sum)
    if remaining > tolerance:
        sizes = np.asarray(nbatch, dtype=np.float64)
        if probabilities.shape != sizes.shape:
            raise ValueError(
                "Sampling probabilities and batch counts must have the same length."
            )
        if not np.all(np.isfinite(sizes)) or np.any(sizes < 0):
            raise ValueError("Batch counts must be finite and non-negative.")
        remaining_size = float(sizes[~assigned].sum())
        if remaining_size <= 0:
            raise ValueError(
                "The remaining probability requires at least one unassigned, "
                "nonempty dataset."
            )
        result[~assigned] = remaining * sizes[~assigned] / remaining_size
    if result.size == 0 or result.sum() <= 0:
        raise ValueError("Sampling probabilities must contain a positive entry.")
    return result / result.sum()


def prob_sys_size_ext(
    keywords: str, nsystems: int, nbatch: Sequence[int] | np.ndarray
) -> np.ndarray:
    """Distribute each configured block's weight by its system sizes."""
    blocks = [part.split(":") for part in keywords.split(";")[1:]]
    weights = np.asarray([float(block[2]) for block in blocks])
    if np.any(weights < 0):
        raise ValueError("The weight of a block must be non-negative.")
    block_probs = weights / weights.sum()
    sys_probs = np.zeros(nsystems, dtype=np.float64)
    sizes = np.asarray(nbatch, dtype=np.float64)
    for block, probability in zip(blocks, block_probs, strict=True):
        start, end = int(block[0]), int(block[1])
        block_sizes = sizes[start:end]
        sys_probs[start:end] = block_sizes / block_sizes.sum() * probability
    return sys_probs


def uniform_sampling_weights(batch_nframes: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return inverse batch sizes, excluding streams that supply no frames."""
    sizes = np.asarray(batch_nframes, dtype=np.float64)
    return np.divide(1.0, sizes, out=np.zeros_like(sizes), where=sizes > 0)


def system_probabilities(
    nbatches: Sequence[int] | np.ndarray,
    *,
    sys_probs: Sequence[float] | None = None,
    auto_prob: str = "prob_sys_size",
    batch_nframes: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Resolve batch probabilities, correcting uniform sampling for frame counts.

    ``batch_nframes`` gives the mean frames per batch of each native stream.
    Its reciprocal makes ``prob_uniform`` allocate equal sampled frame shares
    rather than equal batch counts. Without batching metadata, every draw is
    treated as one frame. Explicit ``sys_probs`` remain batch probabilities.
    """
    nsystems = len(nbatches)
    if nsystems == 0:
        raise ValueError("No systems provided.")
    if sys_probs is not None:
        return process_sys_probs(sys_probs, nbatches)
    if auto_prob == "prob_uniform":
        sizes = (
            np.ones(nsystems, dtype=np.float64)
            if batch_nframes is None
            else batch_nframes
        )
        weights = uniform_sampling_weights(sizes)
        return group_probabilities(weights, [None] * nsystems)
    if auto_prob == "prob_sys_size":
        auto_prob = f"prob_sys_size;0:{nsystems}:1.0"
    if not auto_prob.startswith("prob_sys_size;"):
        raise RuntimeError(f"Unknown auto prob style: {auto_prob}")
    return prob_sys_size_ext(auto_prob, nsystems, nbatches)


def group_probabilities(
    sizes: Sequence[float] | np.ndarray, probabilities: Sequence[float | None]
) -> np.ndarray:
    """Assign group probabilities, sharing any remainder by group size."""
    if len(sizes) != len(probabilities):
        raise ValueError("Group sizes and probabilities must have the same length.")
    if any(
        probability is not None and probability < 0 for probability in probabilities
    ):
        raise ValueError("A group's prob must be non-negative or omitted.")
    resolved = process_sys_probs(
        np.asarray(
            [
                -1.0 if probability is None else probability
                for probability in probabilities
            ],
            dtype=np.float64,
        ),
        sizes,
    )
    if np.any((np.asarray(sizes) == 0) & (resolved > 0)):
        raise ValueError("A data group with positive probability must contain batches.")
    return resolved


def data_child_seeds(seed: int | None, count: int) -> list[int]:
    """Derive independent selection streams for nodes in a dataset hierarchy."""
    if seed is None:
        seed = int(dp_random.choice(2**32))
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(seed).spawn(count)
    ]


def global_batch_counts(
    frame_counts: Sequence[int], batch_sizes: Sequence[int]
) -> list[int]:
    """Count source batches before distributed sampling and rank-local padding."""
    return [
        (frames + batch_size - 1) // batch_size
        for frames, batch_size in zip(frame_counts, batch_sizes, strict=True)
    ]
