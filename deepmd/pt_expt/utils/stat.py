# SPDX-License-Identifier: LGPL-3.0-or-later
"""Sampling and bounded prediction of output statistics on the pt_expt backend."""

from collections.abc import (
    Callable,
)
from functools import (
    wraps,
)

import numpy as np
import torch

from deepmd.pt.utils.auto_batch_size import (
    AutoBatchSize,
)
from deepmd.utils.model_stat import (
    make_stat_input,
)

__all__ = ["batch_stat_forward", "make_stat_input"]


def batch_stat_forward(forward: Callable) -> Callable:
    """Predict output statistics in bounded, gradient-free inference batches.

    A sampled system holds every frame its statistics use, which a large
    system cannot evaluate at once next to the neighbor lists it builds; the
    inference controller keeps the prediction within ``DP_INFER_BATCH_SIZE``
    atoms, retries a CUDA out-of-memory error with smaller batches, and keeps
    the frame order, which the pairing of predictions with labels relies on.
    The prediction carries no gradient: the statistics only read values.

    Parameters
    ----------
    forward
        A forward of the atomic model, taking per-frame ``coord`` and
        ``atype`` with shape (nframes, ...) and arrays of batch-leading shape
        after them, and returning a dict of batch-leading arrays.

    Returns
    -------
    Callable
        The same call, evaluated batch by batch.
    """
    batch_size = AutoBatchSize()

    @wraps(forward)
    @torch.no_grad()
    def batched(
        coord: np.ndarray,
        atype: np.ndarray,
        box: np.ndarray | None,
        **kwargs: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        return batch_size.execute_all(
            forward, atype.shape[0], atype.shape[1], coord, atype, box, **kwargs
        )

    return batched
