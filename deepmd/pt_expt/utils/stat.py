# SPDX-License-Identifier: LGPL-3.0-or-later
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

    Keep sampled systems on the host and transfer only the current batch in
    ``forward``. The inference controller honors ``DP_INFER_BATCH_SIZE`` in
    atoms, retries CUDA OOM with smaller batches, and preserves frame order.
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
