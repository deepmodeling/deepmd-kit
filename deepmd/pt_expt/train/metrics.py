# SPDX-License-Identifier: LGPL-3.0-or-later
"""Torch helpers for distributed metric-window reduction."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from deepmd.dpmodel.train.metrics import (
        MetricAccumulator,
    )


def all_reduce_metric_accumulator(
    accumulator: MetricAccumulator,
    *,
    group: Any | None = None,
) -> None:
    """Sum local metric windows across ranks in place.

    Call only at a display boundary. Host synchronization of the averaged
    scalars still happens later in :meth:`MetricAccumulator.average`; this
    routine only exchanges the packed sums, weights, and observation counts.
    Single-process runs leave the accumulator unchanged.
    """
    if not dist.is_available() or not dist.is_initialized():
        return
    world_size = dist.get_world_size(group)
    if world_size <= 1:
        return

    packed = accumulator.flat_state()
    if not packed:
        return

    device = _state_device(packed)
    tensor = torch.tensor(
        [float(value) for value in packed],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    accumulator.load_flat_state(tensor.detach().cpu().tolist())


def _state_device(packed: list[Any]) -> torch.device:
    """Prefer the device of the first tensor entry; otherwise use CPU."""
    for value in packed:
        if torch.is_tensor(value):
            return value.device
    return torch.device("cpu")
