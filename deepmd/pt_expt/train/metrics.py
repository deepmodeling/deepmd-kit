# SPDX-License-Identifier: LGPL-3.0-or-later
"""Torch helpers for distributed metric-window reduction."""

from __future__ import (
    annotations,
)

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

    # Packing always host-converts via float(); the collective device must
    # come from the process group, not from whether this rank's window
    # promoted any sums to tensors.
    device = _collective_device(group)
    tensor = torch.tensor(
        [float(value) for value in packed],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    accumulator.load_flat_state(tensor.detach().cpu().tolist())


def _collective_device(group: Any | None = None) -> torch.device:
    """Return one all-reduce device that every rank in ``group`` can use.

    Device choice follows the process-group backend and this rank's current
    CUDA device. It must not depend on local accumulator contents: a rank
    whose window never promoted sums to tensors would otherwise pick CPU
    while another rank picked CUDA, which breaks NCCL collectives.
    """
    backend = str(dist.get_backend(group)).lower()
    if "nccl" in backend and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
