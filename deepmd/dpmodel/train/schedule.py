# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent resolution of the training-step schedule.

A run length is expressed either directly in optimizer steps
(``training.numb_steps``) or as a number of passes over the training data
(``training.numb_epoch`` for single-task runs, ``training.num_epoch_dict``
for multi-task runs).  Converting epochs into steps requires exactly one
backend-specific quantity, the epoch length of a task, which callers supply
through the ``epoch_length`` callback.  Validation of the mutually exclusive
options, the multi-task step split and the resulting task sampling weights
are shared by every backend.
"""

from __future__ import (
    annotations,
)

import logging
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.training_utils import (
    resolve_model_prob,
    resolve_model_prob_from_epochs,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
        Sequence,
    )

log = logging.getLogger(__name__)

__all__ = ["StepSchedule", "resolve_step_schedule"]


@dataclass(frozen=True)
class StepSchedule:
    """Resolved run length and task sampling weights.

    Attributes
    ----------
    num_steps : int
        Total number of optimizer steps of the run.
    model_prob : np.ndarray | None
        Probability of selecting each task at a training step, with shape
        ``(ntasks,)`` and ordered like the ``model_keys`` passed to
        :func:`resolve_step_schedule`.  ``None`` for single-task runs.
    """

    num_steps: int
    model_prob: np.ndarray | None = None


def resolve_step_schedule(
    training_params: Mapping[str, Any],
    *,
    multi_task: bool,
    model_keys: Sequence[str],
    training_data: Mapping[str, Any],
    epoch_length: Callable[[str], int],
    broadcast: Callable[[list[int]], Sequence[int]] | None = None,
    rank: int = 0,
) -> StepSchedule:
    """Resolve the run length and the task sampling weights.

    Parameters
    ----------
    training_params : Mapping[str, Any]
        The normalized ``training`` section.  ``numb_steps``, ``numb_epoch``,
        ``num_epoch_dict`` and ``model_prob`` are read from it.
    multi_task : bool
        Whether the run trains several task branches.  Single-task runs accept
        ``numb_epoch``, multi-task runs accept ``num_epoch_dict``.
    model_keys : Sequence[str]
        Task keys in the order used by the trainer.  Single-task runs pass the
        one key under which their data is registered.
    training_data : Mapping[str, Any]
        Training data of every task, keyed like ``model_keys``.  Only consulted
        when multi-task sampling weights fall back to the per-task data size.
    epoch_length : Callable[[str], int]
        Maps a task key to the number of steps *this rank* performs during one
        epoch of that task.  Backends whose data pipeline is sharded across
        ranks report their per-rank batch count directly; backends that
        replicate the dataset on every rank divide the dataset-wide batch count
        by the world size, so that an epoch always denotes one pass over the
        whole dataset.
    broadcast : Callable[[list[int]], Sequence[int]], optional
        Replaces the epoch lengths by rank 0's values.  Epoch lengths follow
        from floating-point sampling weights and may otherwise differ by one
        unit across ranks, which would desynchronize the run length and
        deadlock later collective calls.
    rank : int, optional
        Process rank, used to restrict informational logging to the chief.

    Returns
    -------
    StepSchedule
        The resolved run length and, for multi-task runs, the task sampling
        weights.

    Raises
    ------
    ValueError
        If the run length is unspecified, over-specified, or non-positive.
    """
    keys = list(model_keys)
    num_steps = training_params.get("numb_steps")
    num_epoch = training_params.get("numb_epoch")
    num_epoch_dict = training_params.get("num_epoch_dict") or {}

    if not multi_task:
        if num_steps is not None:
            return StepSchedule(num_steps=int(num_steps))
        if num_epoch is None:
            raise ValueError(
                "Either training.numb_steps or training.num_epoch must be set."
            )
        num_epoch = float(num_epoch)
        if num_epoch <= 0.0:
            raise ValueError("training.num_epoch must be positive.")
        (total_numb_batch,) = _epoch_lengths(keys, epoch_length, broadcast)
        steps = int(np.ceil(num_epoch * total_numb_batch))
        if rank == 0:
            log.info(
                "Computed num_steps=%d from num_epoch=%s and total_numb_batch=%d.",
                steps,
                num_epoch,
                total_numb_batch,
            )
        return StepSchedule(num_steps=steps)

    if num_epoch_dict:
        if num_steps is not None:
            raise ValueError(
                "training.numb_steps and training.num_epoch_dict "
                "are mutually exclusive."
            )
        per_task_total = _epoch_lengths(keys, epoch_length, broadcast)
        model_prob, steps, per_task_steps = resolve_model_prob_from_epochs(
            keys,
            num_epoch_dict,
            np.asarray(per_task_total, dtype=np.float64),
        )
        if rank == 0:
            log.info(
                "Computed model_prob=%s and num_steps=%d from num_epoch_dict=%s "
                "with per-task target steps: %s.",
                model_prob,
                steps,
                dict(num_epoch_dict),
                {k: int(np.ceil(v)) for k, v in per_task_steps.items()},
            )
        return StepSchedule(num_steps=steps, model_prob=model_prob)

    if num_steps is None:
        raise ValueError(
            "Either training.numb_steps (multi-task only) or "
            "training.num_epoch_dict must be set."
        )
    model_prob = resolve_model_prob(
        keys,
        training_params.get("model_prob"),
        training_data,
        rank=rank,
    )
    return StepSchedule(num_steps=int(num_steps), model_prob=model_prob)


def _epoch_lengths(
    model_keys: Sequence[str],
    epoch_length: Callable[[str], int],
    broadcast: Callable[[list[int]], Sequence[int]] | None,
) -> list[int]:
    """Collect the per-task epoch lengths agreed upon by every rank."""
    lengths = [int(epoch_length(model_key)) for model_key in model_keys]
    if broadcast is not None:
        lengths = [int(value) for value in broadcast(lengths)]
    for model_key, length in zip(model_keys, lengths, strict=True):
        if length <= 0:
            raise ValueError(
                f"Number of training batches per epoch must be positive for "
                f"task '{model_key}', got {length}."
            )
    return lengths
