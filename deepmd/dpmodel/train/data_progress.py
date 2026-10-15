# SPDX-License-Identifier: LGPL-3.0-or-later
"""Checkpointable training-data progress for restartable optimizers.

A training restart restores model, optimizer, learning-rate, and EMA state,
but historically rebuilt every data source from the beginning of epoch zero.
This module defines the backend-neutral payload that records each task's
logical data cursor — the next training batch the optimizer has not yet
consumed — and the process RNG that drives multi-task and directory-system
selection.

Prefetch and decoder read-ahead are excluded from the payload. Restoring
progress rebuilds iterators at the saved cursor so the next ``get_batch``
matches an uninterrupted run without decoding the dataset during seek.
"""

from __future__ import (
    annotations,
)

from collections.abc import (
    Mapping,
)
from typing import (
    Any,
)

from deepmd.utils import random as dp_random

DATA_PROGRESS_CHECKPOINT_KEY = "data_progress"
DATA_PROGRESS_VERSION = 1

__all__ = [
    "DATA_PROGRESS_CHECKPOINT_KEY",
    "DATA_PROGRESS_VERSION",
    "collect_training_data_progress",
    "restore_training_data_progress",
]


def collect_training_data_progress(
    training_data_by_task: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect per-task data-source progress and the shared selection RNG.

    Parameters
    ----------
    training_data_by_task : Mapping[str, Any]
        Training data sources keyed by task name.

    Returns
    -------
    dict[str, Any]
        Versioned checkpoint payload. Tasks whose data source does not
        implement ``state_dict`` are omitted; an empty ``tasks`` map still
        carries the RNG so multi-task selection can resume.
    """
    tasks: dict[str, Any] = {}
    for task_key, data in training_data_by_task.items():
        state_dict = getattr(data, "state_dict", None)
        if callable(state_dict):
            tasks[task_key] = state_dict()
    return {
        "version": DATA_PROGRESS_VERSION,
        "tasks": tasks,
        "rng": dp_random.get_state(),
    }


def restore_training_data_progress(
    training_data_by_task: Mapping[str, Any],
    progress: Mapping[str, Any] | None,
) -> None:
    """Restore per-task data progress and the shared selection RNG.

    Parameters
    ----------
    training_data_by_task : Mapping[str, Any]
        Training data sources keyed by task name. Sources without
        ``load_state_dict`` are left unchanged.
    progress : Mapping[str, Any] or None
        Payload previously produced by :func:`collect_training_data_progress`.
        ``None`` or a missing payload retains the legacy restart behaviour
        (data sources stay at their freshly constructed cursor).

    Raises
    ------
    ValueError
        If the payload version is unsupported, a task is missing, or a data
        source rejects the restored configuration.
    """
    if progress is None:
        return
    version = int(progress.get("version", -1))
    if version != DATA_PROGRESS_VERSION:
        raise ValueError(
            f"Unsupported data-progress checkpoint version {version}; "
            f"expected {DATA_PROGRESS_VERSION}."
        )
    tasks = progress.get("tasks", {})
    if not isinstance(tasks, Mapping):
        raise ValueError("data-progress 'tasks' must be a mapping.")
    for task_key, data in training_data_by_task.items():
        load_state_dict = getattr(data, "load_state_dict", None)
        if not callable(load_state_dict):
            continue
        if task_key not in tasks:
            raise ValueError(f"data-progress checkpoint is missing task {task_key!r}.")
        load_state_dict(tasks[task_key])
    if "rng" in progress:
        dp_random.set_state(progress["rng"])
