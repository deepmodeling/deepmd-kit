# SPDX-License-Identifier: LGPL-3.0-or-later
"""Small, backend-independent checkpoints of the next training batches.

Data sources own their cursors and deterministic sampler state. This module
adds task names and the shared selection RNG, not decoded data or permutations.
"""

from collections.abc import (
    Iterator,
    Mapping,
)
from contextlib import (
    contextmanager,
)
from typing import (
    Any,
)

from deepmd.utils import random as dp_random

DATA_PROGRESS_CHECKPOINT_KEY = "data_progress"
DATA_PROGRESS_VERSION = 1


def collect_training_data_progress(
    training_data_by_task: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect supported data-source cursors and the shared selection RNG."""
    return {
        "version": DATA_PROGRESS_VERSION,
        "tasks": {
            key: data.state_dict()
            for key, data in training_data_by_task.items()
            if callable(getattr(data, "state_dict", None))
            and callable(getattr(data, "load_state_dict", None))
        },
        "rng": dp_random.get_state(),
    }


def restore_training_data_progress(
    training_data_by_task: Mapping[str, Any],
    progress: Mapping[str, Any] | None,
) -> None:
    """Restore cursors lazily; missing progress preserves legacy restarts.

    The source implementations validate their sampling configuration and must
    not read or decode data here. Restoring the shared RNG last also isolates
    it from any source-specific reconstruction.
    """
    if progress is None:
        return
    if progress.get("version") != DATA_PROGRESS_VERSION:
        raise ValueError("Unsupported data-progress checkpoint version.")
    tasks = progress.get("tasks")
    if not isinstance(tasks, Mapping):
        raise ValueError("data-progress 'tasks' must be a mapping.")
    supported = {
        key: data
        for key, data in training_data_by_task.items()
        if callable(getattr(data, "load_state_dict", None))
        and callable(getattr(data, "state_dict", None))
    }
    if set(tasks) != set(supported):
        raise ValueError("data-progress tasks do not match the training data sources.")
    if "rng" not in progress:
        raise ValueError("data-progress checkpoint is missing the selection RNG.")
    for key, data in supported.items():
        data.load_state_dict(tasks[key])
    dp_random.set_state(progress["rng"])


@contextmanager
def preserve_training_data_progress(
    training_data_by_task: Mapping[str, Any],
) -> Iterator[None]:
    """Keep diagnostics and compilation from consuming the training stream."""
    progress = collect_training_data_progress(training_data_by_task)
    try:
        yield
    finally:
        restore_training_data_progress(training_data_by_task, progress)
