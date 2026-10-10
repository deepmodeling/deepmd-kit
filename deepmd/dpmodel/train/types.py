# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared training-loop types with no observer or trainer dependency.

These types are the vocabulary that observers annotate against and that the
common trainer owns at runtime. Keeping them in a leaf module breaks the
observer ↔ trainer import cycle that CodeQL flags: observers import here for
annotations; the trainer imports observers for dispatch and re-exports these
names for backward-compatible ``from .trainer import ...`` paths.
"""

from __future__ import (
    annotations,
)

from collections.abc import (
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

import numpy as np

DEFAULT_TASK_KEY = "Default"

__all__ = [
    "DEFAULT_TASK_KEY",
    "RankContext",
    "TrainingTask",
    "TrainingTaskCollection",
    "TrainStepResult",
]


@dataclass(frozen=True)
class RankContext:
    """Rank metadata used by a trainer.

    A single-process run is represented as rank 0 in a world of size 1.  This
    makes single-rank training a special case of multi-rank training.
    """

    rank: int = 0
    world_size: int = 1

    @property
    def is_chief(self) -> bool:
        """Whether this rank is responsible for user-visible side effects."""
        return self.rank == 0


@dataclass
class TrainingTask:
    """One training task.

    Single-task training is represented by a collection containing one task.
    """

    key: str
    training_data: Any
    validation_data: Any | None = None
    valid_numb_batch: int = 1
    data_requirements: list[Any] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.valid_numb_batch = max(int(self.valid_numb_batch), 1)

    def add_data_requirements(self) -> None:
        """Attach data requirements to train and validation data if possible."""
        if not self.data_requirements:
            return
        for data in (self.training_data, self.validation_data):
            if data is not None and hasattr(data, "add_data_requirements"):
                data.add_data_requirements(self.data_requirements)


class TrainingTaskCollection:
    """Ordered collection of training tasks with optional sampling weights."""

    def __init__(
        self,
        tasks: Mapping[str, TrainingTask] | Sequence[TrainingTask],
        probabilities: Mapping[str, float] | Sequence[float] | None = None,
    ) -> None:
        if isinstance(tasks, Mapping):
            task_dict = dict(tasks)
        else:
            task_list = list(tasks)
            task_dict = {task.key: task for task in task_list}
            if len(task_dict) != len(task_list):
                raise ValueError("Training task keys must be unique.")
        if not task_dict:
            raise ValueError("At least one training task is required.")
        for key, task in task_dict.items():
            if key != task.key:
                raise ValueError(
                    f"Task mapping key {key!r} does not match task key {task.key!r}."
                )
        self._tasks = task_dict
        self._keys = list(task_dict)
        self._probabilities = self._normalize_probabilities(probabilities)

    @classmethod
    def single(
        cls,
        training_data: Any,
        validation_data: Any | None = None,
        *,
        key: str = DEFAULT_TASK_KEY,
        valid_numb_batch: int = 1,
        data_requirements: list[Any] | None = None,
    ) -> TrainingTaskCollection:
        """Build a task collection for single-task training."""
        task = TrainingTask(
            key=key,
            training_data=training_data,
            validation_data=validation_data,
            valid_numb_batch=valid_numb_batch,
            data_requirements=list(data_requirements or []),
        )
        return cls([task])

    @property
    def keys(self) -> list[str]:
        """Task keys in iteration order."""
        return list(self._keys)

    @property
    def probabilities(self) -> np.ndarray:
        """Normalized task sampling probabilities."""
        return self._probabilities.copy()

    @property
    def is_multitask(self) -> bool:
        """Whether more than one task is present."""
        return len(self._tasks) > 1

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[TrainingTask]:
        for key in self._keys:
            yield self._tasks[key]

    def __getitem__(self, key: str) -> TrainingTask:
        return self._tasks[key]

    def select(
        self,
        choice: Callable[..., Any] | None = None,
    ) -> TrainingTask:
        """Select a task according to the configured probabilities."""
        if len(self._keys) == 1:
            return self._tasks[self._keys[0]]
        chooser = choice or np.random.choice
        index = int(
            chooser(np.arange(len(self._keys), dtype=np.int_), p=self._probabilities)
        )
        return self._tasks[self._keys[index]]

    def _normalize_probabilities(
        self,
        probabilities: Mapping[str, float] | Sequence[float] | None,
    ) -> np.ndarray:
        if probabilities is None:
            prob = np.ones(len(self._keys), dtype=np.float64)
        elif isinstance(probabilities, Mapping):
            missing = [key for key in self._keys if key not in probabilities]
            if missing:
                raise ValueError(f"Missing task probabilities for {missing}.")
            unknown = [key for key in probabilities if key not in self._tasks]
            if unknown:
                raise ValueError(f"Unknown task probabilities for {unknown}.")
            prob = np.asarray(
                [probabilities[key] for key in self._keys], dtype=np.float64
            )
        else:
            prob = np.asarray(probabilities, dtype=np.float64)
        if prob.ndim != 1 or prob.shape[0] != len(self._keys):
            raise ValueError("Task probabilities must match the number of tasks.")
        if not np.all(np.isfinite(prob)):
            raise ValueError("Task probabilities must be finite.")
        if np.any(prob < 0.0):
            raise ValueError("Task probabilities must be non-negative.")
        prob_sum = float(np.sum(prob))
        if prob_sum <= 0.0:
            raise ValueError("Task probabilities must sum to a positive value.")
        return prob / prob_sum


@dataclass
class TrainStepResult:
    """Backend payload returned from one optimizer step.

    Averaged display consumes ``train_results`` as detached scalar metrics.
    These may remain backend arrays until they are converted for display.
    """

    task_key: str
    step: int
    payload: Any = None
    train_results: Mapping[str, Any] | None = None
