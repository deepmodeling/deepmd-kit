# SPDX-License-Identifier: LGPL-3.0-or-later
"""Composable observation hooks for the common training loop.

Observers sit outside the optimizer step and outside any compiled model
graph. They receive rank context and already-produced step or display
payloads; each observer decides whether a given boundary is worth acting
on. The training loop never imports logging backends on their behalf.
"""

from __future__ import (
    annotations,
)

from abc import (
    ABC,
)
from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
        Sequence,
    )

    from .timing import (
        DisplayInterval,
    )
    from .trainer import (
        RankContext,
        TrainingTaskCollection,
        TrainStepResult,
    )

__all__ = [
    "CheckpointObservation",
    "DisplayObservation",
    "StepObservation",
    "TrainingObserver",
    "TrainingObserverList",
]


@dataclass(frozen=True)
class StepObservation:
    """Payload delivered after one optimizer step.

    Attributes
    ----------
    step:
        Zero-based index of the step that just finished.
    display_step:
        One-based index used for logging and checkpoint cadence.
    task_key:
        Task selected for the step.
    learning_rate:
        Learning rate associated with ``step``.
    step_result:
        Backend payload returned by :meth:`AbstractTrainer.train_step`.
    rank_context:
        Rank metadata for the calling process.
    """

    step: int
    display_step: int
    task_key: str
    learning_rate: float
    step_result: TrainStepResult
    rank_context: RankContext


@dataclass(frozen=True)
class DisplayObservation:
    """Payload delivered at a learning-curve display boundary.

    Attributes
    ----------
    step:
        Zero-based index of the step that just finished.
    display_step:
        One-based index used for logging.
    task_key:
        Task selected for the step.
    learning_rate:
        Learning rate associated with ``step``.
    train_results:
        Training metrics prepared for the learning curve.
    valid_results:
        Validation metrics prepared for the learning curve, if any.
    timing:
        Wall-clock summary of the interval ending at ``display_step``.
    rank_context:
        Rank metadata for the calling process.
    """

    step: int
    display_step: int
    task_key: str
    learning_rate: float
    train_results: Any
    valid_results: Any | None
    timing: DisplayInterval
    rank_context: RankContext


@dataclass(frozen=True)
class CheckpointObservation:
    """Payload delivered when a checkpoint boundary is reached."""

    display_step: int
    rank_context: RankContext


class TrainingObserver(ABC):
    """Optional side-effect hooks around the common training loop.

    Default implementations are no-ops so an observer can override only the
    boundaries it cares about. Implementations must keep work outside
    compiled model execution and must not force device-to-host traffic when
    they choose not to emit.
    """

    def wants_step(self, display_step: int) -> bool:
        """Return whether :meth:`on_step_end` should run for this step."""
        return False

    def on_train_begin(
        self,
        tasks: TrainingTaskCollection,
        *,
        rank_context: RankContext,
    ) -> None:
        """Called once before the first optimizer step."""
        return None

    def on_step_end(self, observation: StepObservation) -> None:
        """Called after an optimizer step when :meth:`wants_step` is true."""
        return None

    def on_display(self, observation: DisplayObservation) -> None:
        """Called after learning-curve metrics are collected on the chief."""
        return None

    def on_checkpoint(self, observation: CheckpointObservation) -> None:
        """Called after a checkpoint boundary has been handled."""
        return None

    def on_train_end(
        self,
        tasks: TrainingTaskCollection,
        *,
        rank_context: RankContext,
    ) -> None:
        """Called after the run finishes or aborts, once resources close."""
        return None


class TrainingObserverList(TrainingObserver):
    """Fan-out wrapper that preserves observer order."""

    def __init__(self, observers: Sequence[TrainingObserver] | None = None) -> None:
        self._observers = tuple(observer for observer in (observers or ()) if observer)

    def __bool__(self) -> bool:
        return bool(self._observers)

    def __iter__(self) -> Iterator[TrainingObserver]:
        return iter(self._observers)

    def wants_step(self, display_step: int) -> bool:
        return any(observer.wants_step(display_step) for observer in self._observers)

    def on_train_begin(
        self,
        tasks: TrainingTaskCollection,
        *,
        rank_context: RankContext,
    ) -> None:
        for observer in self._observers:
            observer.on_train_begin(tasks, rank_context=rank_context)

    def on_step_end(self, observation: StepObservation) -> None:
        for observer in self._observers:
            if observer.wants_step(observation.display_step):
                observer.on_step_end(observation)

    def on_display(self, observation: DisplayObservation) -> None:
        for observer in self._observers:
            observer.on_display(observation)

    def on_checkpoint(self, observation: CheckpointObservation) -> None:
        for observer in self._observers:
            observer.on_checkpoint(observation)

    def on_train_end(
        self,
        tasks: TrainingTaskCollection,
        *,
        rank_context: RankContext,
    ) -> None:
        for observer in self._observers:
            observer.on_train_end(tasks, rank_context=rank_context)
