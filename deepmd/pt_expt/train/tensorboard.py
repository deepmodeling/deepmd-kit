# SPDX-License-Identifier: LGPL-3.0-or-later
"""Rank-aware TensorBoard observation for pt_expt training.

The observer owns the SummaryWriter lifecycle. Only the chief rank creates a
writer; non-chief ranks and disabled runs perform no import, filesystem, or
metric host sync work. Legacy PT may construct the same observer.
"""

from __future__ import (
    annotations,
)

import logging
from collections.abc import (
    Mapping,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import torch

from deepmd.dpmodel.train.observer import (
    CheckpointObservation,
    DisplayObservation,
    StepObservation,
    TrainingObserver,
)
from deepmd.dpmodel.train.trainer import (
    DEFAULT_TASK_KEY,
    RankContext,
    TrainingTaskCollection,
)

log = logging.getLogger(__name__)

__all__ = [
    "TensorBoardObserver",
    "create_tensorboard_observer",
]


def create_tensorboard_observer(
    training_params: Mapping[str, Any],
    *,
    rank_context: RankContext,
    multi_task: bool = False,
) -> TensorBoardObserver | None:
    """Return a TensorBoard observer when enabled; otherwise ``None``.

    A disabled configuration returns ``None`` so the trainer installs no
    observer and never imports TensorBoard.
    """
    if not bool(training_params.get("tensorboard", False)):
        return None
    return TensorBoardObserver(
        log_dir=str(training_params.get("tensorboard_log_dir", "log")),
        freq=int(training_params.get("tensorboard_freq", 1)),
        rank_context=rank_context,
        multi_task=multi_task,
    )


class TensorBoardObserver(TrainingObserver):
    """Emit training scalars through a chief-owned SummaryWriter."""

    def __init__(
        self,
        *,
        log_dir: str,
        freq: int,
        rank_context: RankContext,
        multi_task: bool = False,
        writer_factory: Any | None = None,
    ) -> None:
        self._log_dir = str(log_dir)
        self._freq = int(freq)
        self._rank_context = rank_context
        self._multi_task = bool(multi_task)
        self._writer_factory = writer_factory
        self._writer: Any | None = None
        self._closed = False

    @property
    def enabled(self) -> bool:
        """Whether this rank would create a writer."""
        return self._freq > 0 and self._rank_context.is_chief

    @property
    def log_dir(self) -> str:
        return self._log_dir

    @property
    def writer(self) -> Any | None:
        return self._writer

    def wants_step(self, display_step: int) -> bool:
        return self.enabled and self._is_due(display_step)

    def on_train_begin(
        self,
        tasks: TrainingTaskCollection,
        *,
        rank_context: RankContext,
    ) -> None:
        if not self.enabled:
            return
        writer_factory = self._writer_factory
        if writer_factory is None:
            from torch.utils.tensorboard import (
                SummaryWriter,
            )

            writer_factory = SummaryWriter

        Path(self._log_dir).mkdir(parents=True, exist_ok=True)
        # A restart keeps existing event files; TensorBoard merges by step, so
        # continuing with restored global steps does not overwrite history.
        self._writer = writer_factory(log_dir=self._log_dir)
        self._closed = False
        log.info("TensorBoard events will be written under %s", self._log_dir)

    def on_step_end(self, observation: StepObservation) -> None:
        writer = self._writer
        if writer is None or not self._is_due(observation.display_step):
            return
        writer.add_scalar(
            "learning_rate",
            float(observation.learning_rate),
            observation.display_step,
        )
        self._write_train_from_step(observation)

    def on_display(self, observation: DisplayObservation) -> None:
        writer = self._writer
        if writer is None or not self._is_due(observation.display_step):
            return
        # Validation and timing exist only at display boundaries. Emit them
        # when that boundary also lands on tensorboard_freq so every tag
        # family shares the configured cadence.
        writer.add_scalar(
            "learning_rate",
            float(observation.learning_rate),
            observation.display_step,
        )
        self._write_metric_tree(
            "train",
            observation.train_results,
            observation.display_step,
        )
        self._write_metric_tree(
            "valid",
            observation.valid_results,
            observation.display_step,
        )
        timing = observation.timing
        writer.add_scalar(
            "timing/interval_wall_time",
            float(timing.wall_time),
            observation.display_step,
        )
        writer.add_scalar(
            "timing/interval_steps",
            float(timing.steps),
            observation.display_step,
        )
        if timing.eta is not None:
            writer.add_scalar(
                "timing/eta",
                float(timing.eta),
                observation.display_step,
            )

    def on_checkpoint(self, observation: CheckpointObservation) -> None:
        writer = self._writer
        if writer is not None:
            writer.flush()

    def on_train_end(
        self,
        tasks: TrainingTaskCollection,
        *,
        rank_context: RankContext,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Flush and close the writer if this rank owns one."""
        writer = self._writer
        if writer is None or self._closed:
            self._writer = None
            self._closed = True
            return
        try:
            writer.flush()
            writer.close()
        finally:
            self._writer = None
            self._closed = True

    def _is_due(self, display_step: int) -> bool:
        if self._freq <= 0:
            return False
        return display_step == 1 or display_step % self._freq == 0

    def _write_train_from_step(self, observation: StepObservation) -> None:
        step_result = observation.step_result
        metrics = step_result.train_results
        if metrics is None and isinstance(step_result.payload, Mapping):
            more_loss = step_result.payload.get("more_loss")
            if isinstance(more_loss, Mapping):
                metrics = {
                    key: value
                    for key, value in more_loss.items()
                    if "l2_" not in key
                }
        if not metrics:
            return
        if self._multi_task:
            self._write_metric_mapping(
                f"train/{observation.task_key}",
                metrics,
                observation.display_step,
            )
        else:
            self._write_metric_mapping("train", metrics, observation.display_step)

    def _write_metric_tree(
        self,
        prefix: str,
        results: Any,
        display_step: int,
    ) -> None:
        if results is None:
            return
        if self._is_task_results(results):
            for task_key, task_metrics in results.items():
                if not task_metrics:
                    continue
                self._write_metric_mapping(
                    f"{prefix}/{task_key}",
                    task_metrics,
                    display_step,
                )
            return
        if isinstance(results, Mapping):
            tag_prefix = (
                f"{prefix}/{DEFAULT_TASK_KEY}" if self._multi_task else prefix
            )
            self._write_metric_mapping(tag_prefix, results, display_step)

    def _write_metric_mapping(
        self,
        prefix: str,
        metrics: Mapping[str, Any],
        display_step: int,
    ) -> None:
        writer = self._writer
        if writer is None:
            return
        for key, value in metrics.items():
            scalar = _as_float(value)
            if scalar is None:
                continue
            writer.add_scalar(f"{prefix}/{key}", scalar, display_step)

    @staticmethod
    def _is_task_results(results: Any) -> bool:
        if not isinstance(results, Mapping) or not results:
            return False
        first = next(iter(results.values()))
        return first is None or isinstance(first, Mapping)


def _as_float(value: Any) -> float | None:
    """Convert a metric to float only when an event is being written."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
