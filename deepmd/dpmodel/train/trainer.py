# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent training driver abstractions.

The classes in this module intentionally know nothing about TensorFlow,
PyTorch, JAX, or Paddle tensor semantics.  Backend trainers provide the
numerical hooks; this layer owns task/rank normalization, display scheduling,
learning-curve output, and checkpoint cadence.
"""

from __future__ import (
    annotations,
)

import datetime
import logging
import time
from abc import (
    ABC,
    abstractmethod,
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
from pathlib import (
    Path,
)
from typing import (
    Any,
    TextIO,
)

import numpy as np

from deepmd.loggers.training import (
    format_training_message,
    format_training_message_per_task,
)

DEFAULT_TASK_KEY = "Default"

log = logging.getLogger(__name__)

LossResults = dict[str, float]
TaskResults = dict[str, LossResults | None]
DisplayResults = LossResults | TaskResults


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


@dataclass(frozen=True)
class TrainerConfig:
    """Common trainer configuration shared by backend implementations."""

    num_steps: int
    start_step: int = 0
    disp_file: str = "lcurve.out"
    disp_freq: int = 1000
    save_ckpt: str = "model.ckpt"
    save_freq: int = 1000
    max_ckpt_keep: int = 5
    display_in_training: bool = True
    timing_in_training: bool = True
    restart_training: bool = False

    @classmethod
    def from_training_params(
        cls,
        training_params: Mapping[str, Any],
        *,
        num_steps: int | None = None,
        start_step: int = 0,
        restart_training: bool = False,
    ) -> TrainerConfig:
        """Create common trainer config from a normalized training section."""
        return cls(
            num_steps=(
                int(num_steps)
                if num_steps is not None
                else int(training_params["numb_steps"])
            ),
            start_step=int(start_step),
            disp_file=str(training_params.get("disp_file", "lcurve.out")),
            disp_freq=int(training_params.get("disp_freq", 1000)),
            save_ckpt=str(training_params.get("save_ckpt", "model.ckpt")),
            save_freq=int(training_params.get("save_freq", 1000)),
            max_ckpt_keep=int(training_params.get("max_ckpt_keep", 5)),
            display_in_training=bool(training_params.get("disp_training", True)),
            timing_in_training=bool(training_params.get("time_training", True)),
            restart_training=restart_training,
        )


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
    """Backend payload returned from one optimizer step."""

    task_key: str
    step: int
    payload: Any = None
    train_results: LossResults | None = None


class LearningCurveWriter:
    """Formatter for learning-curve files and per-task training logs."""

    def __init__(self, task_keys: Sequence[str] | None = None) -> None:
        self.task_keys = list(task_keys or [])

    def write_header(
        self,
        fp: TextIO,
        train_results: DisplayResults,
        valid_results: DisplayResults | None,
    ) -> None:
        """Write a learning-curve header."""
        fp.write(self.format_header(train_results, valid_results))
        fp.flush()

    def write_row(
        self,
        fp: TextIO,
        *,
        step: int,
        learning_rate: float,
        train_results: DisplayResults,
        valid_results: DisplayResults | None,
    ) -> None:
        """Write one learning-curve row."""
        fp.write(
            self.format_row(
                step=step,
                learning_rate=learning_rate,
                train_results=train_results,
                valid_results=valid_results,
            )
        )
        fp.flush()

    def log_results(
        self,
        *,
        step: int,
        learning_rate: float,
        train_results: DisplayResults,
        valid_results: DisplayResults | None,
    ) -> None:
        """Log per-task loss results."""
        if self._is_multitask(train_results):
            task_results = train_results
            valid_task_results = (
                valid_results if isinstance(valid_results, Mapping) else {}
            )
            assert isinstance(task_results, Mapping)
            for task_key in self._ordered_task_keys(task_results):
                task_train = task_results.get(task_key)
                if task_train is None:
                    continue
                log.info(
                    format_training_message_per_task(
                        batch=step,
                        task_name=f"{task_key}_trn",
                        rmse=task_train,
                        learning_rate=learning_rate,
                    )
                )
                task_valid = valid_task_results.get(task_key)
                if task_valid:
                    log.info(
                        format_training_message_per_task(
                            batch=step,
                            task_name=f"{task_key}_val",
                            rmse=task_valid,
                            learning_rate=None,
                        )
                    )
        else:
            assert not self._is_multitask(train_results)
            log.info(
                format_training_message_per_task(
                    batch=step,
                    task_name="trn",
                    rmse=train_results,
                    learning_rate=learning_rate,
                )
            )
            if valid_results:
                assert not self._is_multitask(valid_results)
                log.info(
                    format_training_message_per_task(
                        batch=step,
                        task_name="val",
                        rmse=valid_results,
                        learning_rate=None,
                    )
                )

    def format_header(
        self,
        train_results: DisplayResults,
        valid_results: DisplayResults | None,
    ) -> str:
        """Format a learning-curve header."""
        header = "# {:5s}".format("step")
        if self._is_multitask(train_results):
            assert isinstance(train_results, Mapping)
            valid_task_results = (
                valid_results if isinstance(valid_results, Mapping) else {}
            )
            for task_key in self._ordered_task_keys(train_results):
                task_train = train_results.get(task_key)
                if not task_train:
                    continue
                task_valid = valid_task_results.get(task_key)
                if task_valid:
                    for key in task_train:
                        header += (
                            f"   {key + '_val_' + task_key:>11s}"
                            f" {key + '_trn_' + task_key:>11s}"
                        )
                else:
                    for key in task_train:
                        header += f"   {key + '_trn_' + task_key:>11s}"
        else:
            assert not self._is_multitask(train_results)
            if valid_results is not None:
                assert not self._is_multitask(valid_results)
                for key in train_results:
                    header += f"   {key + '_val':>11s} {key + '_trn':>11s}"
            else:
                for key in train_results:
                    header += f"   {key + '_trn':>11s}"
        header += "   {:8s}\n".format("lr")
        header += "# If there is no available reference data, rmse_*_{val,trn} will print nan\n"
        return header

    def format_row(
        self,
        *,
        step: int,
        learning_rate: float,
        train_results: DisplayResults,
        valid_results: DisplayResults | None,
    ) -> str:
        """Format one learning-curve row."""
        row = f"{step:7d}"
        if self._is_multitask(train_results):
            assert isinstance(train_results, Mapping)
            valid_task_results = (
                valid_results if isinstance(valid_results, Mapping) else {}
            )
            for task_key in self._ordered_task_keys(train_results):
                task_train = train_results.get(task_key)
                if not task_train:
                    continue
                task_valid = valid_task_results.get(task_key)
                if task_valid:
                    for key in task_train:
                        row += (
                            f"   {float(task_valid.get(key, float('nan'))):11.2e}"
                            f" {float(task_train[key]):11.2e}"
                        )
                else:
                    for key in task_train:
                        row += f"   {float(task_train[key]):11.2e}"
        else:
            assert not self._is_multitask(train_results)
            if valid_results is not None:
                assert not self._is_multitask(valid_results)
                for key in train_results:
                    row += (
                        f"   {float(valid_results.get(key, float('nan'))):11.2e}"
                        f" {float(train_results[key]):11.2e}"
                    )
            else:
                for key in train_results:
                    row += f"   {float(train_results[key]):11.2e}"
        row += f"   {learning_rate:8.1e}\n"
        return row

    def _ordered_task_keys(self, results: Mapping[str, Any]) -> list[str]:
        keys = self.task_keys or list(results)
        return [key for key in keys if key in results]

    @staticmethod
    def _is_multitask(results: Any) -> bool:
        if not isinstance(results, Mapping) or not results:
            return False
        return all(
            isinstance(value, Mapping) or value is None for value in results.values()
        )


class AbstractTrainer(ABC):
    """Backend-independent trainer driver.

    Backend trainers implement one optimizer step, metric evaluation, learning
    rate lookup, and checkpoint persistence.  This base class handles the
    common training loop around those hooks.
    """

    def __init__(
        self,
        trainer_config: TrainerConfig,
        *,
        rank_context: RankContext | None = None,
    ) -> None:
        self.trainer_config = trainer_config
        self.rank_context = rank_context or RankContext()
        self.lcurve_writer = LearningCurveWriter()

    def run(self, tasks: TrainingTaskCollection) -> None:
        """Run the common training loop."""
        self.lcurve_writer = LearningCurveWriter(tasks.keys)
        start_step = self.trainer_config.start_step
        num_steps = self.trainer_config.num_steps
        fout: TextIO | None = None
        try:
            self.on_train_begin(tasks)
            fout = self._open_learning_curve()
            wall_start = time.time()
            last_log_time = wall_start
            last_log_step = start_step
            for step in range(start_step, num_steps):
                task = self.select_task(tasks)
                step_result = self.train_step(task, step)
                display_step = step + 1

                if self._should_display(display_step):
                    if self.rank_context.is_chief:
                        train_results, valid_results = self.collect_display_results(
                            tasks,
                            active_task=task,
                            step=step,
                            step_result=step_result,
                        )
                        current_time = time.time()
                        interval_wall_time = current_time - last_log_time
                        interval_steps = max(1, display_step - last_log_step)
                        self._log_interval(
                            display_step=display_step,
                            interval_wall_time=interval_wall_time,
                            interval_steps=interval_steps,
                            wall_elapsed=current_time - wall_start,
                        )
                        current_lr = self.learning_rate(step)
                        self.lcurve_writer.log_results(
                            step=display_step,
                            learning_rate=current_lr,
                            train_results=train_results,
                            valid_results=valid_results,
                        )
                        if fout is not None:
                            if fout.tell() == 0:
                                self.lcurve_writer.write_header(
                                    fout,
                                    train_results=train_results,
                                    valid_results=valid_results,
                                )
                            self.lcurve_writer.write_row(
                                fout,
                                step=display_step,
                                learning_rate=current_lr,
                                train_results=train_results,
                                valid_results=valid_results,
                            )
                        last_log_time = current_time
                        last_log_step = display_step

                self.run_full_validation(
                    step=step,
                    display_step=display_step,
                    learning_rate=self.learning_rate(step),
                )

                if (
                    self.rank_context.is_chief
                    and self.trainer_config.save_freq > 0
                    and display_step % self.trainer_config.save_freq == 0
                ):
                    self.save_checkpoint(display_step)

            if self._should_save_final_checkpoint():
                self.save_checkpoint(num_steps)
        finally:
            if fout is not None:
                fout.close()
            self.on_train_end(tasks)

    def select_task(self, tasks: TrainingTaskCollection) -> TrainingTask:
        """Select the task for the next optimizer step."""
        return tasks.select()

    def collect_display_results(
        self,
        tasks: TrainingTaskCollection,
        *,
        active_task: TrainingTask,
        step: int,
        step_result: TrainStepResult,
    ) -> tuple[DisplayResults, DisplayResults | None]:
        """Collect training and validation results for display."""
        if not tasks.is_multitask:
            return (
                self.evaluate_training(active_task, step, step_result),
                self.evaluate_validation(active_task, step, step_result),
            )

        train_results: TaskResults = {}
        valid_results: TaskResults = {}
        for task in tasks:
            task_step_result = step_result if task.key == active_task.key else None
            train_results[task.key] = self.evaluate_training(
                task,
                step,
                task_step_result,
            )
            valid_results[task.key] = self.evaluate_validation(
                task,
                step,
                task_step_result,
            )
        return train_results, valid_results

    def on_train_begin(self, tasks: TrainingTaskCollection) -> None:
        """Hook called before the first optimizer step."""
        return None

    def on_train_end(self, tasks: TrainingTaskCollection) -> None:
        """Hook called after training resources have been closed."""
        return None

    def run_full_validation(
        self,
        *,
        step: int,
        display_step: int,
        learning_rate: float,
    ) -> None:
        """Run optional backend-specific full validation for one step."""
        return None

    @abstractmethod
    def train_step(self, task: TrainingTask, step: int) -> TrainStepResult:
        """Run one backend-specific optimizer step."""

    @abstractmethod
    def evaluate_training(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> LossResults:
        """Evaluate training metrics for one task."""

    def evaluate_validation(
        self,
        task: TrainingTask,
        step: int,
        step_result: TrainStepResult | None,
    ) -> LossResults | None:
        """Evaluate validation metrics for one task."""
        return None

    @abstractmethod
    def learning_rate(self, step: int) -> float:
        """Return the learning rate associated with a zero-based step."""

    @abstractmethod
    def save_checkpoint(self, step: int) -> None:
        """Persist a checkpoint for a one-based step."""

    def _open_learning_curve(self) -> TextIO | None:
        if (
            not self.rank_context.is_chief
            or not self.trainer_config.display_in_training
        ):
            return None
        disp_path = Path(self.trainer_config.disp_file)
        append = (
            self.trainer_config.restart_training or self.trainer_config.start_step > 0
        ) and disp_path.exists()
        return open(disp_path, "a" if append else "w")

    def _should_display(self, display_step: int) -> bool:
        if not self.trainer_config.display_in_training:
            return False
        return display_step == 1 or (
            self.trainer_config.disp_freq > 0
            and display_step % self.trainer_config.disp_freq == 0
        )

    def _should_save_final_checkpoint(self) -> bool:
        if not self.rank_context.is_chief:
            return False
        if self.trainer_config.num_steps <= self.trainer_config.start_step:
            return False
        if self.trainer_config.save_freq <= 0:
            return True
        return self.trainer_config.num_steps % self.trainer_config.save_freq != 0

    def _log_interval(
        self,
        *,
        display_step: int,
        interval_wall_time: float,
        interval_steps: int,
        wall_elapsed: float,
    ) -> None:
        if self.trainer_config.timing_in_training:
            completed = max(1, display_step - self.trainer_config.start_step)
            eta = int(
                (self.trainer_config.num_steps - display_step)
                / completed
                * wall_elapsed
            )
            log.info(
                format_training_message(
                    batch=display_step,
                    wall_time=interval_wall_time,
                    eta=eta,
                    current_time=datetime.datetime.fromtimestamp(
                        time.time(),
                        tz=datetime.timezone.utc,
                    ).astimezone(),
                    step_time=interval_wall_time / interval_steps,
                )
            )
        else:
            log.info(
                format_training_message(
                    batch=display_step,
                    wall_time=interval_wall_time,
                )
            )
