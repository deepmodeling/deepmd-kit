# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent training abstractions."""

from .checkpoint import (
    CheckpointStore,
    build_checkpoint_stores,
    resolve_keep_ckpt_count,
)
from .data import (
    TrainingTaskConfig,
    iter_training_task_configs,
    make_task_maps,
    print_data_summaries,
)
from .entrypoint import (
    AbstractTrainEntrypoint,
    TrainEntrypointOptions,
)
from .metrics import (
    TrainingMetricAccumulator,
)
from .observer import (
    CheckpointObservation,
    DisplayObservation,
    StepObservation,
    TrainingObserver,
    TrainingObserverList,
)
from .schedule import (
    StepSchedule,
    resolve_step_schedule,
)
from .sharding import (
    ShardingPolicy,
)
from .timing import (
    DisplayInterval,
    TrainingTimer,
)
from .trainer import (
    AbstractTrainer,
    LearningCurveWriter,
    TrainerConfig,
    change_model_out_bias,
    change_model_out_bias_by_task,
)
from .types import (
    DEFAULT_TASK_KEY,
    RankContext,
    TrainingTask,
    TrainingTaskCollection,
    TrainStepResult,
)

__all__ = [
    "DEFAULT_TASK_KEY",
    "AbstractTrainEntrypoint",
    "AbstractTrainer",
    "CheckpointObservation",
    "CheckpointStore",
    "DisplayInterval",
    "DisplayObservation",
    "LearningCurveWriter",
    "RankContext",
    "ShardingPolicy",
    "StepObservation",
    "StepSchedule",
    "TrainEntrypointOptions",
    "TrainStepResult",
    "TrainerConfig",
    "TrainingMetricAccumulator",
    "TrainingObserver",
    "TrainingObserverList",
    "TrainingTask",
    "TrainingTaskCollection",
    "TrainingTaskConfig",
    "TrainingTimer",
    "build_checkpoint_stores",
    "change_model_out_bias",
    "change_model_out_bias_by_task",
    "iter_training_task_configs",
    "make_task_maps",
    "print_data_summaries",
    "resolve_keep_ckpt_count",
    "resolve_step_schedule",
]
