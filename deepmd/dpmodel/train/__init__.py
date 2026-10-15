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
from .data_progress import (
    DATA_PROGRESS_CHECKPOINT_KEY,
    collect_training_data_progress,
    restore_training_data_progress,
)
from .entrypoint import (
    AbstractTrainEntrypoint,
    TrainEntrypointOptions,
)
from .metrics import (
    TrainingMetricAccumulator,
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
    DEFAULT_TASK_KEY,
    AbstractTrainer,
    LearningCurveWriter,
    RankContext,
    TrainerConfig,
    TrainingTask,
    TrainingTaskCollection,
    TrainStepResult,
    change_model_out_bias,
    change_model_out_bias_by_task,
)

__all__ = [
    "DATA_PROGRESS_CHECKPOINT_KEY",
    "DEFAULT_TASK_KEY",
    "AbstractTrainEntrypoint",
    "AbstractTrainer",
    "CheckpointStore",
    "DisplayInterval",
    "LearningCurveWriter",
    "RankContext",
    "ShardingPolicy",
    "StepSchedule",
    "TrainEntrypointOptions",
    "TrainStepResult",
    "TrainerConfig",
    "TrainingMetricAccumulator",
    "TrainingTask",
    "TrainingTaskCollection",
    "TrainingTaskConfig",
    "TrainingTimer",
    "build_checkpoint_stores",
    "change_model_out_bias",
    "change_model_out_bias_by_task",
    "collect_training_data_progress",
    "iter_training_task_configs",
    "make_task_maps",
    "print_data_summaries",
    "resolve_keep_ckpt_count",
    "resolve_step_schedule",
    "restore_training_data_progress",
]
