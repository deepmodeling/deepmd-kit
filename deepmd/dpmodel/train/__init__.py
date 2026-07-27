# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-independent training abstractions."""

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
    "DEFAULT_TASK_KEY",
    "AbstractTrainEntrypoint",
    "AbstractTrainer",
    "LearningCurveWriter",
    "RankContext",
    "TrainEntrypointOptions",
    "TrainStepResult",
    "TrainerConfig",
    "TrainingTask",
    "TrainingTaskCollection",
    "TrainingTaskConfig",
    "change_model_out_bias",
    "change_model_out_bias_by_task",
    "iter_training_task_configs",
    "make_task_maps",
    "print_data_summaries",
]
