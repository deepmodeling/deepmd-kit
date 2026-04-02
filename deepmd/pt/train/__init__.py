# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch training module with modular, extensible design.

This module provides a clean, component-based training system:

- TrainingConfig: Configuration management with validation
- DataManager: Data loading and batch iteration
- OptimizerFactory: Strategy pattern for optimizer creation
- CheckpointManager: Model persistence and recovery
- TrainingLoop: Specialized training step implementations
- HookManager: Extensible callback system
- TrainingLogger: Formatted output and file I/O
- Trainer: Main orchestrator coordinating all components

Example:
    >>> from deepmd.pt.train import Trainer, TrainingConfig
    >>>
    >>> # Create trainer
    >>> trainer = Trainer(
    ...     config=config_dict,
    ...     training_data=train_dataset,
    ...     validation_data=valid_dataset,
    ... )
    >>>
    >>> # Run training
    >>> trainer.run()

Future extensions for multi-backend support:
- AbstractTrainingLoop can be extended for JAX/NumPy
- OptimizerFactory can support backend-specific optimizers
- DataManager can use backend-specific data loading
"""

from deepmd.pt.train.checkpoint_manager import (
    CheckpointManager,
)
from deepmd.pt.train.config import (
    CheckpointConfig,
    DisplayConfig,
    LearningRateConfig,
    OptimizerConfig,
    TrainingConfig,
)
from deepmd.pt.train.data_manager import (
    DataManager,
)
from deepmd.pt.train.hooks import (
    HookManager,
    HookPriority,
    TensorBoardHook,
    TimingHook,
    TrainingHook,
)
from deepmd.pt.train.logger import (
    LossAccumulator,
    TrainingLogger,
)
from deepmd.pt.train.optimizer_factory import (
    OptimizerFactory,
)
from deepmd.pt.train.trainer import (
    Trainer,
)

# Keep old Trainer available for backward compatibility during transition
from deepmd.pt.train.training import Trainer as LegacyTrainer
from deepmd.pt.train.training_loop import (
    AdamTrainingLoop,
    BaseTrainingLoop,
    LKFEnergyTrainingLoop,
    TrainingLoopFactory,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)

__all__ = [
    # New modular components
    "AdamTrainingLoop",
    "BaseTrainingLoop",
    "CheckpointConfig",
    "CheckpointManager",
    "DataManager",
    "DisplayConfig",
    "HookManager",
    "HookPriority",
    "LKFEnergyTrainingLoop",
    "LearningRateConfig",
    # Legacy support
    "LegacyTrainer",
    "LossAccumulator",
    "ModelWrapper",
    "OptimizerConfig",
    "OptimizerFactory",
    "TensorBoardHook",
    "TimingHook",
    "Trainer",
    "TrainingConfig",
    "TrainingHook",
    "TrainingLogger",
    "TrainingLoopFactory",
]
