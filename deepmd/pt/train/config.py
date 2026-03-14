# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training configuration management with validation and defaults.

This module defines dataclasses for training configuration. It works
in conjunction with deepmd/utils/argcheck.py which validates the
input configuration against a schema defined using dargs.

Configuration flow:
1. User provides input JSON/YAML
2. argcheck.py validates against schema (deepmd/utils/argcheck.py)
3. normalize() normalizes the configuration
4. This module converts the normalized dict to typed dataclasses

Default values here should match those in argcheck.py's Argument definitions
to ensure consistency between validation and runtime behavior.
"""

from __future__ import (
    annotations,
)

import logging
from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
)

log = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Optimizer configuration with type-specific parameters."""

    opt_type: str = "Adam"
    weight_decay: float = 0.001
    momentum: float = 0.95
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    lr_adjust: float = 10.0
    lr_adjust_coeff: float = 0.2
    muon_2d_only: bool = True
    min_2d_dim: int = 1
    kf_blocksize: int = 5120
    kf_start_pref_e: float = 1.0
    kf_limit_pref_e: float = 1.0
    kf_start_pref_f: float = 1.0
    kf_limit_pref_f: float = 1.0

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> OptimizerConfig:
        """Create OptimizerConfig from dictionary."""
        return cls(
            opt_type=params.get("opt_type", "Adam"),
            weight_decay=params.get("weight_decay", 0.001),
            momentum=params.get("momentum", 0.95),
            adam_beta1=params.get("adam_beta1", 0.9),
            adam_beta2=params.get("adam_beta2", 0.95),
            lr_adjust=params.get("lr_adjust", 10.0),
            lr_adjust_coeff=params.get("lr_adjust_coeff", 0.2),
            muon_2d_only=params.get("muon_2d_only", True),
            min_2d_dim=params.get("min_2d_dim", 1),
            kf_blocksize=params.get("kf_blocksize", 5120),
            kf_start_pref_e=params.get("kf_start_pref_e", 1.0),
            kf_limit_pref_e=params.get("kf_limit_pref_e", 1.0),
            kf_start_pref_f=params.get("kf_start_pref_f", 1.0),
            kf_limit_pref_f=params.get("kf_limit_pref_f", 1.0),
        )


@dataclass
class LearningRateConfig:
    """Learning rate schedule configuration."""

    start_lr: float = 1e-3
    stop_lr: float = 1e-8
    decay_steps: int = 100000
    decay_rate: float = 0.95
    stop_steps: int = 0

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> LearningRateConfig:
        """Create LearningRateConfig from dictionary."""
        return cls(
            start_lr=params.get("start_lr", 1e-3),
            stop_lr=params.get("stop_lr", 1e-8),
            decay_steps=params.get("decay_steps", 100000),
            decay_rate=params.get("decay_rate", 0.95),
        )


@dataclass
class DisplayConfig:
    """Training display and logging configuration.

    Default values match those in argcheck.py training_args().
    """

    disp_file: str = "lcurve.out"  # argcheck default: "lcurve.out"
    disp_freq: int = 1000  # argcheck default: 1000
    disp_avg: bool = False  # argcheck default: False (PyTorch only)
    disp_training: bool = True  # argcheck default: True
    time_training: bool = True  # argcheck default: True
    tensorboard: bool = False  # argcheck default: False
    tensorboard_log_dir: str = "log"  # argcheck default: "log"
    tensorboard_freq: int = 1  # argcheck default: 1
    enable_profiler: bool = False  # argcheck default: False
    profiling: bool = False  # argcheck default: False
    profiling_file: str = "timeline.json"  # argcheck default: "timeline.json"

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> DisplayConfig:
        """Create DisplayConfig from dictionary."""
        return cls(
            disp_file=params.get("disp_file", "lcurve.out"),
            disp_freq=params.get("disp_freq", 1000),
            disp_avg=params.get("disp_avg", False),
            disp_training=params.get("disp_training", True),
            time_training=params.get("time_training", True),
            tensorboard=params.get("tensorboard", False),
            tensorboard_log_dir=params.get("tensorboard_log_dir", "log"),
            tensorboard_freq=params.get("tensorboard_freq", 1),
            enable_profiler=params.get("enable_profiler", False),
            profiling=params.get("profiling", False),
            profiling_file=params.get("profiling_file", "timeline.json"),
        )


@dataclass
class CheckpointConfig:
    """Model checkpoint configuration.

    Default values match those in argcheck.py training_args().
    """

    save_ckpt: str = "model.ckpt"  # argcheck default: "model.ckpt"
    save_freq: int = 1000  # argcheck default: 1000
    max_ckpt_keep: int = 5  # argcheck default: 5
    change_bias_after_training: bool = False  # argcheck default: False

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> CheckpointConfig:
        """Create CheckpointConfig from dictionary."""
        return cls(
            save_ckpt=params.get("save_ckpt", "model.ckpt"),
            save_freq=params.get("save_freq", 1000),
            max_ckpt_keep=params.get("max_ckpt_keep", 5),
            change_bias_after_training=params.get("change_bias_after_training", False),
        )


@dataclass
class TrainingConfig:
    """Complete training configuration container."""

    num_steps: int = 0
    warmup_steps: int = 0
    warmup_start_factor: float = 0.0
    gradient_max_norm: float = 0.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    learning_rate: LearningRateConfig = field(default_factory=LearningRateConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    is_multitask: bool = False
    optimizer_dict: dict[str, OptimizerConfig] | None = None
    learning_rate_dict: dict[str, LearningRateConfig] | None = None

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        model_keys: list[str] | None = None,
    ) -> TrainingConfig:
        """Create TrainingConfig from a configuration dictionary."""
        training_params = config.get("training", {})

        num_steps = training_params.get("numb_steps", 0)
        if num_steps <= 0:
            raise ValueError(f"numb_steps must be positive, got {num_steps}")

        warmup_steps = training_params.get("warmup_steps", None)
        warmup_ratio = training_params.get("warmup_ratio", None)
        if warmup_steps is not None:
            computed_warmup_steps = warmup_steps
        elif warmup_ratio is not None:
            if not 0 <= warmup_ratio < 1:
                raise ValueError(f"warmup_ratio must be in [0, 1), got {warmup_ratio}")
            computed_warmup_steps = int(warmup_ratio * num_steps)
            if computed_warmup_steps == 0 and warmup_ratio > 0:
                log.warning(
                    f"warmup_ratio {warmup_ratio} results in 0 warmup steps. "
                    "Consider using a larger ratio or specify warmup_steps directly."
                )
        else:
            computed_warmup_steps = 0

        assert num_steps - computed_warmup_steps > 0 or computed_warmup_steps == 0, (
            "Warm up steps must be less than total training steps!"
        )

        is_multitask = model_keys is not None and len(model_keys) > 1

        if is_multitask and training_params.get("optim_dict") is not None:
            optim_dict = {
                key: OptimizerConfig.from_dict(training_params["optim_dict"][key])
                for key in model_keys
                if key in training_params["optim_dict"]
            }
            missing_keys = [key for key in model_keys if key not in optim_dict]
            if missing_keys:
                raise ValueError(f"Missing optimizer config for keys: {missing_keys}")
            optimizer = optim_dict[model_keys[0]]
        else:
            optim_dict = None
            optimizer = OptimizerConfig.from_dict(training_params)

        lr_params = config.get("learning_rate", {})
        if is_multitask and config.get("learning_rate_dict") is not None:
            lr_dict = {
                key: LearningRateConfig.from_dict(config["learning_rate_dict"][key])
                for key in model_keys
                if key in config["learning_rate_dict"]
            }
            learning_rate = lr_dict.get(
                model_keys[0], LearningRateConfig.from_dict(lr_params)
            )
        else:
            lr_dict = None
            learning_rate = LearningRateConfig.from_dict(lr_params)

        learning_rate.stop_steps = num_steps - computed_warmup_steps
        if lr_dict:
            for lr_config in lr_dict.values():
                lr_config.stop_steps = num_steps - computed_warmup_steps

        return cls(
            num_steps=num_steps,
            warmup_steps=computed_warmup_steps,
            warmup_start_factor=training_params.get("warmup_start_factor", 0.0),
            gradient_max_norm=training_params.get("gradient_max_norm", 0.0),
            optimizer=optimizer,
            learning_rate=learning_rate,
            display=DisplayConfig.from_dict(training_params),
            checkpoint=CheckpointConfig.from_dict(training_params),
            is_multitask=is_multitask,
            optimizer_dict=optim_dict,
            learning_rate_dict=lr_dict,
        )

    def get_optimizer_config(self, task_key: str = "Default") -> OptimizerConfig:
        """Get optimizer config for a specific task."""
        if self.is_multitask and self.optimizer_dict is not None:
            return self.optimizer_dict.get(task_key, self.optimizer)
        return self.optimizer

    def get_lr_config(self, task_key: str = "Default") -> LearningRateConfig:
        """Get learning rate config for a specific task."""
        if self.is_multitask and self.learning_rate_dict is not None:
            return self.learning_rate_dict.get(task_key, self.learning_rate)
        return self.learning_rate
