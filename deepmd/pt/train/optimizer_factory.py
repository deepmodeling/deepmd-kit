# SPDX-License-Identifier: LGPL-3.0-or-later
"""Optimizer and learning rate scheduler factory.

This module provides a factory pattern for creating optimizers and
learning rate schedulers, making it easy to add new optimizer types
and customize their behavior.
"""

from __future__ import (
    annotations,
)

import logging
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from deepmd.pt.optimizer import (
    AdaMuonOptimizer,
    HybridMuonOptimizer,
    LKFOptimizer,
)
from deepmd.pt.utils.env import (
    DEVICE,
)

if TYPE_CHECKING:
    from torch.optim import (
        Optimizer,
    )
    from torch.optim.lr_scheduler import (
        LRScheduler,
    )

    from deepmd.pt.train.config import (
        OptimizerConfig,
    )

log = logging.getLogger(__name__)


class OptimizerStrategy(ABC):
    """Abstract base class for optimizer creation strategies.

    This class defines the interface for creating optimizers and their
    associated learning rate schedulers. Subclasses implement specific
    optimizer types.
    """

    @abstractmethod
    def create_optimizer(
        self,
        parameters: Any,
        config: OptimizerConfig,
        lr_config: Any,
    ) -> Optimizer | Any:
        """Create an optimizer instance.

        Parameters
        ----------
        parameters : Any
            Model parameters to optimize.
        config : OptimizerConfig
            Optimizer configuration.
        lr_config : Any
            Learning rate configuration.

        Returns
        -------
        Optimizer | Any
            The created optimizer instance.
        """
        pass

    @abstractmethod
    def create_scheduler(
        self,
        optimizer: Optimizer | Any,
        warmup_steps: int,
        warmup_start_factor: float,
        lr_schedule: Any,
        start_step: int = 0,
    ) -> LRScheduler | None:
        """Create a learning rate scheduler.

        Parameters
        ----------
        optimizer : Optimizer | Any
            The optimizer to schedule.
        warmup_steps : int
            Number of warmup steps.
        warmup_start_factor : float
            Initial LR factor during warmup.
        lr_schedule : Any
            Learning rate schedule object.
        start_step : int
            Starting step for scheduler.

        Returns
        -------
        LRScheduler | None
            The created scheduler, or None if not applicable.
        """
        pass

    @abstractmethod
    def supports_scheduler(self) -> bool:
        """Whether this optimizer supports LR scheduling.

        Returns
        -------
        bool
            True if scheduler is supported, False otherwise.
        """
        pass


class AdamStrategy(OptimizerStrategy):
    """Strategy for creating Adam optimizer."""

    def create_optimizer(
        self,
        parameters: Any,
        config: OptimizerConfig,
        lr_config: Any,
    ) -> Optimizer:
        """Create Adam optimizer."""
        return torch.optim.Adam(
            parameters,
            lr=lr_config.start_lr,
            fused=False if DEVICE.type == "cpu" else True,
        )

    def create_scheduler(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_start_factor: float,
        lr_schedule: Any,
        start_step: int = 0,
    ) -> LRScheduler:
        """Create LambdaLR scheduler with warmup."""

        def warmup_linear(step: int) -> float:
            """Compute LR multiplier with warmup."""
            current_step = step + start_step
            if current_step < warmup_steps:
                return warmup_start_factor + (1.0 - warmup_start_factor) * (
                    current_step / warmup_steps
                )
            else:
                return (
                    lr_schedule.value(current_step - warmup_steps)
                    / lr_schedule.start_lr
                )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_linear)

    def supports_scheduler(self) -> bool:
        return True


class AdamWStrategy(OptimizerStrategy):
    """Strategy for creating AdamW optimizer."""

    def create_optimizer(
        self,
        parameters: Any,
        config: OptimizerConfig,
        lr_config: Any,
    ) -> Optimizer:
        """Create AdamW optimizer."""
        return torch.optim.AdamW(
            parameters,
            lr=lr_config.start_lr,
            weight_decay=config.weight_decay,
            fused=False if DEVICE.type == "cpu" else True,
        )

    def create_scheduler(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        warmup_start_factor: float,
        lr_schedule: Any,
        start_step: int = 0,
    ) -> LRScheduler:
        """Create LambdaLR scheduler with warmup."""

        def warmup_linear(step: int) -> float:
            """Compute LR multiplier with warmup."""
            current_step = step + start_step
            if current_step < warmup_steps:
                return warmup_start_factor + (1.0 - warmup_start_factor) * (
                    current_step / warmup_steps
                )
            else:
                return (
                    lr_schedule.value(current_step - warmup_steps)
                    / lr_schedule.start_lr
                )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_linear)

    def supports_scheduler(self) -> bool:
        return True


class LKFStrategy(OptimizerStrategy):
    """Strategy for creating LKF (Levenberg-Kalman Filter) optimizer."""

    def create_optimizer(
        self,
        parameters: Any,
        config: OptimizerConfig,
        lr_config: Any,
    ) -> LKFOptimizer:
        """Create LKF optimizer."""
        return LKFOptimizer(
            parameters,
            0.98,  # Kalman lambda
            0.99870,  # Kalman nu
            config.kf_blocksize,
        )

    def create_scheduler(
        self,
        optimizer: Optimizer | Any,
        warmup_steps: int,
        warmup_start_factor: float,
        lr_schedule: Any,
        start_step: int = 0,
    ) -> None:
        """LKF doesn't use a scheduler."""
        return None

    def supports_scheduler(self) -> bool:
        return False


class AdaMuonStrategy(OptimizerStrategy):
    """Strategy for creating AdaMuon optimizer."""

    def create_optimizer(
        self,
        parameters: Any,
        config: OptimizerConfig,
        lr_config: Any,
    ) -> AdaMuonOptimizer:
        """Create AdaMuon optimizer."""
        return AdaMuonOptimizer(
            parameters,
            lr=lr_config.start_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            adam_betas=(config.adam_beta1, config.adam_beta2),
            lr_adjust=config.lr_adjust,
            lr_adjust_coeff=config.lr_adjust_coeff,
        )

    def create_scheduler(
        self,
        optimizer: Optimizer | Any,
        warmup_steps: int,
        warmup_start_factor: float,
        lr_schedule: Any,
        start_step: int = 0,
    ) -> LRScheduler:
        """Create LambdaLR scheduler with warmup."""

        def warmup_linear(step: int) -> float:
            """Compute LR multiplier with warmup."""
            current_step = step + start_step
            if current_step < warmup_steps:
                return warmup_start_factor + (1.0 - warmup_start_factor) * (
                    current_step / warmup_steps
                )
            else:
                return (
                    lr_schedule.value(current_step - warmup_steps)
                    / lr_schedule.start_lr
                )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_linear)

    def supports_scheduler(self) -> bool:
        return True


class HybridMuonStrategy(OptimizerStrategy):
    """Strategy for creating HybridMuon optimizer."""

    def create_optimizer(
        self,
        parameters: Any,
        config: OptimizerConfig,
        lr_config: Any,
    ) -> HybridMuonOptimizer:
        """Create HybridMuon optimizer."""
        return HybridMuonOptimizer(
            parameters,
            lr=lr_config.start_lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            adam_betas=(config.adam_beta1, config.adam_beta2),
            lr_adjust=config.lr_adjust,
            lr_adjust_coeff=config.lr_adjust_coeff,
            muon_2d_only=config.muon_2d_only,
            min_2d_dim=config.min_2d_dim,
        )

    def create_scheduler(
        self,
        optimizer: Optimizer | Any,
        warmup_steps: int,
        warmup_start_factor: float,
        lr_schedule: Any,
        start_step: int = 0,
    ) -> LRScheduler:
        """Create LambdaLR scheduler with warmup."""

        def warmup_linear(step: int) -> float:
            """Compute LR multiplier with warmup."""
            current_step = step + start_step
            if current_step < warmup_steps:
                return warmup_start_factor + (1.0 - warmup_start_factor) * (
                    current_step / warmup_steps
                )
            else:
                return (
                    lr_schedule.value(current_step - warmup_steps)
                    / lr_schedule.start_lr
                )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_linear)

    def supports_scheduler(self) -> bool:
        return True


class OptimizerFactory:
    """Factory for creating optimizers and schedulers.

    This factory centralizes optimizer creation and makes it easy to
    register new optimizer types.

    Example:
        >>> factory = OptimizerFactory()
        >>> optimizer = factory.create_optimizer(
        ...     parameters=model.parameters(),
        ...     config=OptimizerConfig(opt_type="Adam"),
        ...     lr_config=lr_config,
        ... )
    """

    def __init__(self) -> None:
        """Initialize factory with default strategies."""
        self._strategies: dict[str, OptimizerStrategy] = {
            "Adam": AdamStrategy(),
            "AdamW": AdamWStrategy(),
            "LKF": LKFStrategy(),
            "AdaMuon": AdaMuonStrategy(),
            "HybridMuon": HybridMuonStrategy(),
        }

    def register(self, opt_type: str, strategy: OptimizerStrategy) -> None:
        """Register a new optimizer strategy.

        Parameters
        ----------
        opt_type : str
            Identifier for the optimizer type.
        strategy : OptimizerStrategy
            Strategy instance for creating the optimizer.
        """
        self._strategies[opt_type] = strategy
        log.info(f"Registered optimizer strategy: {opt_type}")

    def create_optimizer(
        self,
        parameters: Any,
        config: OptimizerConfig,
        lr_config: Any,
    ) -> Optimizer | Any:
        """Create an optimizer.

        Parameters
        ----------
        parameters : Any
            Model parameters to optimize.
        config : OptimizerConfig
            Optimizer configuration.
        lr_config : Any
            Learning rate configuration.

        Returns
        -------
        Optimizer | Any
            The created optimizer.

        Raises
        ------
        ValueError
            If optimizer type is not registered.
        """
        if config.opt_type not in self._strategies:
            raise ValueError(
                f"Unknown optimizer type: {config.opt_type}. "
                f"Available: {list(self._strategies.keys())}"
            )

        strategy = self._strategies[config.opt_type]
        return strategy.create_optimizer(parameters, config, lr_config)

    def create_scheduler(
        self,
        opt_type: str,
        optimizer: Optimizer | Any,
        warmup_steps: int,
        warmup_start_factor: float,
        lr_schedule: Any,
        start_step: int = 0,
    ) -> LRScheduler | None:
        """Create a learning rate scheduler.

        Parameters
        ----------
        opt_type : str
            Type of optimizer.
        optimizer : Optimizer | Any
            The optimizer to schedule.
        warmup_steps : int
            Number of warmup steps.
        warmup_start_factor : float
            Initial LR factor during warmup.
        lr_schedule : Any
            Learning rate schedule object.
        start_step : int
            Starting step for scheduler.

        Returns
        -------
        LRScheduler | None
            The created scheduler, or None if not supported.
        """
        if opt_type not in self._strategies:
            return None

        strategy = self._strategies[opt_type]
        if not strategy.supports_scheduler():
            return None

        return strategy.create_scheduler(
            optimizer,
            warmup_steps,
            warmup_start_factor,
            lr_schedule,
            start_step,
        )

    def supports_scheduler(self, opt_type: str) -> bool:
        """Check if optimizer type supports LR scheduling.

        Parameters
        ----------
        opt_type : str
            Type of optimizer.

        Returns
        -------
        bool
            True if scheduler is supported.
        """
        if opt_type not in self._strategies:
            return False
        return self._strategies[opt_type].supports_scheduler()

    def get_available_optimizers(self) -> list[str]:
        """Get list of available optimizer types.

        Returns
        -------
        list[str]
            List of registered optimizer type names.
        """
        return list(self._strategies.keys())


# Global factory instance for convenience
_default_factory = OptimizerFactory()


def create_optimizer(
    parameters: Any,
    config: OptimizerConfig,
    lr_config: Any,
) -> Optimizer | Any:
    """Convenience function to create optimizer using default factory."""
    return _default_factory.create_optimizer(parameters, config, lr_config)


def create_scheduler(
    opt_type: str,
    optimizer: Optimizer | Any,
    warmup_steps: int,
    warmup_start_factor: float,
    lr_schedule: Any,
    start_step: int = 0,
) -> LRScheduler | None:
    """Convenience function to create scheduler using default factory."""
    return _default_factory.create_scheduler(
        opt_type,
        optimizer,
        warmup_steps,
        warmup_start_factor,
        lr_schedule,
        start_step,
    )
