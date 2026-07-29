# SPDX-License-Identifier: LGPL-3.0-or-later
"""Core training loop implementations for different optimizer types.

This module provides specialized training loops for different optimizers,
making it easy to add new training strategies while keeping the main
trainer clean.
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
    Any,
)

import torch
import torch.distributed as dist

from deepmd.pt.loss import (
    DenoiseLoss,
    EnergyStdLoss,
)
from deepmd.pt.optimizer import (
    KFOptimizerWrapper,
)

log = logging.getLogger(__name__)


class TrainingStepResult:
    """Container for training step results.

    Attributes
    ----------
    loss : torch.Tensor
        The computed loss.
    model_pred : dict[str, torch.Tensor]
        Model predictions.
    more_loss : dict[str, Any]
        Additional loss components.
    lr : float
        Current learning rate.
    """

    def __init__(
        self,
        loss: torch.Tensor,
        model_pred: dict[str, torch.Tensor],
        more_loss: dict[str, Any],
        lr: float,
    ) -> None:
        self.loss = loss
        self.model_pred = model_pred
        self.more_loss = more_loss
        self.lr = lr


class BaseTrainingLoop(ABC):
    """Abstract base class for training loops.

    Subclasses implement specific training strategies for different
    optimizer types and training modes.
    """

    def __init__(
        self,
        wrapper: torch.nn.Module,
        optimizer: Any,
        gradient_max_norm: float = 0.0,
    ) -> None:
        """Initialize training loop.

        Parameters
        ----------
        wrapper : torch.nn.Module
            Model wrapper (may be wrapped in DDP).
        optimizer : Any
            Optimizer instance.
        gradient_max_norm : float
            Maximum gradient norm for clipping (0.0 = disabled).
        """
        self.wrapper = wrapper
        self.optimizer = optimizer
        self.gradient_max_norm = gradient_max_norm

    @abstractmethod
    def step(
        self,
        input_dict: dict[str, torch.Tensor],
        label_dict: dict[str, torch.Tensor],
        cur_lr: float,
        pref_lr: float,
        task_key: str = "Default",
    ) -> TrainingStepResult:
        """Execute a single training step.

        Parameters
        ----------
        input_dict : dict[str, torch.Tensor]
            Input tensors.
        label_dict : dict[str, torch.Tensor]
            Label tensors.
        cur_lr : float
            Current learning rate from scheduler.
        pref_lr : float
            Preferred learning rate for loss computation.
        task_key : str
            Task key for multi-task training.

        Returns
        -------
        TrainingStepResult
            Results from the training step.
        """
        pass

    def zero_grad(self) -> None:
        """Zero optimizer gradients."""
        self.optimizer.zero_grad(set_to_none=True)

    def _get_module(self) -> torch.nn.Module:
        """Get unwrapped module from DDP if needed."""
        module = self.wrapper
        if dist.is_available() and dist.is_initialized():
            if hasattr(module, "module"):
                module = module.module
        return module


class AdamTrainingLoop(BaseTrainingLoop):
    """Training loop for Adam/AdamW/AdaMuon/HybridMuon optimizers.

    Standard backpropagation with gradient clipping support.
    """

    def step(
        self,
        input_dict: dict[str, torch.Tensor],
        label_dict: dict[str, torch.Tensor],
        cur_lr: float,
        pref_lr: float,
        task_key: str = "Default",
    ) -> TrainingStepResult:
        """Execute training step with standard backpropagation."""
        self.zero_grad()

        # Forward pass
        model_pred, loss, more_loss = self.wrapper(
            **input_dict,
            cur_lr=pref_lr,
            label=label_dict,
            task_key=task_key,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        if self.gradient_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                self.wrapper.parameters(),
                self.gradient_max_norm,
                error_if_nonfinite=True,
            )

        # Optimizer step
        with torch.device("cpu"):
            self.optimizer.step()

        return TrainingStepResult(
            loss=loss,
            model_pred=model_pred,
            more_loss=more_loss,
            lr=cur_lr,
        )


class LKFEnergyTrainingLoop(BaseTrainingLoop):
    """Training loop for LKF optimizer with energy/force loss.

    Uses Kalman Filter optimizer for energy and force updates.
    """

    def __init__(
        self,
        wrapper: torch.nn.Module,
        optimizer: Any,
        opt_param: dict[str, Any],
        num_steps: int,
        gradient_max_norm: float = 0.0,
    ) -> None:
        """Initialize LKF training loop.

        Parameters
        ----------
        wrapper : torch.nn.Module
            Model wrapper.
        optimizer : Any
            LKF optimizer.
        opt_param : dict[str, Any]
            Optimizer parameters including kf_start_pref_e, etc.
        num_steps : int
            Total training steps for prefactor scheduling.
        gradient_max_norm : float
            Maximum gradient norm (not used for LKF).
        """
        super().__init__(wrapper, optimizer, gradient_max_norm)
        self.opt_param = opt_param
        self.num_steps = num_steps

        # Create KF wrapper
        self.kf_wrapper = KFOptimizerWrapper(
            wrapper,
            optimizer,
            24,  # kp
            6,  # kq
            dist.is_available() and dist.is_initialized(),
        )

    def _compute_prefactors(self, step: int) -> tuple[float, float]:
        """Compute energy and force prefactors for current step."""
        start_pref_e = self.opt_param["kf_start_pref_e"]
        limit_pref_e = self.opt_param["kf_limit_pref_e"]
        start_pref_f = self.opt_param["kf_start_pref_f"]
        limit_pref_f = self.opt_param["kf_limit_pref_f"]

        ratio = step / self.num_steps

        pref_e = start_pref_e * (limit_pref_e / start_pref_e) ** ratio
        pref_f = start_pref_f * (limit_pref_f / start_pref_f) ** ratio

        return pref_e, pref_f

    def step(
        self,
        input_dict: dict[str, torch.Tensor],
        label_dict: dict[str, torch.Tensor],
        cur_lr: float,
        pref_lr: float,
        task_key: str = "Default",
    ) -> TrainingStepResult:
        """Execute LKF training step."""
        # Compute prefactors
        step = self.optimizer.state.get("step", 0)
        pref_e, pref_f = self._compute_prefactors(step)

        # Update energy
        _ = self.kf_wrapper.update_energy(input_dict, label_dict["energy"], pref_e)

        # Update force
        p_energy, p_force = self.kf_wrapper.update_force(
            input_dict, label_dict["force"], pref_f
        )

        model_pred = {"energy": p_energy, "force": p_force}

        # Compute loss using wrapper's loss function
        module = self._get_module()

        def fake_model() -> dict[str, torch.Tensor]:
            return model_pred

        natoms = int(input_dict["atype"].shape[-1])

        _, loss, more_loss = module.loss[task_key](
            {},
            fake_model,
            label_dict,
            natoms,
            learning_rate=pref_lr,
        )

        return TrainingStepResult(
            loss=loss,
            model_pred=model_pred,
            more_loss=more_loss,
            lr=cur_lr,
        )


class LKFDenoiseTrainingLoop(BaseTrainingLoop):
    """Training loop for LKF optimizer with denoising loss."""

    def __init__(
        self,
        wrapper: torch.nn.Module,
        optimizer: Any,
        loss: DenoiseLoss,
        gradient_max_norm: float = 0.0,
    ) -> None:
        """Initialize LKF denoise training loop."""
        super().__init__(wrapper, optimizer, gradient_max_norm)

        self.kf_wrapper = KFOptimizerWrapper(
            wrapper,
            optimizer,
            24,  # kp
            6,  # kq
            dist.is_available() and dist.is_initialized(),
        )
        self.loss_module = loss

    def step(
        self,
        input_dict: dict[str, torch.Tensor],
        label_dict: dict[str, torch.Tensor],
        cur_lr: float,
        pref_lr: float,
        task_key: str = "Default",
    ) -> TrainingStepResult:
        """Execute LKF denoise training step."""
        module = self._get_module()
        loss_fn = module.loss[task_key]

        # Update coordinates via KF
        model_pred = self.kf_wrapper.update_denoise_coord(
            input_dict,
            label_dict["clean_coord"],
            1,  # prefactor
            loss_fn.mask_loss_coord,
            label_dict.get("coord_mask"),
        )

        # Compute loss
        loss, more_loss = loss_fn(
            model_pred,
            label_dict,
            input_dict["natoms"],
            learning_rate=pref_lr,
        )

        return TrainingStepResult(
            loss=loss,
            model_pred=model_pred,
            more_loss=more_loss,
            lr=cur_lr,
        )


class TrainingLoopFactory:
    """Factory for creating appropriate training loops.

    Selects the correct training loop implementation based on
    optimizer type and loss function.
    """

    def __init__(
        self,
        opt_type: str,
        opt_param: dict[str, Any],
        num_steps: int,
    ) -> None:
        """Initialize factory.

        Parameters
        ----------
        opt_type : str
            Type of optimizer.
        opt_param : dict[str, Any]
            Optimizer parameters.
        num_steps : int
            Total training steps.
        """
        self.opt_type = opt_type
        self.opt_param = opt_param
        self.num_steps = num_steps

    def create(
        self,
        wrapper: torch.nn.Module,
        optimizer: Any,
        loss: Any,
        gradient_max_norm: float = 0.0,
    ) -> BaseTrainingLoop:
        """Create training loop instance.

        Parameters
        ----------
        wrapper : torch.nn.Module
            Model wrapper.
        optimizer : Any
            Optimizer instance.
        loss : Any
            Loss function/module.
        gradient_max_norm : float
            Maximum gradient norm.

        Returns
        -------
        BaseTrainingLoop
            Appropriate training loop for the configuration.

        Raises
        ------
        ValueError
            If optimizer type is not supported.
        """
        if self.opt_type in ["Adam", "AdamW", "AdaMuon", "HybridMuon"]:
            return AdamTrainingLoop(
                wrapper,
                optimizer,
                gradient_max_norm,
            )

        elif self.opt_type == "LKF":
            if isinstance(loss, EnergyStdLoss):
                return LKFEnergyTrainingLoop(
                    wrapper,
                    optimizer,
                    self.opt_param,
                    self.num_steps,
                    gradient_max_norm,
                )
            elif isinstance(loss, DenoiseLoss):
                return LKFDenoiseTrainingLoop(
                    wrapper,
                    optimizer,
                    loss,
                    gradient_max_norm,
                )
            else:
                raise ValueError(
                    f"LKF optimizer not supported for loss type: {type(loss)}"
                )

        else:
            raise ValueError(f"Unsupported optimizer type: {self.opt_type}")
