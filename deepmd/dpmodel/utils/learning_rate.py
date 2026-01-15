# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.common import (
    j_get_type,
)
from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.utils.plugin import (
    PluginVariant,
    make_plugin_registry,
)


class BaseLR(ABC, PluginVariant, make_plugin_registry("lr")):
    def __new__(cls: type, *args: Any, **kwargs: Any) -> Any:
        if cls is BaseLR:
            cls = cls.get_class_by_type(j_get_type(kwargs, cls.__name__))
        return super().__new__(cls)

    def __init__(
        self,
        start_lr: float,
        stop_lr: float | None = None,
        stop_ratio: float | None = None,
        stop_steps: int = 100000,
        warmup_steps: int = 0,
        warmup_ratio: float | None = None,
        warmup_start_factor: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Base class for learning rate schedules with warmup support.

        Parameters
        ----------
        start_lr : float
            The learning rate at the start of the training (after warmup).
        stop_lr : float, optional
            The final learning rate at the end of the training.
            Mutually exclusive with stop_ratio.
        stop_ratio : float, optional
            The ratio of stop_lr to start_lr. stop_lr = start_lr * stop_ratio.
            Mutually exclusive with stop_lr.
            One of stop_lr or stop_ratio must be provided.
        stop_steps : int
            The total training steps (including warmup).
        warmup_steps : int, optional
            The number of steps for learning rate warmup.
            Mutually exclusive with warmup_ratio. Default is 0 (no warmup).
        warmup_ratio : float, optional
            The ratio of warmup steps to total training steps.
            warmup_steps = int(warmup_ratio * stop_steps).
            Mutually exclusive with warmup_steps.
        warmup_start_factor : float, optional
            The factor of start_lr for the initial warmup learning rate.
            The warmup learning rate starts from warmup_start_factor * start_lr.
            Default is 0.0.
        """
        # === Step 1. Compute stop_lr from stop_ratio if needed ===
        # Mutual exclusion validated in argcheck.py
        if stop_ratio is not None:
            self.stop_lr = start_lr * stop_ratio
        else:
            self.stop_lr = stop_lr  # type: ignore[assignment]

        # === Step 2. Compute warmup_steps from warmup_ratio if needed ===
        # Mutual exclusion validated in argcheck.py
        if warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * stop_steps)
        else:
            self.warmup_steps = warmup_steps

        # === Step 3. Validate step ranges (runtime check) ===
        if stop_steps <= 0:
            raise ValueError("stop_steps must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.warmup_steps >= stop_steps:
            raise ValueError("warmup_steps must be smaller than stop_steps")

        # === Step 4. Compute warmup_start_lr ===
        self.warmup_start_lr = warmup_start_factor * start_lr

        # === Step 5. Store core parameters ===
        self.start_lr = start_lr
        self.stop_steps = stop_steps
        # Decay phase covers (stop_steps - warmup_steps) steps
        self.decay_stop_steps = stop_steps - self.warmup_steps

    @abstractmethod
    def _decay_value(self, step: int | Array) -> Array:
        """
        Get the decayed learning rate at the given step (after warmup).

        This method should implement the actual decay logic (exp, cosine, etc.)
        without considering warmup.

        Parameters
        ----------
        step : int or Array
            The step index relative to the end of warmup.
            For example, if warmup_steps=100 and total_step=150, this method
            will be called with step=50.

        Returns
        -------
        Array
            The decayed learning rate (absolute value, not factor).
        """
        pass

    def value(self, step: int | Array) -> Array | float:
        """
        Get the learning rate at the given step, including warmup.

        Parameters
        ----------
        step : int or Array
            The absolute step index from the start of training.

        Returns
        -------
        Array
            The learning rate at the given step.
        """
        is_scalar = isinstance(step, (int, float))
        if not array_api_compat.is_array_api_obj(step):
            step = np.asarray(step)
        xp = array_api_compat.array_namespace(step)

        # === Step 1. Handle no-warmup case directly ===
        if self.warmup_steps == 0:
            lr = self._decay_value(xp.astype(step, xp.float64))
        else:
            # === Step 2. Warmup phase ===
            # Linear warmup from warmup_start_lr to start_lr
            warmup_progress = xp.astype(step, xp.float64) / self.warmup_steps
            warmup_lr = (
                self.warmup_start_lr
                + (self.start_lr - self.warmup_start_lr) * warmup_progress
            )

            # === Step 3. Decay phase ===
            # Call subclass decay logic for steps after warmup
            decay_step = xp.maximum(
                xp.astype(step, xp.float64) - self.warmup_steps, 0.0
            )
            decay_lr = self._decay_value(decay_step)

            # === Step 4. Select warmup or decay based on step ===
            lr = xp.where(step < self.warmup_steps, warmup_lr, decay_lr)

        if is_scalar:
            return float(lr)
        return lr


@BaseLR.register("exp")
class LearningRateExp(BaseLR):
    def __init__(
        self,
        start_lr: float,
        stop_lr: float | None = None,
        stop_ratio: float | None = None,
        decay_steps: int = 5000,
        stop_steps: int = 100000,
        decay_rate: float | None = None,
        warmup_steps: int = 0,
        warmup_ratio: float | None = None,
        warmup_start_factor: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Construct an exponential-decayed learning rate with optional warmup.

        Parameters
        ----------
        start_lr : float
            The learning rate at the start of the training (after warmup).
        stop_lr : float, optional
            The desired learning rate at the end of the training.
            When decay_rate is explicitly set, this value will serve as
            the minimum learning rate during training.
            Mutually exclusive with stop_ratio.
        stop_ratio : float, optional
            The ratio of stop_lr to start_lr.
            Mutually exclusive with stop_lr.
        decay_steps : int
            The learning rate is decaying every this number of training steps.
            Default is 5000.
        stop_steps : int
            The total training steps (including warmup).
        decay_rate : float, optional
            The decay rate for the learning rate.
            If provided, the decay rate will be set instead of
            calculating it through interpolation between start_lr and stop_lr.
        warmup_steps : int, optional
            The number of steps for learning rate warmup.
            Mutually exclusive with warmup_ratio. Default is 0.
        warmup_ratio : float, optional
            The ratio of warmup steps to total training steps.
            Mutually exclusive with warmup_steps.
        warmup_start_factor : float, optional
            The factor of start_lr for the initial warmup learning rate.
            Default is 0.0.

        Raises
        ------
        ValueError
            If both stop_lr and stop_ratio are provided, or neither is provided.
            If both warmup_steps and warmup_ratio are provided.
            If decay_steps is larger than the decay phase total steps.
        """
        super().__init__(
            start_lr=start_lr,
            stop_lr=stop_lr,
            stop_ratio=stop_ratio,
            stop_steps=stop_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            warmup_start_factor=warmup_start_factor,
            **kwargs,
        )
        # === Step 5. Compute decay_rate for exp scheduler ===
        # Use decay_stop_steps (stop_steps - warmup_steps) for decay calculation
        decay_total = self.decay_stop_steps
        self.decay_steps = decay_steps

        if self.decay_steps > decay_total:
            raise ValueError(
                f"decay_steps ({self.decay_steps}) must not exceed decay phase steps ({decay_total})."
            )

        # Avoid log(0) issues by clamping stop_lr for computation
        clamped_stop_lr = max(self.stop_lr, 1e-10)
        self.min_lr = self.stop_lr

        self.decay_rate = np.exp(
            np.log(clamped_stop_lr / self.start_lr) / (decay_total / self.decay_steps)
        ).item()
        if decay_rate is not None:
            self.decay_rate = decay_rate

    def _decay_value(self, step: int | Array) -> Array:
        """
        Get the exponential-decayed learning rate factor at the given step.

        Parameters
        ----------
        step : int or Array
            The step index relative to the end of warmup.

        Returns
        -------
        Array
            The decayed learning rate (absolute value).
        """
        if not array_api_compat.is_array_api_obj(step):
            step = np.asarray(step)
        xp = array_api_compat.array_namespace(step)
        step_lr = self.start_lr * xp.pow(
            xp.asarray(self.decay_rate, device=array_api_compat.device(step)),
            xp.astype(step // self.decay_steps, xp.float64),
        )
        # Clip to min_lr for numerical stability in JIT
        step_lr = xp.clip(step_lr, self.min_lr, None)
        return step_lr


@BaseLR.register("cosine")
class LearningRateCosine(BaseLR):
    def __init__(
        self,
        start_lr: float,
        stop_lr: float | None = None,
        stop_ratio: float | None = None,
        stop_steps: int = 100000,
        warmup_steps: int = 0,
        warmup_ratio: float | None = None,
        warmup_start_factor: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Defines a cosine annealing learning rate schedule with optional warmup.

        The learning rate starts at `start_lr` (after warmup) and gradually
        decreases to `stop_lr` following a cosine curve over the training steps.

        Parameters
        ----------
        start_lr : float
            The learning rate at the start of the training (after warmup).
        stop_lr : float, optional
            The final learning rate at the end of training.
            Mutually exclusive with stop_ratio.
        stop_ratio : float, optional
            The ratio of stop_lr to start_lr.
            Mutually exclusive with stop_lr.
        stop_steps : int
            The total training steps (including warmup).
        warmup_steps : int, optional
            The number of steps for learning rate warmup.
            Mutually exclusive with warmup_ratio. Default is 0.
        warmup_ratio : float, optional
            The ratio of warmup steps to total training steps.
            Mutually exclusive with warmup_steps.
        warmup_start_factor : float, optional
            The factor of start_lr for the initial warmup learning rate.
            Default is 0.0.

        Raises
        ------
        ValueError
            If both stop_lr and stop_ratio are provided, or neither is provided.
            If both warmup_steps and warmup_ratio are provided.
        """
        super().__init__(
            start_lr=start_lr,
            stop_lr=stop_lr,
            stop_ratio=stop_ratio,
            stop_steps=stop_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            warmup_start_factor=warmup_start_factor,
            **kwargs,
        )
        self.lr_min_factor = self.stop_lr / self.start_lr

    def _decay_value(self, step: int | Array) -> Array:
        """
        Get the cosine-annealed learning rate at the given step.

        Parameters
        ----------
        step : int or Array
            The step index relative to the end of warmup.

        Returns
        -------
        Array
            The annealed learning rate (absolute value).
        """
        if not array_api_compat.is_array_api_obj(step):
            step = np.asarray(step)
        xp = array_api_compat.array_namespace(step)
        min_lr = self.start_lr * self.lr_min_factor
        step_lr = self.start_lr * (
            self.lr_min_factor
            + 0.5
            * (1 - self.lr_min_factor)
            * (
                1
                + xp.cos(
                    xp.asarray(
                        xp.pi * (xp.astype(step, xp.float64) / self.decay_stop_steps),
                        device=array_api_compat.device(step),
                    )
                )
            )
        )
        # Clip to min_lr for steps beyond decay_stop_steps
        step_lr = xp.where(step >= self.decay_stop_steps, min_lr, step_lr)
        return step_lr
