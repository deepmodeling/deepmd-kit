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
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
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
        num_steps: int,
        stop_lr: float | None = None,
        stop_lr_ratio: float | None = None,
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
            Mutually exclusive with stop_lr_ratio.
        stop_lr_ratio : float, optional
            The ratio of stop_lr to start_lr. stop_lr = start_lr * stop_lr_ratio.
            Mutually exclusive with stop_lr.
            One of stop_lr or stop_lr_ratio must be provided.
        num_steps : int
            The total training steps (including warmup).
        warmup_steps : int, optional
            The number of steps for learning rate warmup.
            Mutually exclusive with warmup_ratio. Default is 0 (no warmup).
        warmup_ratio : float, optional
            The ratio of warmup steps to total training steps.
            warmup_steps = int(warmup_ratio * num_steps).
            Mutually exclusive with warmup_steps.
        warmup_start_factor : float, optional
            The factor of start_lr for the initial warmup learning rate.
            The warmup learning rate starts from warmup_start_factor * start_lr.
            Default is 0.0.
        """
        # === Step 1. Validate stop_lr and stop_lr_ratio (runtime check) ===
        has_stop_lr = stop_lr is not None
        has_stop_lr_ratio = stop_lr_ratio is not None

        if has_stop_lr and has_stop_lr_ratio:
            raise ValueError(
                "stop_lr and stop_lr_ratio are mutually exclusive. "
                f"Got stop_lr={stop_lr}, stop_lr_ratio={stop_lr_ratio}"
            )
        if not has_stop_lr and not has_stop_lr_ratio:
            raise ValueError(
                "Either stop_lr or stop_lr_ratio must be provided. "
                "Got stop_lr=None, stop_lr_ratio=None"
            )

        # === Step 2. Compute stop_lr from stop_lr_ratio if needed ===
        if stop_lr_ratio is not None:
            self.stop_lr = start_lr * stop_lr_ratio
        else:
            self.stop_lr = stop_lr

        # === Step 3. Validate warmup_steps and warmup_ratio (runtime check) ===
        has_warmup_steps = warmup_steps != 0
        has_warmup_ratio = warmup_ratio is not None

        if has_warmup_steps and has_warmup_ratio:
            raise ValueError(
                "warmup_steps and warmup_ratio are mutually exclusive. "
                f"Got warmup_steps={warmup_steps}, warmup_ratio={warmup_ratio}"
            )

        # === Step 4. Compute warmup_steps from warmup_ratio if needed ===
        if warmup_ratio is not None:
            self.warmup_steps = int(warmup_ratio * num_steps)
        else:
            self.warmup_steps = warmup_steps

        # === Step 5. Validate step ranges (runtime check) ===
        if num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if num_steps > 0 and self.warmup_steps >= num_steps:
            raise ValueError("warmup_steps must be smaller than num_steps")
        if num_steps == 0 and self.warmup_steps != 0:
            raise ValueError("warmup_steps must be 0 when num_steps is 0")

        # === Step 6. Compute warmup_start_lr ===
        self.warmup_start_lr = warmup_start_factor * start_lr

        # === Step 7. Store core parameters ===
        self._start_lr = start_lr
        self.num_steps = num_steps
        # Decay phase covers (num_steps - warmup_steps) steps
        self.decay_num_steps = num_steps - self.warmup_steps

    @property
    def start_lr(self) -> float:
        """
        Get the starting learning rate.

        Returns
        -------
        float
            The starting learning rate.
        """
        return self._start_lr

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
        # Use input dtype for floating point, or default to GLOBAL_NP_FLOAT_PRECISION for integers
        step_dtype = (
            step.dtype
            if np.issubdtype(step.dtype, np.floating)
            else GLOBAL_NP_FLOAT_PRECISION
        )
        if self.warmup_steps == 0:
            lr = self._decay_value(xp.astype(step, step_dtype))
        else:
            # === Step 2. Warmup phase ===
            # Linear warmup from warmup_start_lr to start_lr
            warmup_progress = xp.astype(step, step_dtype) / self.warmup_steps
            warmup_lr = (
                self.warmup_start_lr
                + (self._start_lr - self.warmup_start_lr) * warmup_progress
            )

            # === Step 3. Decay phase ===
            # Call subclass decay logic for steps after warmup
            decay_step = xp.maximum(
                xp.astype(step, step_dtype) - self.warmup_steps,
                xp.asarray(0.0, dtype=step_dtype),
            )
            decay_lr = self._decay_value(decay_step)

            # === Step 4. Select warmup or decay based on step ===
            lr = xp.where(step < self.warmup_steps, warmup_lr, decay_lr)

        if is_scalar:
            return float(lr)
        return lr


@BaseLR.register("exp")
class LearningRateExp(BaseLR):
    r"""
    Exponential decay learning rate schedule with optional warmup.

    The decay phase (after warmup) follows the exponential decay formula.

    **Stepped mode (smooth=False, default):**

    .. math::

        lr(t) = lr_0 \cdot r^{\lfloor t / s \rfloor}

    The learning rate decays every ``decay_steps`` steps, creating a staircase
    pattern.

    **Smooth mode (smooth=True):**

    .. math::

        lr(t) = lr_0 \cdot r^{t / s}

    The learning rate decays continuously at every step.

    where:
    - :math:`lr_0` is ``start_lr`` (learning rate at the start of decay phase)
    - :math:`r` is the decay rate ``decay_rate``
    - :math:`t` is the step index within the decay phase
    - :math:`s` is ``decay_steps`` (the decay period)

    The decay rate is automatically computed from ``start_lr`` and ``stop_lr``
    over the total decay steps unless explicitly provided:

    .. math::

        r = \left(\frac{lr_{\text{stop}}}{lr_0}\right)^{\frac{s}{T}}

    where :math:`T = \text{num\_steps} - \text{warmup\_steps}` is the total
    number of decay steps, and :math:`lr_{\text{stop}}` is ``stop_lr``.
    """

    def __init__(
        self,
        start_lr: float,
        num_steps: int,
        stop_lr: float | None = None,
        stop_lr_ratio: float | None = None,
        decay_steps: int = 5000,
        decay_rate: float | None = None,
        warmup_steps: int = 0,
        warmup_ratio: float | None = None,
        warmup_start_factor: float = 0.0,
        smooth: bool = False,
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
            Mutually exclusive with stop_lr_ratio.
        stop_lr_ratio : float, optional
            The ratio of stop_lr to start_lr.
            Mutually exclusive with stop_lr.
        decay_steps : int
            The learning rate is decaying every this number of training steps.
            Default is 5000.
        num_steps : int
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
        smooth : bool, optional
            If True, use smooth exponential decay (lr decays continuously).
            If False (default), use stepped decay (lr decays every decay_steps).
            Default is False.

        Raises
        ------
        ValueError
            If both stop_lr and stop_lr_ratio are provided, or neither is provided.
            If both warmup_steps and warmup_ratio are provided.
            If decay_steps is not positive.
        """
        super().__init__(
            start_lr=start_lr,
            stop_lr=stop_lr,
            stop_lr_ratio=stop_lr_ratio,
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            warmup_start_factor=warmup_start_factor,
            **kwargs,
        )
        # === Step 5. Compute decay_rate for exp scheduler ===
        # Use decay_num_steps (num_steps - warmup_steps) for decay calculation
        decay_total = self.decay_num_steps
        self.decay_steps = decay_steps

        if self.decay_steps <= 0:
            raise ValueError(f"decay_steps ({self.decay_steps}) must be positive.")

        # Auto-adjust decay_steps if it exceeds decay_total and decay_rate is not provided
        if decay_rate is None and self.decay_steps >= decay_total:
            # Compute sensible default: cap at 100, but ensure at least 1 for small decay_total
            default_ds = 100 if decay_total // 10 > 100 else decay_total // 100 + 1
            self.decay_steps = default_ds

        # Avoid log(0) issues by clamping stop_lr for computation
        clamped_stop_lr = max(self.stop_lr, 1e-10)
        self.min_lr = self.stop_lr

        # Compute decay_rate from start_lr/stop_lr if not explicitly provided
        if decay_rate is not None:
            self.decay_rate = decay_rate
        elif decay_total == 0:
            # No decay phase (num_steps == warmup_steps or num_steps == 0)
            self.decay_rate = 1.0  # No decay
        else:
            self.decay_rate = np.exp(
                np.log(clamped_stop_lr / self._start_lr)
                / (decay_total / self.decay_steps)
            ).item()

        # === Step 6. Store smooth mode ===
        self.smooth = smooth

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
        # === Step 1. Compute exponent based on smooth mode ===
        # Use input dtype for floating point, or default to GLOBAL_NP_FLOAT_PRECISION for integers
        step_dtype = (
            step.dtype
            if np.issubdtype(step.dtype, np.floating)
            else GLOBAL_NP_FLOAT_PRECISION
        )
        if self.smooth:
            exponent = xp.astype(step, step_dtype) / self.decay_steps
        else:
            exponent = xp.astype(step // self.decay_steps, step_dtype)
        step_lr = self._start_lr * xp.pow(
            xp.asarray(self.decay_rate, dtype=step_dtype),
            exponent,
        )
        # Clip to min_lr for numerical stability in JIT
        step_lr = xp.clip(step_lr, self.min_lr, None)
        return step_lr


@BaseLR.register("cosine")
class LearningRateCosine(BaseLR):
    r"""
    Cosine annealing learning rate schedule with optional warmup.

    The decay phase (after warmup) follows the cosine annealing formula:

    .. math::

        lr(t) = lr_{\text{stop}} + \frac{lr_0 - lr_{\text{stop}}}{2} \left(1 + \cos\left(\pi \frac{t}{T}\right)\right)

    where:
    - :math:`lr_0` is ``start_lr`` (learning rate at the start of decay phase)
    - :math:`lr_{\text{stop}}` is ``stop_lr`` (minimum learning rate)
    - :math:`t` is the step index within the decay phase
    - :math:`T = \text{num\_steps} - \text{warmup\_steps}` is the total
      number of decay steps

    Equivalently, using :math:`\alpha = lr_{\text{stop}} / lr_0`:

    .. math::

        lr(t) = lr_0 \cdot \left[\alpha + \frac{1}{2}(1 - \alpha) \left(1 + \cos\left(\pi \frac{t}{T}\right)\right)\right]
    """

    def __init__(
        self,
        start_lr: float,
        num_steps: int,
        stop_lr: float | None = None,
        stop_lr_ratio: float | None = None,
        warmup_steps: int = 0,
        warmup_ratio: float | None = None,
        warmup_start_factor: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Construct a cosine annealing learning rate schedule with optional warmup.

        Parameters
        ----------
        start_lr : float
            The learning rate at the start of the training (after warmup).
        stop_lr : float, optional
            The final learning rate at the end of training.
            Mutually exclusive with stop_lr_ratio.
        stop_lr_ratio : float, optional
            The ratio of stop_lr to start_lr.
            Mutually exclusive with stop_lr.
        num_steps : int
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
            If both stop_lr and stop_lr_ratio are provided, or neither is provided.
            If both warmup_steps and warmup_ratio are provided.
        """
        super().__init__(
            start_lr=start_lr,
            stop_lr=stop_lr,
            stop_lr_ratio=stop_lr_ratio,
            num_steps=num_steps,
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
        min_lr = self._start_lr * self.lr_min_factor
        # Use input dtype for floating point, or default to GLOBAL_NP_FLOAT_PRECISION for integers
        step_dtype = (
            step.dtype
            if np.issubdtype(step.dtype, np.floating)
            else GLOBAL_NP_FLOAT_PRECISION
        )
        # Handle decay_num_steps=0 (no training steps) - return start_lr
        if self.decay_num_steps == 0:
            return xp.full_like(step, self._start_lr, dtype=step_dtype)
        step_lr = self._start_lr * (
            self.lr_min_factor
            + 0.5
            * (1 - self.lr_min_factor)
            * (
                1
                + xp.cos(
                    xp.asarray(
                        xp.pi * (xp.astype(step, step_dtype) / self.decay_num_steps),
                        dtype=step_dtype,
                    )
                )
            )
        )
        # Clip to min_lr for steps beyond decay_num_steps
        step_lr = xp.where(step >= self.decay_num_steps, min_lr, step_lr)
        return step_lr
