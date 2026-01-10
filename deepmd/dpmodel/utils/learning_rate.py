# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from types import (
    ModuleType,
)
from typing import (
    Any,
    overload,
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
        self, start_lr: float, stop_lr: float, stop_steps: int, **kwargs: Any
    ) -> None:
        """
        Base class for learning rate schedules.

        Parameters
        ----------
        start_lr
            The initial learning rate.
        stop_lr
            The final learning rate.
        stop_steps
            The total training steps for learning rate scheduler.
        """
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.stop_steps = stop_steps

    @abstractmethod
    def value(self, step: int | Array) -> Array:
        """Get the learning rate at the given step."""
        # in optax, step will be a jnp.ndarray passed in JIT mode
        pass


@BaseLR.register("exp")
class LearningRateExp(BaseLR):
    def __init__(
        self,
        start_lr: float,
        stop_lr: float,
        decay_steps: int,
        stop_steps: int,
        decay_rate: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Construct an exponential-decayed learning rate.

        Parameters
        ----------
        start_lr
            The learning rate at the start of the training.
        stop_lr
            The desired learning rate at the end of the training.
            When decay_rate is explicitly set, this value will serve as
            the minimum learning rate during training. In other words,
            if the learning rate decays below stop_lr, stop_lr will be applied instead.
        decay_steps
            The learning rate is decaying every this number of training steps.
        stop_steps
            The total training steps for learning rate scheduler.
        decay_rate
            The decay rate for the learning rate.
            If provided, the decay rate will be set instead of
            calculating it through interpolation between start_lr and stop_lr.
        """
        super().__init__(start_lr, stop_lr, stop_steps, **kwargs)
        default_ds = 100 if stop_steps // 10 > 100 else stop_steps // 100 + 1
        self.decay_steps = decay_steps
        if self.decay_steps >= stop_steps:
            self.decay_steps = default_ds
        self.decay_rate = np.exp(
            np.log(stop_lr / self.start_lr) / (stop_steps / self.decay_steps)
        ).item()
        if decay_rate is not None:
            self.decay_rate = decay_rate
        self.min_lr = self.stop_lr

    def value(self, step: int | Array) -> Array:
        """Get the learning rate at the given step."""
        if not array_api_compat.is_array_api_obj(step):
            step = np.asarray(step)
        xp = array_api_compat.array_namespace(step)
        step_lr = self.start_lr * xp.pow(
            xp.asarray(self.decay_rate, device=array_api_compat.device(step)),
            xp.astype(step // self.decay_steps, xp.float64),
        )
        # the original implementation `if step_lr < self.min_lr:`
        # will cause a dynamic graph which is unsupported in JAX JIT
        step_lr = xp.clip(step_lr, self.min_lr, None)
        return step_lr


@BaseLR.register("cosine")
class LearningRateCosine(BaseLR):
    def __init__(
        self,
        start_lr: float,
        stop_lr: float,
        stop_steps: int,
        **kwargs: Any,
    ) -> None:
        """
        Defines a cosine annealing learning rate schedule.
        The learning rate starts at `start_lr` and gradually decreases to `stop_lr`
        following a cosine curve over the training steps.

        Parameters
        ----------
        start_lr
            The initial learning rate at the beginning of training.
        stop_lr
            The final learning rate at the end of training.
        stop_steps
            The total number of training steps over which the learning rate
            will be annealed from start_lr to stop_lr.
        """
        super().__init__(start_lr, stop_lr, stop_steps, **kwargs)
        self.lr_min_factor = stop_lr / start_lr

    def value(self, step: int | Array) -> Array:
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
                        xp.pi * (xp.astype(step, xp.float64) / self.stop_steps),
                        device=array_api_compat.device(step),
                    )
                )
            )
        )
        step_lr = xp.where(step >= self.stop_steps, min_lr, step_lr)
        return step_lr
