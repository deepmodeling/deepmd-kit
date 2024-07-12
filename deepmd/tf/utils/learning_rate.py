# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import numpy as np

from deepmd.tf.env import (
    tf,
)


class LearningRateExp:
    r"""The exponentially decaying learning rate.

    The learning rate at step :math:`t` is given by

    .. math::

        \alpha(t) = \alpha_0 \lambda ^ { t / \tau }

    where :math:`\alpha` is the learning rate, :math:`\alpha_0` is the starting learning rate,
    :math:`\lambda` is the decay rate, and :math:`\tau` is the decay steps.

    Parameters
    ----------
    start_lr
            Starting learning rate :math:`\alpha_0`
    stop_lr
            Stop learning rate :math:`\alpha_1`
    decay_steps
            Learning rate decay every this number of steps :math:`\tau`
    decay_rate
            The decay rate :math:`\lambda`.
            If `stop_step` is provided in `build`, then it will be determined automatically and overwritten.
    """

    def __init__(
        self,
        start_lr: float,
        stop_lr: float = 5e-8,
        decay_steps: int = 5000,
        decay_rate: float = 0.95,
    ) -> None:
        """Constructor."""
        self.cd = {}
        self.cd["start_lr"] = start_lr
        self.cd["stop_lr"] = stop_lr
        self.cd["decay_steps"] = decay_steps
        self.cd["decay_rate"] = decay_rate
        self.start_lr_ = self.cd["start_lr"]

    def build(
        self, global_step: tf.Tensor, stop_step: Optional[int] = None
    ) -> tf.Tensor:
        """Build the learning rate.

        Parameters
        ----------
        global_step
            The tf Tensor prividing the global training step
        stop_step
            The stop step. If provided, the decay_rate will be determined automatically and overwritten.

        Returns
        -------
        learning_rate
            The learning rate
        """
        if stop_step is None:
            self.decay_steps_ = (
                self.cd["decay_steps"] if self.cd["decay_steps"] is not None else 5000
            )
            self.decay_rate_ = (
                self.cd["decay_rate"] if self.cd["decay_rate"] is not None else 0.95
            )
        else:
            self.stop_lr_ = (
                self.cd["stop_lr"] if self.cd["stop_lr"] is not None else 5e-8
            )
            default_ds = 100 if stop_step // 10 > 100 else stop_step // 100 + 1
            self.decay_steps_ = (
                self.cd["decay_steps"]
                if self.cd["decay_steps"] is not None
                else default_ds
            )
            if self.decay_steps_ >= stop_step:
                self.decay_steps_ = default_ds
            self.decay_rate_ = np.exp(
                np.log(self.stop_lr_ / self.start_lr_) / (stop_step / self.decay_steps_)
            )

        return tf.train.exponential_decay(
            self.start_lr_,
            global_step,
            self.decay_steps_,
            self.decay_rate_,
            staircase=True,
        )

    def start_lr(self) -> float:
        """Get the start lr."""
        return self.start_lr_

    def value(self, step: int) -> float:
        """Get the lr at a certain step."""
        return self.start_lr_ * np.power(self.decay_rate_, (step // self.decay_steps_))
