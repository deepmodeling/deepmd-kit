# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import numpy as np

from deepmd.env import (
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
    
class LearningRateCos:
    r"""The cosine decaying learning rate.

  The function returns the decayed learning rate.  It is computed as:
  ```python
  global_step = min(global_step, decay_steps)
  cosine_decay = 0.5 * (1 + cos(pi * global_step / decay_steps))
  decayed = (1 - alpha) * cosine_decay + alpha
  decayed_learning_rate = learning_rate * decayed
  ```

    Parameters
    ----------
    start_lr
            Starting learning rate 
    stop_lr
            Minimum learning rate value as a fraction of learning_rate.
    decay_steps
            Number of steps to decay over.
    """

    def __init__(
        self,
        start_lr: float,
        stop_lr: float = 5e-8,
        decay_steps: int = 100000,
    ) -> None:
        """Constructor."""
        self.cd = {}
        self.cd["start_lr"] = start_lr
        self.cd["stop_lr"] = stop_lr
        self.cd["decay_steps"] = decay_steps
        self.start_lr_ = self.cd["start_lr"]
        self.alpha_ = self.cd["stop_lr"]/self.cd["start_lr"]

    def build(
        self, global_step: tf.Tensor, stop_step: Optional[int] = None
    ) -> tf.Tensor:
        """Build the learning rate.

        Parameters
        ----------
        global_step
            The tf Tensor prividing the global training step
        stop_step
            The stop step.

        Returns
        -------
        learning_rate
            The learning rate
        """
        if stop_step is None:
            self.decay_steps_ = (
                self.cd["decay_steps"] if self.cd["decay_steps"] is not None else 100000
            )
        else:
            self.stop_lr_ = (
                self.cd["stop_lr"] if self.cd["stop_lr"] is not None else 5e-8
            )
            self.decay_steps_ = (
                self.cd["decay_steps"]
                if self.cd["decay_steps"] is not None
                else stop_step
            )

        return tf.train.cosine_decay(
            self.start_lr_,
            global_step,
            self.decay_steps_,
            self.alpha_,
            name="cosine",
        )

    def start_lr(self) -> float:
        """Get the start lr."""
        return self.start_lr_

    def value(self, step: int) -> float:
        """Get the lr at a certain step."""
        step = min(step, self.decay_steps_)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps_))
        decayed = (1 - self.alpha_) * cosine_decay + self.alpha_
        decayed_learning_rate = self.start_lr_ * decayed
        return decayed_learning_rate


class LearningRateCosRestarts:
    r"""The cosine decaying restart learning rate.

  The function returns the cosine decayed learning rate while taking into account
  possible warm restarts.
  ```

    Parameters
    ----------
    start_lr
            Starting learning rate 
    stop_lr
            Minimum learning rate value as a fraction of learning_rate.
    decay_steps
            Number of steps to decay over.
    """

    def __init__(
        self,
        start_lr: float,
        stop_lr: float = 5e-8,
        decay_steps: int = 10000,
    ) -> None:
        """Constructor."""
        self.cd = {}
        self.cd["start_lr"] = start_lr
        self.cd["stop_lr"] = stop_lr
        self.cd["decay_steps"] = decay_steps
        self.start_lr_ = self.cd["start_lr"]
        self.alpha_ = self.cd["stop_lr"]/self.cd["start_lr"]

    def build(
        self, global_step: tf.Tensor, stop_step: Optional[int] = None
    ) -> tf.Tensor:
        """Build the learning rate.

        Parameters
        ----------
        global_step
            The tf Tensor prividing the global training step
        stop_step
            The stop step.

        Returns
        -------
        learning_rate
            The learning rate
        """
        if stop_step is None:
            self.decay_steps_ = (
                self.cd["decay_steps"] if self.cd["decay_steps"] is not None else 10000
            )
        else:
            self.stop_lr_ = (
                self.cd["stop_lr"] if self.cd["stop_lr"] is not None else 5e-8
            )
            self.decay_steps_ = (
                self.cd["decay_steps"]
                if self.cd["decay_steps"] is not None
                else stop_step
            )

 

        return tf.train.cosine_decay_restarts(
            learning_rate=self.start_lr_,
            global_step=global_step,
            first_decay_steps=self.decay_steps_,
            alpha=self.alpha_,
            name="cosinerestart",
        )

    def start_lr(self) -> float:
        """Get the start lr."""
        return self.start_lr_

    def value(self, step: int) -> float:
        """Get the lr at a certain step. Need to revise later"""
        step = min(step, self.decay_steps_)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * step / self.decay_steps_))
        decayed = (1 - self.alpha_) * cosine_decay + self.alpha_
        decayed_learning_rate = self.start_lr_ * decayed
        return decayed_learning_rate
