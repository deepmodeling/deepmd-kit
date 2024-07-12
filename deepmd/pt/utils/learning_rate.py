# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


class LearningRateExp:
    def __init__(
        self,
        start_lr,
        stop_lr,
        decay_steps,
        stop_steps,
        decay_rate=None,
        **kwargs,
    ):
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
        self.start_lr = start_lr
        default_ds = 100 if stop_steps // 10 > 100 else stop_steps // 100 + 1
        self.decay_steps = decay_steps
        if self.decay_steps >= stop_steps:
            self.decay_steps = default_ds
        self.decay_rate = np.exp(
            np.log(stop_lr / self.start_lr) / (stop_steps / self.decay_steps)
        )
        if decay_rate is not None:
            self.decay_rate = decay_rate
        self.min_lr = stop_lr

    def value(self, step):
        """Get the learning rate at the given step."""
        step_lr = self.start_lr * np.power(self.decay_rate, step // self.decay_steps)
        if step_lr < self.min_lr:
            step_lr = self.min_lr
        return step_lr
