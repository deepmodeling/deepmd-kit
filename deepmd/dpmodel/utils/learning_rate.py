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

    def value(self, step) -> np.float64:
        """Get the learning rate at the given step."""
        step_lr = self.start_lr * np.power(self.decay_rate, step // self.decay_steps)
        if step_lr < self.min_lr:
            step_lr = self.min_lr
        return step_lr

class LearningRateWSD:
    def __init__(
        self,
        start_lr,
        stop_lr,
        stop_steps,
        decay_mode="85:10:5",  # stable-decay-stable
        **kwargs,
    ) -> None:
        self.start_lr = start_lr
        self.stop_lr = stop_lr
        self.stop_steps = stop_steps
        self.decay_mode = [float(ii) for ii in decay_mode.split(":")]
        assert len(self.decay_mode) == 3
        self.decay_start_rate = self.decay_mode[0] / sum(self.decay_mode)
        self.decay_end_rate = (self.decay_mode[0] + self.decay_mode[1]) / sum(
            self.decay_mode
        )
    def value(self, step) -> np.float64:
        if step < self.decay_start_rate * self.stop_steps:
            return self.start_lr
        elif step >= self.decay_end_rate * self.stop_steps:
            return self.stop_lr
        else:
            # linear decay
            decay_rate = (self.start_lr - self.stop_lr) / (
                self.decay_end_rate * self.stop_steps
                - self.decay_start_rate * self.stop_steps
            )
            return self.start_lr - decay_rate * (
                step - self.decay_start_rate * self.stop_steps
            )