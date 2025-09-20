# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for detecting NaN values in loss during training."""

import logging
import math

log = logging.getLogger(__name__)


class LossNaNError(RuntimeError):
    """Exception raised when NaN is detected in total loss during training."""

    def __init__(self, step: int, total_loss: float) -> None:
        """Initialize the exception.

        Parameters
        ----------
        step : int
            The training step where NaN was detected
        total_loss : float
            The total loss value that contains NaN
        """
        self.step = step
        self.total_loss = total_loss
        message = (
            f"NaN detected in total loss at training step {step}: {total_loss}. "
            f"Training stopped to prevent wasting time with corrupted parameters. "
            f"This typically indicates unstable training conditions such as "
            f"learning rate too high, poor data quality, or numerical instability."
        )
        super().__init__(message)


def check_total_loss_nan(step: int, total_loss: float) -> None:
    """Check if the total loss contains NaN and raise an exception if found.

    This function is designed to be called during training after the total loss
    is computed and converted to a CPU float value.

    Parameters
    ----------
    step : int
        Current training step
    total_loss : float
        Total loss value to check for NaN

    Raises
    ------
    LossNaNError
        If the total loss contains NaN
    """
    if math.isnan(total_loss):
        log.error(f"NaN detected in total loss at step {step}: {total_loss}")
        raise LossNaNError(step, total_loss)
