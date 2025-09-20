# SPDX-License-Identifier: LGPL-3.0-or-later
"""Utilities for detecting NaN values in loss during training."""

import logging
import math
from typing import (
    Any,
)

import numpy as np

log = logging.getLogger(__name__)


class LossNaNError(Exception):
    """Exception raised when NaN is detected in loss during training."""

    def __init__(self, step: int, loss_dict: dict[str, Any]) -> None:
        """Initialize the exception.

        Parameters
        ----------
        step : int
            The training step where NaN was detected
        loss_dict : dict[str, Any]
            Dictionary containing the loss values where NaN was found
        """
        self.step = step
        self.loss_dict = loss_dict
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message."""
        nan_losses = []
        for key, value in self.loss_dict.items():
            if self._is_nan(value):
                nan_losses.append(f"{key}={value}")

        message = (
            f"NaN detected in loss at training step {self.step}. "
            f"Training stopped to prevent wasting time with corrupted parameters. "
            f"NaN values found in: {', '.join(nan_losses)}. "
            f"This typically indicates unstable training conditions such as "
            f"learning rate too high, poor data quality, or numerical instability."
        )
        return message

    @staticmethod
    def _is_nan(value: Any) -> bool:
        """Check if a value is NaN."""
        if value is None:
            return False
        try:
            # Handle various tensor types and Python scalars
            if hasattr(value, "item"):
                # PyTorch/TensorFlow/PaddlePaddle tensor
                return math.isnan(value.item())
            elif isinstance(value, (int, float)):
                # Python scalar
                return math.isnan(value)
            elif isinstance(value, np.ndarray):
                # NumPy array
                return np.isnan(value).any()
            else:
                # Try to convert to float and check
                return math.isnan(float(value))
        except (TypeError, ValueError):
            # If we can't convert to float, assume it's not NaN
            return False


def check_loss_nan(step: int, loss_dict: dict[str, Any]) -> None:
    """Check if any loss values contain NaN and raise an exception if found.

    This function is designed to be called during training after loss values
    are computed and available on CPU, typically during the logging/display phase.

    Parameters
    ----------
    step : int
        Current training step
    loss_dict : dict[str, Any]
        Dictionary containing loss values to check for NaN

    Raises
    ------
    LossNaNError
        If any loss value contains NaN
    """
    nan_found = False
    for key, value in loss_dict.items():
        if LossNaNError._is_nan(value):
            nan_found = True
            log.error(f"NaN detected in {key} at step {step}: {value}")

    if nan_found:
        raise LossNaNError(step, loss_dict)


def check_single_loss_nan(step: int, loss_name: str, loss_value: Any) -> None:
    """Check if a single loss value contains NaN and raise an exception if found.

    Parameters
    ----------
    step : int
        Current training step
    loss_name : str
        Name/identifier of the loss
    loss_value : Any
        Loss value to check for NaN

    Raises
    ------
    LossNaNError
        If the loss value contains NaN
    """
    if LossNaNError._is_nan(loss_value):
        log.error(f"NaN detected in {loss_name} at step {step}: {loss_value}")
        raise LossNaNError(step, {loss_name: loss_value})
