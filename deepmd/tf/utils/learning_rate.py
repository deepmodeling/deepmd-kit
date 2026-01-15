# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.utils.learning_rate import (
    BaseLR,
)
from deepmd.tf.env import (
    tf,
)


class LearningRateSchedule:
    """
    TensorFlow wrapper for BaseLR.

    Parameters
    ----------
    params : dict[str, Any]
        Learning rate configuration dictionary.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        # === Step 1. Store configuration ===
        self._params = dict(params)
        if "start_lr" not in self._params:
            raise ValueError("start_lr must be provided")
        self._start_lr = float(self._params["start_lr"])
        self._base_lr: BaseLR | None = None

    def start_lr(self) -> float:
        """
        Get the starting learning rate.

        Returns
        -------
        float
            The starting learning rate.
        """
        return self._start_lr

    @property
    def base_lr(self) -> BaseLR:
        """
        Get the built BaseLR instance.

        Returns
        -------
        BaseLR
            The built learning rate schedule.

        Raises
        ------
        RuntimeError
            If the schedule has not been built.
        """
        if self._base_lr is None:
            raise RuntimeError("Learning rate schedule is not built yet.")
        return self._base_lr

    def build(self, global_step: tf.Tensor, stop_steps: int) -> tf.Tensor:
        """
        Build a TensorFlow learning rate tensor.

        Parameters
        ----------
        global_step : tf.Tensor
            The global training step tensor.
        stop_steps : int
            The total training steps.

        Returns
        -------
        tf.Tensor
            The learning rate tensor.
        """
        # === Step 1. Instantiate backend-agnostic schedule ===
        params = dict(self._params)
        params["stop_steps"] = stop_steps
        # Default to 'exp' type if not specified
        if "type" not in params:
            params["type"] = "exp"
        self._base_lr = BaseLR(**params)

        # === Step 2. Bind a numpy_function for runtime evaluation ===
        def _lr_value(step: np.ndarray) -> np.ndarray:
            return np.asarray(self._base_lr.value(step), dtype=np.float64)

        lr = tf.numpy_function(
            _lr_value, [global_step], Tout=tf.float64, name="lr_schedule"
        )
        lr.set_shape(global_step.get_shape())
        return tf.cast(lr, tf.float32)

    def value(self, step: int) -> float:
        """
        Get the learning rate at the given step.

        Parameters
        ----------
        step : int
            The step index.

        Returns
        -------
        float
            The learning rate value.

        Raises
        ------
        RuntimeError
            If the schedule has not been built.
        """
        if self._base_lr is None:
            raise RuntimeError("Learning rate schedule is not built yet.")
        return float(np.asarray(self._base_lr.value(step)))


__all__ = [
    "LearningRateSchedule",
]
