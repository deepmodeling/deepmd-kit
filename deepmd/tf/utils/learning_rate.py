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
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)


class LearningRateSchedule:
    """
    TensorFlow wrapper for BaseLR.

    The learning rate is computed via :func:`tf.numpy_function`, which prevents
    TensorFlow from optimizing this operation in the graph. This overhead is
    typically negligible compared to forward/backward passes.

    Parameters
    ----------
    params : dict[str, Any]
        Learning rate configuration dictionary.
    """

    def __init__(self, params: dict[str, Any]) -> None:
        self._params = dict(params)
        self._base_lr: BaseLR | None = None

    def start_lr(self) -> float:
        """
        Get the starting learning rate.

        Returns
        -------
        float
            The starting learning rate.
        """
        return float(self._params["start_lr"])

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

    def build(self, global_step: tf.Tensor, num_steps: int) -> tf.Tensor:
        """
        Build a TensorFlow learning rate tensor.

        Parameters
        ----------
        global_step : tf.Tensor
            The global training step tensor.
        num_steps : int
            The total training steps.

        Returns
        -------
        tf.Tensor
            The learning rate tensor.
        """
        # === Step 1. Instantiate backend-agnostic schedule ===
        params = dict(self._params)
        params["num_steps"] = num_steps
        # Default to 'exp' type if not specified
        if "type" not in params:
            params["type"] = "exp"
        self._base_lr = BaseLR(**params)

        # === Step 2. Bind a numpy_function for runtime evaluation ===
        base_lr = self._base_lr

        def _lr_value(step: np.ndarray) -> np.ndarray:
            # Use GLOBAL_TF_FLOAT_PRECISION (float64) for learning rate,
            # consistent with energy precision in TF backend
            return np.asarray(
                base_lr.value(step),
                dtype=GLOBAL_TF_FLOAT_PRECISION.as_numpy_dtype,
            )

        lr = tf.numpy_function(
            _lr_value, [global_step], Tout=GLOBAL_TF_FLOAT_PRECISION, name="lr_schedule"
        )
        lr.set_shape(global_step.get_shape())
        return lr

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
        return self._base_lr.value(step)


# Backward compatibility alias
LearningRateExp = LearningRateSchedule

__all__ = [
    "LearningRateExp",
    "LearningRateSchedule",
]
