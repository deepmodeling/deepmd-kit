# SPDX-License-Identifier: LGPL-3.0-or-later
"""TensorFlow eager backend environment."""

from typing import (
    Any,
)

import tensorflow as tf

from deepmd._vendors import ndtensorflow as xp

if not tf.executing_eagerly():
    raise RuntimeError(
        "The tf2 backend requires TensorFlow eager execution. "
        "It cannot be imported after eager execution has been disabled."
    )

Array = xp.Array
xp.ndarray = xp.Array


def stop_gradient(value: Any) -> Any:
    """Stop gradients on TensorFlow-backed Array objects."""
    if isinstance(value, xp.Array):
        return xp.asarray(tf.stop_gradient(value.unwrap()))
    return xp.asarray(tf.stop_gradient(value))


__all__ = ["Array", "stop_gradient", "tf", "xp"]
