# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

if not tf.executing_eagerly():
    # TF disallow temporary eager execution
    raise RuntimeError(
        "Unfortunatly, jax2tf (requires eager execution) cannot be used with the "
        "TensorFlow backend (disables eager execution). "
        "If you are converting a model between different backends, "
        "considering converting to the `.dp` format first."
    )

tnp.experimental_enable_numpy_behavior()
