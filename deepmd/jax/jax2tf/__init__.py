# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

if not tf.executing_eagerly():
    # TF disallow temporary eager execution
    raise RuntimeError(
        "The TensorFlow SavedModel compatibility layer requires eager execution. "
        "It cannot be used with the TensorFlow v1 backend after eager execution "
        "has been disabled. "
        "If you are converting a model between different backends, "
        "consider converting to the `.dp` format first."
    )

tnp.experimental_enable_numpy_behavior()
