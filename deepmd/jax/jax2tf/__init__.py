# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf

from deepmd.env import (
    get_default_nthreads,
    set_default_nthreads,
)

if not tf.executing_eagerly():
    # TF disallow temporary eager execution
    raise RuntimeError(
        "Unfortunatly, jax2tf (requires eager execution) cannot be used with the "
        "TensorFlow backend (disables eager execution). "
        "If you are converting a model between different backends, "
        "considering converting to the `.dp` format first."
    )

set_default_nthreads()
inter_nthreads, intra_nthreads = get_default_nthreads()
tf.config.threading.set_inter_op_parallelism_threads(inter_nthreads)
tf.config.threading.set_intra_op_parallelism_threads(intra_nthreads)
