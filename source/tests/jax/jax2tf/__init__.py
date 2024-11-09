# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf

from ...common import (
    DP_TEST_TF2_ONLY,
)

if DP_TEST_TF2_ONLY:
    # limit the number of threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
