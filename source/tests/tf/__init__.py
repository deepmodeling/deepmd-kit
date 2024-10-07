# SPDX-License-Identifier: LGPL-3.0-or-later
from ..common import (
    TEST_DEVICE,
)

# ensure GPU is used when testing GPU code
if TEST_DEVICE == "cuda":
    import tensorflow as tf

    assert tf.test.is_gpu_available()
