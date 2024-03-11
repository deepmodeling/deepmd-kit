# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.network import get_activation_fn as get_activation_fn_dp

from .common import (
    INSTALLED_PT,
    INSTALLED_TF,
    parameterized,
)

if INSTALLED_PT:
    from deepmd.pt.utils.utils import get_activation_fn as get_activation_fn_pt
    from deepmd.pt.utils.utils import (
        to_numpy_array,
        to_torch_tensor,
    )
if INSTALLED_TF:
    from deepmd.tf.common import get_activation_func as get_activation_fn_tf
    from deepmd.tf.env import (
        tf,
    )


@parameterized(
    (
        "Relu",
        "Relu6",
        "Softplus",
        "Sigmoid",
        "Tanh",
        "Gelu",
        "Gelu_tf",
        "Linear",
        "None",
    ),
)
class TestActivationFunctionConsistent(unittest.TestCase):
    def setUp(self):
        (self.activation,) = self.param
        self.random_input = np.random.default_rng().normal(scale=10, size=(10, 10))
        self.ref = get_activation_fn_dp(self.activation)(self.random_input)

    @unittest.skipUnless(INSTALLED_TF, "TensorFlow is not installed")
    def test_tf_consistent_with_ref(self):
        if INSTALLED_TF:
            place_holder = tf.placeholder(tf.float64, self.random_input.shape)
            t_test = get_activation_fn_tf(self.activation)(place_holder)
            with tf.Session() as sess:
                test = sess.run(t_test, feed_dict={place_holder: self.random_input})
            np.testing.assert_allclose(self.ref, test, atol=1e-10)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self):
        if INSTALLED_PT:
            test = to_numpy_array(
                get_activation_fn_pt(self.activation)(
                    to_torch_tensor(self.random_input)
                )
            )
            np.testing.assert_allclose(self.ref, test, atol=1e-10)
