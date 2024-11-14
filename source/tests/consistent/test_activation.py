# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest

import numpy as np

from deepmd.common import (
    VALID_ACTIVATION,
)
from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.dpmodel.utils.network import get_activation_fn as get_activation_fn_dp

from ..seed import (
    GLOBAL_SEED,
)
from .common import (
    INSTALLED_JAX,
    INSTALLED_PD,
    INSTALLED_PT,
    INSTALLED_TF,
    parameterized,
)

if INSTALLED_PT:
    from deepmd.pt.utils.utils import ActivationFn as ActivationFn_pt
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import (
        to_torch_tensor,
    )
if INSTALLED_TF:
    from deepmd.tf.common import get_activation_func as get_activation_fn_tf
    from deepmd.tf.env import (
        tf,
    )
if INSTALLED_JAX:
    from deepmd.jax.env import (
        jnp,
    )
if INSTALLED_PD:
    from deepmd.pd.utils.utils import ActivationFn as ActivationFn_pd
    from deepmd.pd.utils.utils import to_numpy_array as paddle_to_numpy
    from deepmd.pd.utils.utils import (
        to_paddle_tensor,
    )


@parameterized(
    tuple([x.capitalize() for x in VALID_ACTIVATION]),
)
class TestActivationFunctionConsistent(unittest.TestCase):
    def setUp(self) -> None:
        (self.activation,) = self.param
        self.random_input = np.random.default_rng(GLOBAL_SEED).normal(
            scale=10, size=(10, 10)
        )
        self.ref = get_activation_fn_dp(self.activation)(self.random_input)

    @unittest.skipUnless(INSTALLED_TF, "TensorFlow is not installed")
    def test_tf_consistent_with_ref(self) -> None:
        if INSTALLED_TF:
            place_holder = tf.placeholder(tf.float64, self.random_input.shape)
            t_test = get_activation_fn_tf(self.activation)(place_holder)
            with tf.Session() as sess:
                test = sess.run(t_test, feed_dict={place_holder: self.random_input})
            np.testing.assert_allclose(self.ref, test, atol=1e-10)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self) -> None:
        if INSTALLED_PT:
            test = torch_to_numpy(
                ActivationFn_pt(self.activation)(to_torch_tensor(self.random_input))
            )
            np.testing.assert_allclose(self.ref, test, atol=1e-10)

    @unittest.skipUnless(
        sys.version_info >= (3, 9), "array_api_strict doesn't support Python<=3.8"
    )
    def test_arary_api_strict(self) -> None:
        import array_api_strict as xp

        input = xp.asarray(self.random_input)
        test = get_activation_fn_dp(self.activation)(input)
        np.testing.assert_allclose(self.ref, to_numpy_array(test), atol=1e-10)

    @unittest.skipUnless(INSTALLED_JAX, "JAX is not installed")
    def test_jax_consistent_with_ref(self) -> None:
        input = jnp.from_dlpack(self.random_input)
        test = get_activation_fn_dp(self.activation)(input)
        self.assertTrue(isinstance(test, jnp.ndarray))
        np.testing.assert_allclose(self.ref, np.from_dlpack(test), atol=1e-10)

    @unittest.skipUnless(INSTALLED_PD, "Paddle is not installed")
    def test_pd_consistent_with_ref(self):
        if INSTALLED_PD:
            test = paddle_to_numpy(
                ActivationFn_pd(self.activation)(to_paddle_tensor(self.random_input))
            )
            np.testing.assert_allclose(self.ref, test, atol=1e-10)
