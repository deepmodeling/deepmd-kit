# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest
from importlib.util import (
    find_spec,
)

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
    INSTALLED_PT_EXPT,
    INSTALLED_TF,
    INSTALLED_TF2,
    parameterized_cases,
)

if INSTALLED_PT:
    from deepmd.pt.utils.utils import ActivationFn as ActivationFn_pt
    from deepmd.pt.utils.utils import to_numpy_array as torch_to_numpy
    from deepmd.pt.utils.utils import (
        to_torch_tensor,
    )
if INSTALLED_PT_EXPT:
    import torch

    from deepmd.pt_expt.utils.env import DEVICE as PT_EXPT_DEVICE
    from deepmd.pt_expt.utils.network import (
        _torch_activation,
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


ACTIVATION_FUNCTION_CURATED_CASES = tuple((x.capitalize(),) for x in VALID_ACTIVATION)


@parameterized_cases(*ACTIVATION_FUNCTION_CURATED_CASES)
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

    @unittest.skipUnless(INSTALLED_TF2, "TensorFlow 2 is not installed")
    def test_tf2_consistent_with_ref(self) -> None:
        from deepmd.tf2.common import (
            to_tensorflow_array,
        )

        input = to_tensorflow_array(self.random_input)
        test = get_activation_fn_dp(self.activation)(input)
        np.testing.assert_allclose(self.ref, to_numpy_array(test), atol=1e-7)

    def test_the_tf2_namespace_name_still_resolves(self) -> None:
        """Anchor the literal the tf2 dispatch compares against.

        ``xp_erf`` and its neighbours in ``deepmd/dpmodel/array_api.py`` select
        the TensorFlow path by comparing ``xp.__name__`` against the import
        path of the vendored namespace -- the idiom this tree uses in
        ``utils/nlist.py`` and ``utils/default_neighbor_list.py`` too, nine
        occurrences across three modules.

        A literal like that fails silently if the module is ever renamed or
        moved: the comparison simply stops matching, the code falls back to the
        NumPy round-trip, and the gradient is lost again with every
        forward-value test still passing. This asserts the name resolves to a
        real module, so a rename breaks a test instead of a derivative.

        Deliberately not gated on ``INSTALLED_TF2``: ``find_spec`` does not
        import the module, so this runs everywhere, including the ordinary runs
        where the tf2 cases skip. A guard that skips alongside the thing it
        guards would protect nothing.
        """
        self.assertIsNotNone(
            find_spec("deepmd._vendors.ndtensorflow"),
            "the namespace name that array_api.py dispatches on no longer "
            "resolves; the TensorFlow branches there are now dead code",
        )

    @unittest.skipUnless(INSTALLED_TF2, "TensorFlow 2 is not installed")
    def test_tf2_gradient_consistent_with_ref(self) -> None:
        """The derivative has to survive the dispatch, not only the value.

        A backend that falls through to a NumPy conversion still returns the
        right number, because the conversion happens after the value is
        computed -- but it detaches the term from the tape, so the derivative
        comes out wrong with nothing to show for it. Comparing forward values
        alone cannot see that, which is why this compares the gradient.

        The input is deliberately a narrow range rather than the wide random
        sample the value tests use: for the exact GELU the missing term is
        ``x * phi(x)``, which vanishes for large ``|x|`` and would hide the
        very defect this pins.

        The point count is even so that the grid straddles zero without
        landing on it. ``relu`` and ``relu6`` have a kink there, where the
        derivative does not exist: autodiff reports the subgradient 0 while a
        central difference reports 0.5, and neither is wrong. Comparing them
        at that point tests nothing about dispatch, which is what this is for.
        """
        from deepmd._vendors import ndtensorflow as ndtf
        from deepmd.tf2.env import (
            tf,
        )

        probe = np.linspace(-3.0, 3.0, 24)
        raw = tf.constant(probe, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(raw)
            out = get_activation_fn_dp(self.activation)(ndtf.asarray(raw))
            total = tf.reduce_sum(out.unwrap())
        grad = tape.gradient(total, raw)
        self.assertIsNotNone(
            grad, f"{self.activation} left no gradient path on the tf2 backend"
        )
        # central differences on the reference implementation
        eps = 1e-6
        ref = get_activation_fn_dp(self.activation)
        expected = (ref(probe + eps) - ref(probe - eps)) / (2 * eps)
        np.testing.assert_allclose(to_numpy_array(grad), expected, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(INSTALLED_TF2, "TensorFlow 2 is not installed")
    def test_tf2_activation_is_traceable_in_graph_mode(self) -> None:
        """Every activation must survive ``tf.function``.

        A NumPy conversion is refused outright on a graph tensor, so an
        activation that reaches one cannot be trained or frozen on tf2 at all.
        """
        from deepmd._vendors import ndtensorflow as ndtf
        from deepmd.tf2.env import (
            tf,
        )

        activation = self.activation

        @tf.function
        def traced(values):
            return get_activation_fn_dp(activation)(ndtf.asarray(values)).unwrap()

        traced_out = traced(tf.constant(self.random_input, dtype=tf.float64))
        np.testing.assert_allclose(self.ref, traced_out.numpy(), atol=1e-7)

    @unittest.skipUnless(INSTALLED_PD, "Paddle is not installed")
    def test_pd_consistent_with_ref(self):
        if INSTALLED_PD:
            test = paddle_to_numpy(
                ActivationFn_pd(self.activation)(to_paddle_tensor(self.random_input))
            )
            np.testing.assert_allclose(self.ref, test, atol=1e-10)

    @unittest.skipUnless(INSTALLED_PT_EXPT, "PyTorch Exportable is not installed")
    def test_pt_expt_consistent_with_ref(self) -> None:
        if INSTALLED_PT_EXPT:
            x = torch.tensor(
                self.random_input, dtype=torch.float64, device=PT_EXPT_DEVICE
            )
            test = _torch_activation(x, self.activation).detach().cpu().numpy()
            np.testing.assert_allclose(self.ref, test, atol=1e-10)


SILUT_VARIANT_CURATED_CASES = (
    ("silut",),  # default threshold 3.0
    ("silut:3.0",),  # explicit threshold 3.0
    ("silut:10.0",),  # large threshold
    ("custom_silu:5.0",),  # alias
)


@parameterized_cases(*SILUT_VARIANT_CURATED_CASES)
class TestSilutVariantsConsistent(unittest.TestCase):
    """Cross-backend consistency for silut with different thresholds."""

    def setUp(self) -> None:
        (self.activation,) = self.param
        # Parse threshold to build input that covers both branches
        threshold = (
            float(self.activation.split(":")[-1]) if ":" in self.activation else 3.0
        )
        rng = np.random.default_rng(GLOBAL_SEED)
        # Values below threshold (silu branch) and above threshold (tanh branch)
        below = rng.uniform(-threshold - 5, threshold - 0.1, size=(5, 10))
        above = rng.uniform(threshold + 0.1, threshold + 20, size=(5, 10))
        self.random_input = np.concatenate([below, above], axis=0)
        self.ref = get_activation_fn_dp(self.activation)(self.random_input)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch is not installed")
    def test_pt_consistent_with_ref(self) -> None:
        if INSTALLED_PT:
            test = torch_to_numpy(
                ActivationFn_pt(self.activation)(to_torch_tensor(self.random_input))
            )
            np.testing.assert_allclose(self.ref, test, atol=1e-10)

    @unittest.skipUnless(INSTALLED_PT_EXPT, "PyTorch Exportable is not installed")
    def test_pt_expt_consistent_with_ref(self) -> None:
        if INSTALLED_PT_EXPT:
            x = torch.tensor(
                self.random_input, dtype=torch.float64, device=PT_EXPT_DEVICE
            )
            test = _torch_activation(x, self.activation).detach().cpu().numpy()
            np.testing.assert_allclose(self.ref, test, atol=1e-10)
