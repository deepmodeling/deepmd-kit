# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.network import (
    get_activation_fn,
)
from deepmd.pt.utils.tabulate import (
    unaggregated_dy2_dx,
    unaggregated_dy2_dx_s,
    unaggregated_dy_dx,
    unaggregated_dy_dx_s,
)
from deepmd.tf.env import (
    op_module,
    tf,
)

ACTIVATION_NAMES = {
    1: "tanh",
    2: "gelu",
    3: "relu",
    4: "relu6",
    5: "softplus",
    6: "sigmoid",
    7: "silu",
}


def get_activation_function(functype: int):
    """Get activation function corresponding to functype."""
    if functype not in ACTIVATION_NAMES:
        raise ValueError(f"Unknown functype: {functype}")

    return get_activation_fn(ACTIVATION_NAMES[functype])


def setUpModule() -> None:
    tf.reset_default_graph()
    tf.compat.v1.enable_eager_execution()


def tearDownModule() -> None:
    tf.compat.v1.disable_eager_execution()


class TestDPTabulate(unittest.TestCase):
    def setUp(self) -> None:
        self.w = np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            dtype=np.float64,
        )

        self.x = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]],
            dtype=np.float64,  # 4 x 3
        )

        self.b = np.array([[0.1], [0.2], [0.3], [0.4]], dtype=np.float64)  # 4 x 1

        self.xbar = np.matmul(self.x, self.w) + self.b  # 4 x 4

    def test_ops(self) -> None:
        """Test all activation functions using parameterized subtests."""
        for functype in ACTIVATION_NAMES.keys():
            activation_name = ACTIVATION_NAMES[functype]
            activation_fn = get_activation_function(functype)

            with self.subTest(activation=activation_name, functype=functype):
                self._test_single_activation(functype, activation_fn, activation_name)

    def _test_single_activation(
        self, functype: int, activation_fn, activation_name: str
    ) -> None:
        """Test tabulation operations for a specific activation function."""
        # Compute y using the specific activation function
        y = activation_fn(self.xbar)

        # Test unaggregated_dy_dx_s
        dy_tf = op_module.unaggregated_dy_dx_s(
            tf.constant(y, dtype="double"),
            tf.constant(self.w, dtype="double"),
            tf.constant(self.xbar, dtype="double"),
            tf.constant(functype),
        )

        dy_pt = unaggregated_dy_dx_s(
            y,
            self.w,
            self.xbar,
            functype,
        )

        dy_tf_numpy = dy_tf.numpy()
        dy_pt_numpy = np.asarray(dy_pt)

        np.testing.assert_almost_equal(
            dy_tf_numpy,
            dy_pt_numpy,
            decimal=10,
            err_msg=f"unaggregated_dy_dx_s failed for {activation_name}",
        )

        # Test unaggregated_dy2_dx_s
        dy2_tf = op_module.unaggregated_dy2_dx_s(
            tf.constant(y, dtype="double"),
            dy_tf,
            tf.constant(self.w, dtype="double"),
            tf.constant(self.xbar, dtype="double"),
            tf.constant(functype),
        )

        dy2_pt = unaggregated_dy2_dx_s(
            y,
            dy_pt,
            self.w,
            self.xbar,
            functype,
        )

        dy2_tf_numpy = dy2_tf.numpy()
        dy2_pt_numpy = np.asarray(dy2_pt)

        np.testing.assert_almost_equal(
            dy2_tf_numpy,
            dy2_pt_numpy,
            decimal=10,
            err_msg=f"unaggregated_dy2_dx_s failed for {activation_name}",
        )

        # Test unaggregated_dy_dx
        dz_tf = op_module.unaggregated_dy_dx(
            tf.constant(y, dtype="double"),
            tf.constant(self.w, dtype="double"),
            dy_tf,
            tf.constant(self.xbar, dtype="double"),
            tf.constant(functype),
        )

        dz_pt = unaggregated_dy_dx(
            y,
            self.w,
            dy_pt,
            self.xbar,
            functype,
        )

        dz_tf_numpy = dz_tf.numpy()
        dz_pt_numpy = np.asarray(dz_pt)

        np.testing.assert_almost_equal(
            dz_tf_numpy,
            dz_pt_numpy,
            decimal=10,
            err_msg=f"unaggregated_dy_dx failed for {activation_name}",
        )

        # Test unaggregated_dy2_dx
        dy2_tf = op_module.unaggregated_dy2_dx(
            tf.constant(y, dtype="double"),
            tf.constant(self.w, dtype="double"),
            dy_tf,
            dy2_tf,
            tf.constant(self.xbar, dtype="double"),
            tf.constant(functype),
        )

        dy2_pt = unaggregated_dy2_dx(
            y,
            self.w,
            dy_pt,
            dy2_pt,
            self.xbar,
            functype,
        )

        dy2_tf_numpy = dy2_tf.numpy()
        dy2_pt_numpy = np.asarray(dy2_pt)

        np.testing.assert_almost_equal(
            dy2_tf_numpy,
            dy2_pt_numpy,
            decimal=10,
            err_msg=f"unaggregated_dy2_dx failed for {activation_name}",
        )

    def test_linear_activation(self) -> None:
        """Test functype=0 (linear/none) with direct numpy expectations.

        TF custom ops don't support functype=0, so we validate the numpy
        derivative helpers and unaggregated tabulate ops directly.
        """
        from deepmd.utils.tabulate_math import (
            grad,
            grad_grad,
        )

        fn = get_activation_fn("linear")
        y = fn(self.xbar)

        # grad: f'(x) = 1 for identity
        dy_ana = grad(self.xbar, y, 0)
        np.testing.assert_allclose(dy_ana, np.ones_like(self.xbar), atol=1e-12)

        # grad_grad: f''(x) = 0 for identity
        dy2_ana = grad_grad(self.xbar, y, 0)
        np.testing.assert_allclose(dy2_ana, np.zeros_like(self.xbar), atol=1e-12)

        # Also verify unaggregated functions work with functype=0
        dy = unaggregated_dy_dx_s(y, self.w, self.xbar, 0)
        self.assertEqual(dy.shape, (4, 4))

        dy2 = unaggregated_dy2_dx_s(y, dy, self.w, self.xbar, 0)
        # Second derivative of identity is zero everywhere
        np.testing.assert_allclose(dy2, np.zeros_like(dy2), atol=1e-12)

    def test_softplus_activation_is_numerically_stable(self) -> None:
        """Test softplus tabulation helpers on large extrapolated inputs."""
        from deepmd.utils.tabulate_math import (
            grad,
            grad_grad,
        )

        xbar = np.array([[100.0, 500.0, 1000.0]], dtype=np.float64)

        with np.errstate(over="raise", invalid="raise"):
            y = get_activation_fn("softplus")(xbar)
            dy = grad(xbar, y, 5)
            dy2 = grad_grad(xbar, y, 5)

        np.testing.assert_allclose(y, xbar, atol=1e-12)
        np.testing.assert_allclose(dy, np.ones_like(xbar), atol=1e-12)
        np.testing.assert_allclose(dy2, np.zeros_like(xbar), atol=1e-12)

    def test_softplus_derivatives_match_finite_differences(self) -> None:
        """Test softplus derivatives against finite differences on both branches."""
        from deepmd.utils.tabulate_math import (
            grad,
            grad_grad,
        )

        fn = get_activation_fn("softplus")
        xbar = np.array([[-5.0, -0.5, 0.0, 0.5, 5.0]], dtype=np.float64)
        y = fn(xbar)

        dy = grad(xbar, y, 5)
        dy2 = grad_grad(xbar, y, 5)

        h_grad = 3e-5
        y_plus = fn(xbar + h_grad)
        y_minus = fn(xbar - h_grad)
        dy_fd = (y_plus - y_minus) / (2 * h_grad)

        h_grad2 = 3e-4
        y_plus = fn(xbar + h_grad2)
        y_minus = fn(xbar - h_grad2)
        dy2_fd = (y_plus - 2 * y + y_minus) / (h_grad2**2)

        np.testing.assert_allclose(dy, dy_fd, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(dy2, dy2_fd, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
