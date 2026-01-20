# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.utils.network import (
    get_activation_fn,
)
from deepmd.pt.utils import (
    env,
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
            torch.from_numpy(y),
            self.w,
            torch.from_numpy(self.xbar),
            functype,
        )

        dy_tf_numpy = dy_tf.numpy()
        dy_pt_numpy = dy_pt.detach().cpu().numpy()

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
            torch.from_numpy(y),
            dy_pt,
            self.w,
            torch.from_numpy(self.xbar),
            functype,
        )

        dy2_tf_numpy = dy2_tf.numpy()
        dy2_pt_numpy = dy2_pt.detach().cpu().numpy()

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
            torch.from_numpy(y).to(env.DEVICE),
            self.w,
            dy_pt,
            torch.from_numpy(self.xbar).to(env.DEVICE),
            functype,
        )

        dz_tf_numpy = dz_tf.numpy()
        dz_pt_numpy = dz_pt.detach().cpu().numpy()

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
            torch.from_numpy(y).to(env.DEVICE),
            self.w,
            dy_pt,
            dy2_pt,
            torch.from_numpy(self.xbar).to(env.DEVICE),
            functype,
        )

        dy2_tf_numpy = dy2_tf.numpy()
        dy2_pt_numpy = dy2_pt.detach().cpu().numpy()

        np.testing.assert_almost_equal(
            dy2_tf_numpy,
            dy2_pt_numpy,
            decimal=10,
            err_msg=f"unaggregated_dy2_dx failed for {activation_name}",
        )


if __name__ == "__main__":
    unittest.main()
