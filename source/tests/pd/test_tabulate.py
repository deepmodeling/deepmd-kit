# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import paddle

from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.tabulate import (
    unaggregated_dy2_dx,
    unaggregated_dy2_dx_s,
    unaggregated_dy_dx,
    unaggregated_dy_dx_s,
)
from deepmd.tf.env import (
    op_module,
    tf,
)


def setUpModule() -> None:
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

        self.y = np.tanh(self.xbar)

    def test_ops(self) -> None:
        dy_tf = op_module.unaggregated_dy_dx_s(
            tf.constant(self.y, dtype="double"),
            tf.constant(self.w, dtype="double"),
            tf.constant(self.xbar, dtype="double"),
            tf.constant(1),
        )

        dy_pd = unaggregated_dy_dx_s(
            paddle.to_tensor(self.y),
            self.w,
            paddle.to_tensor(self.xbar),
            1,
        )

        dy_tf_numpy = dy_tf.numpy()
        dy_pd_numpy = dy_pd.detach().cpu().numpy()

        np.testing.assert_almost_equal(dy_tf_numpy, dy_pd_numpy, decimal=10)

        dy2_tf = op_module.unaggregated_dy2_dx_s(
            tf.constant(self.y, dtype="double"),
            dy_tf,
            tf.constant(self.w, dtype="double"),
            tf.constant(self.xbar, dtype="double"),
            tf.constant(1),
        )

        dy2_pd = unaggregated_dy2_dx_s(
            paddle.to_tensor(self.y, place="cpu"),
            dy_pd,
            self.w,
            paddle.to_tensor(self.xbar, place="cpu"),
            1,
        )

        dy2_tf_numpy = dy2_tf.numpy()
        dy2_pd_numpy = dy2_pd.detach().cpu().numpy()

        np.testing.assert_almost_equal(dy2_tf_numpy, dy2_pd_numpy, decimal=10)

        dz_tf = op_module.unaggregated_dy_dx(
            tf.constant(self.y, dtype="double"),
            tf.constant(self.w, dtype="double"),
            dy_tf,
            tf.constant(self.xbar, dtype="double"),
            tf.constant(1),
        )

        dz_pd = unaggregated_dy_dx(
            paddle.to_tensor(self.y, place=env.DEVICE),
            self.w,
            dy_pd,
            paddle.to_tensor(self.xbar, place=env.DEVICE),
            1,
        )

        dz_tf_numpy = dz_tf.numpy()
        dz_pd_numpy = dz_pd.detach().cpu().numpy()

        np.testing.assert_almost_equal(dz_tf_numpy, dz_pd_numpy, decimal=10)

        dy2_tf = op_module.unaggregated_dy2_dx(
            tf.constant(self.y, dtype="double"),
            tf.constant(self.w, dtype="double"),
            dy_tf,
            dy2_tf,
            tf.constant(self.xbar, dtype="double"),
            tf.constant(1),
        )

        dy2_pd = unaggregated_dy2_dx(
            paddle.to_tensor(self.y, place=env.DEVICE),
            self.w,
            dy_pd,
            dy2_pd,
            paddle.to_tensor(self.xbar, place=env.DEVICE),
            1,
        )

        dy2_tf_numpy = dy2_tf.numpy()
        dy2_pd_numpy = dy2_pd.detach().cpu().numpy()

        np.testing.assert_almost_equal(dy2_tf_numpy, dy2_pd_numpy, decimal=10)


if __name__ == "__main__":
    unittest.main()
