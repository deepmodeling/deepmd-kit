# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

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

        dy_pt = unaggregated_dy_dx_s(
            torch.from_numpy(self.y),
            self.w,
            torch.from_numpy(self.xbar),
            1,
        )

        dy_tf_numpy = dy_tf.numpy()
        dy_pt_numpy = dy_pt.detach().cpu().numpy()

        np.testing.assert_almost_equal(dy_tf_numpy, dy_pt_numpy, decimal=10)

        dy2_tf = op_module.unaggregated_dy2_dx_s(
            tf.constant(self.y, dtype="double"),
            dy_tf,
            tf.constant(self.w, dtype="double"),
            tf.constant(self.xbar, dtype="double"),
            tf.constant(1),
        )

        dy2_pt = unaggregated_dy2_dx_s(
            torch.from_numpy(self.y),
            dy_pt,
            self.w,
            torch.from_numpy(self.xbar),
            1,
        )

        dy2_tf_numpy = dy2_tf.numpy()
        dy2_pt_numpy = dy2_pt.detach().cpu().numpy()

        np.testing.assert_almost_equal(dy2_tf_numpy, dy2_pt_numpy, decimal=10)

        dz_tf = op_module.unaggregated_dy_dx(
            tf.constant(self.y, dtype="double"),
            tf.constant(self.w, dtype="double"),
            dy_tf,
            tf.constant(self.xbar, dtype="double"),
            tf.constant(1),
        )

        dz_pt = unaggregated_dy_dx(
            torch.from_numpy(self.y).to(env.DEVICE),
            self.w,
            dy_pt,
            torch.from_numpy(self.xbar).to(env.DEVICE),
            1,
        )

        dz_tf_numpy = dz_tf.numpy()
        dz_pt_numpy = dz_pt.detach().cpu().numpy()

        np.testing.assert_almost_equal(dz_tf_numpy, dz_pt_numpy, decimal=10)

        dy2_tf = op_module.unaggregated_dy2_dx(
            tf.constant(self.y, dtype="double"),
            tf.constant(self.w, dtype="double"),
            dy_tf,
            dy2_tf,
            tf.constant(self.xbar, dtype="double"),
            tf.constant(1),
        )

        dy2_pt = unaggregated_dy2_dx(
            torch.from_numpy(self.y).to(env.DEVICE),
            self.w,
            dy_pt,
            dy2_pt,
            torch.from_numpy(self.xbar).to(env.DEVICE),
            1,
        )

        dy2_tf_numpy = dy2_tf.numpy()
        dy2_pt_numpy = dy2_pt.detach().cpu().numpy()

        np.testing.assert_almost_equal(dy2_tf_numpy, dy2_pt_numpy, decimal=10)


if __name__ == "__main__":
    unittest.main()
