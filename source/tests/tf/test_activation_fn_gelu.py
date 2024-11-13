# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.tf.common import (
    get_activation_func,
)
from deepmd.tf.env import (
    tf,
)
from deepmd.tf.utils.network import (
    embedding_net,
)


class TestGelu(tf.test.TestCase):
    def setUp(self) -> None:
        self.places = 6
        self.sess = self.cached_session().__enter__()
        self.inputs = tf.reshape(
            tf.constant([0.0, 1.0, 2.0, 3.0], dtype=tf.float64), [-1, 1]
        )
        self.refout = [
            [
                0.37703893,
                -0.38242253,
                -0.1862878,
                -0.23220415,
                2.28706995,
                -0.40754364,
                0.22086098,
                -0.2690335,
            ],
            [
                2.167494,
                0.72560347,
                0.99234317,
                0.50832127,
                5.20665818,
                0.58361587,
                1.57217107,
                0.67395218,
            ],
            [
                4.19655852,
                2.04779208,
                2.20239826,
                1.69247695,
                8.38305924,
                1.69006845,
                2.97176052,
                1.76098426,
            ],
            [
                6.21460216,
                3.52613278,
                3.39508271,
                2.817003,
                11.521799,
                2.91028145,
                4.41870371,
                2.82610791,
            ],
        ]

    def test_activation_function_gelu_custom(self) -> None:
        network_size = [2, 4, 8]
        out = embedding_net(
            self.inputs,
            network_size,
            tf.float64,
            activation_fn=get_activation_func("gelu"),
            name_suffix="gelu_custom",
            seed=1,
            uniform_seed=True,
        )
        self.sess.run(tf.global_variables_initializer())
        myout = self.sess.run(out)
        np.testing.assert_almost_equal(self.refout, myout, self.places)

    def test_activation_function_gelu_tensorflow(self) -> None:
        network_size = [2, 4, 8]
        out = embedding_net(
            self.inputs,
            network_size,
            tf.float64,
            activation_fn=get_activation_func("gelu_tf"),
            name_suffix="gelu_tensorflow",
            seed=1,
            uniform_seed=True,
        )
        self.sess.run(tf.global_variables_initializer())
        myout = self.sess.run(out)
        np.testing.assert_almost_equal(self.refout, myout, self.places)


if __name__ == "__main__":
    unittest.main()
