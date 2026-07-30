# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.tf.env import (
    op_module,
    tf,
)


class TestTabulateShapeValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.table = tf.constant(np.zeros((2, 12)), dtype=tf.float64)
        self.table_info = tf.constant([0, 1, 2, 1, 1, -1], dtype=tf.float64)
        self.em_x = tf.constant([[0.25, 0.5, 0.75]], dtype=tf.float64)
        self.em = tf.constant(np.zeros((1, 3, 4)), dtype=tf.float64)

    def test_rejects_short_two_embed(self) -> None:
        descriptor = op_module.tabulate_fusion_se_atten(
            self.table,
            self.table_info,
            self.em_x,
            self.em,
            tf.constant([[1.0]], dtype=tf.float64),
            last_layer_size=2,
            is_sorted=True,
        )
        with (
            self.assertRaisesRegex(
                tf.errors.InvalidArgumentError, "two_embed must be rank 2"
            ),
            tf.Session() as sess,
        ):
            sess.run(descriptor)

    def test_accepts_flattened_em_x_layout(self) -> None:
        descriptor = op_module.tabulate_fusion_se_a(
            self.table,
            self.table_info,
            tf.reshape(self.em_x, [-1, 1]),
            self.em,
            last_layer_size=2,
        )
        with tf.Session() as sess:
            self.assertEqual(sess.run(descriptor).shape, (1, 4, 2))

    def test_rejects_mismatched_em_x(self) -> None:
        descriptor = op_module.tabulate_fusion_se_a(
            self.table,
            self.table_info,
            self.em_x[:, :2],
            self.em,
            last_layer_size=2,
        )
        with (
            self.assertRaisesRegex(
                tf.errors.InvalidArgumentError, "em_x must be rank 2"
            ),
            tf.Session() as sess,
        ):
            sess.run(descriptor)

    def test_rejects_mismatched_gradient_shape(self) -> None:
        gradients = op_module.tabulate_fusion_se_a_grad(
            self.table,
            self.table_info,
            self.em_x,
            self.em,
            tf.zeros((1, 4, 1), dtype=tf.float64),
            tf.zeros((1, 4, 2), dtype=tf.float64),
        )
        with (
            self.assertRaisesRegex(
                tf.errors.InvalidArgumentError, "dy has an unexpected shape"
            ),
            tf.Session() as sess,
        ):
            sess.run(gradients)

    def test_rejects_short_table_info(self) -> None:
        descriptor = op_module.tabulate_fusion_se_a(
            self.table,
            self.table_info[:4],
            self.em_x,
            self.em,
            last_layer_size=2,
        )
        with (
            self.assertRaisesRegex(
                tf.errors.InvalidArgumentError, "table_info must contain"
            ),
            tf.Session() as sess,
        ):
            sess.run(descriptor)

    def test_rejects_short_table(self) -> None:
        descriptor = op_module.tabulate_fusion_se_a(
            self.table[:, :-1],
            self.table_info,
            self.em_x,
            self.em,
            last_layer_size=2,
        )
        with (
            self.assertRaisesRegex(
                tf.errors.InvalidArgumentError, "table does not contain enough"
            ),
            tf.Session() as sess,
        ):
            sess.run(descriptor)


if __name__ == "__main__":
    unittest.main()
