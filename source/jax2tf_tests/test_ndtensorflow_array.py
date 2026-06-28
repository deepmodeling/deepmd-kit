# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import tensorflow as tf

from deepmd._vendors import ndtensorflow as xp


def test_dynamic_boolean_mask_indexing() -> None:
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None, 3], tf.float64),
            tf.TensorSpec([None, None], tf.bool),
        ]
    )
    def mask_values(array: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
        return xp.asarray(array)[mask].unwrap()

    array = tf.reshape(tf.range(18, dtype=tf.float64), (2, 3, 3))
    mask = tf.constant([[True, False, True], [False, True, False]])

    np.testing.assert_equal(
        mask_values(array, mask).numpy(),
        np.array(
            [
                [0.0, 1.0, 2.0],
                [6.0, 7.0, 8.0],
                [12.0, 13.0, 14.0],
            ]
        ),
    )


def test_empty_boolean_mask_indexing() -> None:
    out = xp.asarray(tf.constant([False]))[xp.asarray(tf.constant([], dtype=tf.bool))]

    np.testing.assert_equal(out.unwrap().numpy(), np.array([], dtype=bool))


def test_tile_uint16() -> None:
    out = xp.tile(xp.asarray(tf.constant(0, dtype=tf.uint16)), (2,))

    np.testing.assert_equal(out.unwrap().numpy(), np.array([0, 0], dtype=np.uint16))
