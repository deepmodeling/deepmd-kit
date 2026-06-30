# SPDX-License-Identifier: LGPL-3.0-or-later
"""TensorFlow geometry helpers used while exporting JAX models through jax2tf.

Keep these helpers free of TensorFlow eager array wrappers. They run inside the
SavedModel tracing path for ``.savedmodel`` and should stay as small, plain TF
graph functions that AutoGraph and the TensorFlow serializer can inspect.
"""

import tensorflow as tf


def phys2inter(
    coord: tf.Tensor,
    cell: tf.Tensor,
) -> tf.Tensor:
    """Convert physical coordinates to internal coordinates."""
    rec_cell = tf.linalg.inv(cell)
    return tf.matmul(coord, rec_cell)


def inter2phys(
    coord: tf.Tensor,
    cell: tf.Tensor,
) -> tf.Tensor:
    """Convert internal coordinates to physical coordinates."""
    return tf.matmul(coord, cell)


def normalize_coord(
    coord: tf.Tensor,
    cell: tf.Tensor,
) -> tf.Tensor:
    """Apply PBC according to the atomic coordinates."""
    icoord = phys2inter(coord, cell)
    icoord = tf.math.floormod(icoord, tf.cast(1.0, icoord.dtype))
    return inter2phys(icoord, cell)


def to_face_distance(
    cell: tf.Tensor,
) -> tf.Tensor:
    """Compute the to-face-distance of the simulation cell."""
    cshape = tf.shape(cell)
    dist = b_to_face_distance(tf.reshape(cell, [-1, 3, 3]))
    return tf.reshape(dist, tf.concat([cshape[:-2], [3]], axis=0))


def b_to_face_distance(cell: tf.Tensor) -> tf.Tensor:
    volume = tf.abs(tf.linalg.det(cell))
    c_yz = tf.linalg.cross(cell[:, 1, ...], cell[:, 2, ...])
    h2yz = volume / tf.linalg.norm(c_yz, axis=-1)
    c_zx = tf.linalg.cross(cell[:, 2, ...], cell[:, 0, ...])
    h2zx = volume / tf.linalg.norm(c_zx, axis=-1)
    c_xy = tf.linalg.cross(cell[:, 0, ...], cell[:, 1, ...])
    h2xy = volume / tf.linalg.norm(c_xy, axis=-1)
    return tf.stack([h2yz, h2zx, h2xy], axis=1)
