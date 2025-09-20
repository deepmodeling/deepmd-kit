# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf
import tensorflow.experimental.numpy as tnp


def phys2inter(
    coord: tnp.ndarray,
    cell: tnp.ndarray,
) -> tnp.ndarray:
    """Convert physical coordinates to internal(direct) coordinates.

    Parameters
    ----------
    coord : tnp.ndarray
        physical coordinates of shape [*, na, 3].
    cell : tnp.ndarray
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    inter_coord: tnp.ndarray
        the internal coordinates

    """
    rec_cell = tf.linalg.inv(cell)
    return tnp.matmul(coord, rec_cell)


def inter2phys(
    coord: tnp.ndarray,
    cell: tnp.ndarray,
) -> tnp.ndarray:
    """Convert internal(direct) coordinates to physical coordinates.

    Parameters
    ----------
    coord : tnp.ndarray
        internal coordinates of shape [*, na, 3].
    cell : tnp.ndarray
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    phys_coord: tnp.ndarray
        the physical coordinates

    """
    return tnp.matmul(coord, cell)


def normalize_coord(
    coord: tnp.ndarray,
    cell: tnp.ndarray,
) -> tnp.ndarray:
    """Apply PBC according to the atomic coordinates.

    Parameters
    ----------
    coord : tnp.ndarray
        original coordinates of shape [*, na, 3].
    cell : tnp.ndarray
        simulation cell shape [*, 3, 3].

    Returns
    -------
    wrapped_coord: tnp.ndarray
        wrapped coordinates of shape [*, na, 3].

    """
    icoord = phys2inter(coord, cell)
    icoord = tnp.remainder(icoord, 1.0)
    return inter2phys(icoord, cell)


def to_face_distance(
    cell: tnp.ndarray,
) -> tnp.ndarray:
    """Compute the to-face-distance of the simulation cell.

    Parameters
    ----------
    cell : tnp.ndarray
        simulation cell tensor of shape [*, 3, 3].

    Returns
    -------
    dist: tnp.ndarray
        the to face distances of shape [*, 3]

    """
    cshape = tf.shape(cell)
    dist = b_to_face_distance(tnp.reshape(cell, [-1, 3, 3]))
    return tnp.reshape(dist, tf.concat([cshape[:-2], [3]], axis=0))


def b_to_face_distance(cell):
    volume = tf.linalg.det(cell)
    c_yz = tf.linalg.cross(cell[:, 1, ...], cell[:, 2, ...])
    _h2yz = volume / tf.linalg.norm(c_yz, axis=-1)
    c_zx = tf.linalg.cross(cell[:, 2, ...], cell[:, 0, ...])
    _h2zx = volume / tf.linalg.norm(c_zx, axis=-1)
    c_xy = tf.linalg.cross(cell[:, 0, ...], cell[:, 1, ...])
    _h2xy = volume / tf.linalg.norm(c_xy, axis=-1)
    return tnp.stack([_h2yz, _h2zx, _h2xy], axis=1)
