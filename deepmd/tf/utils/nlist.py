# SPDX-License-Identifier: LGPL-3.0-or-later
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)
from deepmd.tf.utils.region import (
    to_face_distance,
)


def extend_coord_with_ghosts(
    coord: tf.Tensor,
    atype: tf.Tensor,
    cell: tf.Tensor,
    rcut: float,
    pbc: tf.Tensor,
):
    """Extend the coordinates of the atoms by appending peridoc images.
    The number of images is large enough to ensure all the neighbors
    within rcut are appended.

    Parameters
    ----------
    coord : tf.Tensor
        original coordinates of shape [-1, nloc*3].
    atype : tf.Tensor
        atom type of shape [-1, nloc].
    cell : tf.Tensor
        simulation cell tensor of shape [-1, 9].
    rcut : float
        the cutoff radius
    pbc : tf.Tensor
        whether the simulation cell is periodic or not

    Returns
    -------
    extended_coord: tf.Tensor
        extended coordinates of shape [-1, nall*3].
    extended_atype: tf.Tensor
        extended atom type of shape [-1, nall].
    index_mapping: tf.Tensor
        maping extended index to the local index

    """
    nf = tf.shape(atype)[0]
    nloc = tf.shape(atype)[1]
    aidx = tf.tile(tf.expand_dims(tf.range(nloc), 0), [nf, 1])

    def extend_coord_with_ghosts_nopbc(coord, atype, cell):
        return coord, atype, aidx, nloc

    def extend_coord_with_ghosts_pbc(coord, atype, cell):
        coord = tf.reshape(coord, [nf, nloc, 3])
        cell = tf.reshape(cell, [nf, 3, 3])
        # nf x 3
        to_face = to_face_distance(cell)
        # nf x 3
        # *2: ghost copies on + and - directions
        # +1: central cell
        nbuff = tf.cast(tf.math.ceil(rcut / to_face), tf.int32)
        # 3
        nbuff = tf.reduce_max(nbuff, axis=0)
        xi = tf.range(-nbuff[0], nbuff[0] + 1, 1)
        yi = tf.range(-nbuff[1], nbuff[1] + 1, 1)
        zi = tf.range(-nbuff[2], nbuff[2] + 1, 1)
        xyz = tf.reshape(xi, [-1, 1, 1, 1]) * tf.constant([1, 0, 0], dtype=tf.int32)
        xyz = xyz + tf.reshape(yi, [1, -1, 1, 1]) * tf.constant(
            [0, 1, 0], dtype=tf.int32
        )
        xyz = xyz + tf.reshape(zi, [1, 1, -1, 1]) * tf.constant(
            [0, 0, 1], dtype=tf.int32
        )
        xyz = tf.reshape(xyz, [-1, 3])
        # ns x 3
        shift_idx = tf.gather(
            xyz, tf.argsort(tf.norm(tf.cast(xyz, GLOBAL_TF_FLOAT_PRECISION), axis=1))
        )
        ns = tf.shape(shift_idx)[0]
        nall = ns * nloc
        # nf x ns x 3
        shift_vec = tf.einsum(
            "sd,fdk->fsk", tf.cast(shift_idx, GLOBAL_TF_FLOAT_PRECISION), cell
        )
        # nf x ns x nloc x 3
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        # nf x ns x nloc
        extend_atype = tf.tile(tf.expand_dims(atype, -2), [1, ns, 1])
        # nf x ns x nloc
        extend_aidx = tf.tile(tf.expand_dims(aidx, -2), [1, ns, 1])
        return extend_coord, extend_atype, extend_aidx, nall

    extend_coord, extend_atype, extend_aidx, nall = tf.cond(
        pbc,
        lambda: extend_coord_with_ghosts_pbc(coord, atype, cell),
        lambda: extend_coord_with_ghosts_nopbc(coord, atype, cell),
    )

    return (
        tf.reshape(extend_coord, [nf, nall * 3]),
        tf.reshape(extend_atype, [nf, nall]),
        tf.reshape(extend_aidx, [nf, nall]),
    )
