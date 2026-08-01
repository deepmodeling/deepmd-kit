# SPDX-License-Identifier: LGPL-3.0-or-later
"""Neighbor-list helpers for the JAX/jax2tf SavedModel wrapper.

These routines mirror the backend-independent neighbor-list logic, but keep it
expressed directly as TensorFlow graph code. During SavedModel export the input
sizes are symbolic tensors; using ndtensorflow array-api helpers here can route
dynamic ``shape``/``size`` checks through Python control flow before AutoGraph
has converted them. That breaks tracing and also makes it easy to bypass the
jax2tf/XlaCallModule export path.
"""

import tensorflow as tf

from .region import (
    to_face_distance,
)


def tf_take_along_axis(params: tf.Tensor, indices: tf.Tensor, axis: int) -> tf.Tensor:
    return tf.gather(params, indices, batch_dims=axis)


def build_neighbor_list(
    coord: tf.Tensor,
    atype: tf.Tensor,
    nloc: int,
    rcut: float,
    sel: int | list[int],
    distinguish_types: bool = True,
) -> tf.Tensor:
    """Build neighbor list for a single frame. Keeps nsel neighbors."""
    batch_size = tf.shape(coord)[0]
    coord = tf.reshape(coord, (batch_size, -1))
    nall = tf.shape(coord)[1] // 3
    # Fill virtual atoms with large coords so they are not neighbors of any
    # real atom.
    if tf.size(coord) > 0:
        xmax = tf.reduce_max(coord) + tf.cast(2.0 * rcut, coord.dtype)
    else:
        xmax = tf.cast(2.0 * rcut, coord.dtype)
    is_vir = atype < 0
    coord1 = tf.where(
        is_vir[:, :, None], xmax, tf.reshape(coord, (batch_size, nall, 3))
    )
    coord1 = tf.reshape(coord1, (batch_size, nall * 3))
    if isinstance(sel, int):
        sel = [sel]
    nsel = sum(sel)
    coord0 = coord1[:, : nloc * 3]
    diff = (
        tf.reshape(coord1, [batch_size, -1, 3])[:, None, :, :]
        - tf.reshape(coord0, [batch_size, -1, 3])[:, :, None, :]
    )
    rr = tf.linalg.norm(diff, axis=-1)
    # If central atom has two zero distances, sorting sometimes can not exclude
    # itself.
    rr -= tf.eye(nloc, nall, dtype=diff.dtype)[None, :, :]
    nlist = tf.cast(tf.argsort(rr, axis=-1), tf.int64)
    rr = tf.sort(rr, axis=-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]
    nnei = tf.shape(rr)[2]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = tf.concat(
            [
                rr,
                tf.ones([batch_size, nloc, nsel - nnei], dtype=rr.dtype)
                + tf.cast(rcut, rr.dtype),
            ],
            axis=-1,
        )
        nlist = tf.concat(
            [nlist, tf.ones([batch_size, nloc, nsel - nnei], dtype=nlist.dtype)],
            axis=-1,
        )
    nlist = tf.where(
        tf.logical_or((rr > tf.cast(rcut, rr.dtype)), is_vir[:, :nloc, None]),
        tf.fill(tf.shape(nlist), tf.cast(-1, nlist.dtype)),
        nlist,
    )

    if distinguish_types:
        return nlist_distinguish_types(nlist, atype, sel)
    return nlist


def nlist_distinguish_types(
    nlist: tf.Tensor,
    atype: tf.Tensor,
    sel: list[int],
) -> tf.Tensor:
    """Given a nlist that does not distinguish atom types, return one that does."""
    nloc = tf.shape(nlist)[1]
    ret_nlist = []
    tmp_atype = tf.tile(atype[:, None, :], [1, nloc, 1])
    mask = nlist == -1
    tnlist_0 = tf.where(mask, tf.zeros_like(nlist), nlist)
    tnlist = tf_take_along_axis(tmp_atype, tnlist_0, axis=2)
    tnlist = tf.where(
        mask, tf.fill(tf.shape(tnlist), tf.cast(-1, tnlist.dtype)), tnlist
    )
    for ii, ss in enumerate(sel):
        pick_mask = tf.cast(tnlist == ii, tf.int32)
        sorted_indices = tf.argsort(-pick_mask, stable=True, axis=-1)
        pick_mask_sorted = -tf.sort(-pick_mask, axis=-1)
        inlist = tf_take_along_axis(nlist, sorted_indices, axis=2)
        inlist = tf.where(
            ~tf.cast(pick_mask_sorted, tf.bool),
            tf.fill(tf.shape(inlist), tf.cast(-1, inlist.dtype)),
            inlist,
        )
        ret_nlist.append(inlist[..., :ss])
    ret = tf.concat(ret_nlist, axis=-1)
    return ret


def tf_outer(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    return tf.einsum("i,j->ij", a, b)


def extend_coord_with_ghosts(
    coord: tf.Tensor,
    atype: tf.Tensor,
    cell: tf.Tensor,
    rcut: float,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Extend atom coordinates by appending periodic images."""
    atype_shape = tf.shape(atype)
    nf, nloc = atype_shape[0], atype_shape[1]
    # int64 for index
    aidx = tf.range(nloc, dtype=tf.int64)
    aidx = tf.tile(aidx[None, :], [nf, 1])
    if tf.shape(cell)[-1] == 0:
        nall = nloc
        extend_coord = coord
        extend_atype = atype
        extend_aidx = aidx
    else:
        coord = tf.reshape(coord, (nf, nloc, 3))
        cell = tf.reshape(cell, (nf, 3, 3))
        to_face = to_face_distance(cell)
        nbuff = tf.cast(tf.math.ceil(tf.cast(rcut, to_face.dtype) / to_face), tf.int64)
        nbuff = tf.reduce_max(nbuff, axis=0)
        xi = tf.range(-nbuff[0], nbuff[0] + 1, 1, dtype=tf.int64)
        yi = tf.range(-nbuff[1], nbuff[1] + 1, 1, dtype=tf.int64)
        zi = tf.range(-nbuff[2], nbuff[2] + 1, 1, dtype=tf.int64)
        xyz = tf_outer(xi, tf.constant([1, 0, 0], dtype=tf.int64))[:, None, None, :]
        xyz = (
            xyz + tf_outer(yi, tf.constant([0, 1, 0], dtype=tf.int64))[None, :, None, :]
        )
        xyz = (
            xyz + tf_outer(zi, tf.constant([0, 0, 1], dtype=tf.int64))[None, None, :, :]
        )
        xyz = tf.reshape(xyz, (-1, 3))
        xyz = tf.cast(xyz, coord.dtype)
        shift_idx = tf.gather(xyz, tf.argsort(tf.linalg.norm(xyz, axis=1)), axis=0)
        ns = tf.shape(shift_idx)[0]
        nall = ns * nloc
        shift_vec = tf.einsum("sd,fdk->fsk", shift_idx, cell)
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        extend_atype = tf.tile(atype[:, :, None], [1, ns, 1])
        extend_aidx = tf.tile(aidx[:, :, None], [1, ns, 1])

    return (
        tf.reshape(extend_coord, (nf, nall * 3)),
        tf.reshape(extend_atype, (nf, nall)),
        tf.reshape(extend_aidx, (nf, nall)),
    )
