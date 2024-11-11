# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Union,
)

import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from .region import (
    to_face_distance,
)


## translated from torch implementation by chatgpt
def build_neighbor_list(
    coord: tnp.ndarray,
    atype: tnp.ndarray,
    nloc: int,
    rcut: float,
    sel: Union[int, list[int]],
    distinguish_types: bool = True,
) -> tnp.ndarray:
    """Build neighbor list for a single frame. keeps nsel neighbors.

    Parameters
    ----------
    coord : tnp.ndarray
        exptended coordinates of shape [batch_size, nall x 3]
    atype : tnp.ndarray
        extended atomic types of shape [batch_size, nall]
        type < 0 the atom is treat as virtual atoms.
    nloc : int
        number of local atoms.
    rcut : float
        cut-off radius
    sel : int or list[int]
        maximal number of neighbors (of each type).
        if distinguish_types==True, nsel should be list and
        the length of nsel should be equal to number of
        types.
    distinguish_types : bool
        distinguish different types.

    Returns
    -------
    neighbor_list : tnp.ndarray
        Neighbor list of shape [batch_size, nloc, nsel], the neighbors
        are stored in an ascending order. If the number of
        neighbors is less than nsel, the positions are masked
        with -1. The neighbor list of an atom looks like
        |------ nsel ------|
        xx xx xx xx -1 -1 -1
        if distinguish_types==True and we have two types
        |---- nsel[0] -----| |---- nsel[1] -----|
        xx xx xx xx -1 -1 -1 xx xx xx -1 -1 -1 -1
        For virtual atoms all neighboring positions are filled with -1.

    """
    batch_size = tf.shape(coord)[0]
    coord = tnp.reshape(coord, (batch_size, -1))
    nall = tf.shape(coord)[1] // 3
    # fill virtual atoms with large coords so they are not neighbors of any
    # real atom.
    if tf.size(coord) > 0:
        xmax = tnp.max(coord) + 2.0 * rcut
    else:
        xmax = tf.cast(2.0 * rcut, coord.dtype)
    # nf x nall
    is_vir = atype < 0
    coord1 = tnp.where(
        is_vir[:, :, None], xmax, tnp.reshape(coord, (batch_size, nall, 3))
    )
    coord1 = tnp.reshape(coord1, (batch_size, nall * 3))
    if isinstance(sel, int):
        sel = [sel]
    nsel = sum(sel)
    coord0 = coord1[:, : nloc * 3]
    diff = (
        tnp.reshape(coord1, [batch_size, -1, 3])[:, None, :, :]
        - tnp.reshape(coord0, [batch_size, -1, 3])[:, :, None, :]
    )
    rr = tf.linalg.norm(diff, axis=-1)
    # if central atom has two zero distances, sorting sometimes can not exclude itself
    rr -= tf.eye(nloc, nall, dtype=diff.dtype)[tnp.newaxis, :, :]
    nlist = tnp.argsort(rr, axis=-1)
    rr = tnp.sort(rr, axis=-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]
    nnei = tf.shape(rr)[2]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = tnp.concatenate(
            [rr, tnp.ones([batch_size, nloc, nsel - nnei], dtype=rr.dtype) + rcut],
            axis=-1,
        )
        nlist = tnp.concatenate(
            [nlist, tnp.ones([batch_size, nloc, nsel - nnei], dtype=nlist.dtype)],
            axis=-1,
        )
    nlist = tnp.where(
        tnp.logical_or((rr > rcut), is_vir[:, :nloc, None]),
        tnp.full_like(nlist, -1),
        nlist,
    )

    if distinguish_types:
        return nlist_distinguish_types(nlist, atype, sel)
    else:
        return nlist


def nlist_distinguish_types(
    nlist: tnp.ndarray,
    atype: tnp.ndarray,
    sel: list[int],
):
    """Given a nlist that does not distinguish atom types, return a nlist that
    distinguish atom types.

    """
    nloc = tf.shape(nlist)[1]
    ret_nlist = []
    tmp_atype = tnp.tile(atype[:, None, :], (1, nloc, 1))
    mask = nlist == -1
    tnlist_0 = tnp.where(mask, tnp.zeros_like(nlist), nlist)
    tnlist = tnp.take_along_axis(tmp_atype, tnlist_0, axis=2)
    tnlist = tnp.where(mask, tnp.full_like(tnlist, -1), tnlist)
    for ii, ss in enumerate(sel):
        pick_mask = tf.cast(tnlist == ii, tnp.int32)
        sorted_indices = tnp.argsort(-pick_mask, kind="stable", axis=-1)
        pick_mask_sorted = -tnp.sort(-pick_mask, axis=-1)
        inlist = tnp.take_along_axis(nlist, sorted_indices, axis=2)
        inlist = tnp.where(
            ~tf.cast(pick_mask_sorted, tf.bool), tnp.full_like(inlist, -1), inlist
        )
        ret_nlist.append(inlist[..., :ss])
    ret = tf.concat(ret_nlist, axis=-1)
    return ret


def tf_outer(a, b):
    return tf.einsum("i,j->ij", a, b)


## translated from torch implementation by chatgpt
def extend_coord_with_ghosts(
    coord: tnp.ndarray,
    atype: tnp.ndarray,
    cell: tnp.ndarray,
    rcut: float,
):
    """Extend the coordinates of the atoms by appending peridoc images.
    The number of images is large enough to ensure all the neighbors
    within rcut are appended.

    Parameters
    ----------
    coord : tnp.ndarray
        original coordinates of shape [-1, nloc*3].
    atype : tnp.ndarray
        atom type of shape [-1, nloc].
    cell : tnp.ndarray
        simulation cell tensor of shape [-1, 9].
    rcut : float
        the cutoff radius

    Returns
    -------
    extended_coord: tnp.ndarray
        extended coordinates of shape [-1, nall*3].
    extended_atype: tnp.ndarray
        extended atom type of shape [-1, nall].
    index_mapping: tnp.ndarray
        mapping extended index to the local index

    """
    atype_shape = tf.shape(atype)
    nf, nloc = atype_shape[0], atype_shape[1]
    # int64 for index
    aidx = tf.range(nloc, dtype=tnp.int64)
    aidx = tnp.tile(aidx[tnp.newaxis, :], (nf, 1))
    if tf.shape(cell)[-1] == 0:
        nall = nloc
        extend_coord = coord
        extend_atype = atype
        extend_aidx = aidx
    else:
        coord = tnp.reshape(coord, (nf, nloc, 3))
        cell = tnp.reshape(cell, (nf, 3, 3))
        to_face = to_face_distance(cell)
        nbuff = tf.cast(tnp.ceil(rcut / to_face), tnp.int64)
        nbuff = tnp.max(nbuff, axis=0)
        xi = tf.range(-nbuff[0], nbuff[0] + 1, 1, dtype=tnp.int64)
        yi = tf.range(-nbuff[1], nbuff[1] + 1, 1, dtype=tnp.int64)
        zi = tf.range(-nbuff[2], nbuff[2] + 1, 1, dtype=tnp.int64)
        xyz = tf_outer(xi, tnp.asarray([1, 0, 0]))[:, tnp.newaxis, tnp.newaxis, :]
        xyz = xyz + tf_outer(yi, tnp.asarray([0, 1, 0]))[tnp.newaxis, :, tnp.newaxis, :]
        xyz = xyz + tf_outer(zi, tnp.asarray([0, 0, 1]))[tnp.newaxis, tnp.newaxis, :, :]
        xyz = tnp.reshape(xyz, (-1, 3))
        xyz = tf.cast(xyz, coord.dtype)
        shift_idx = tnp.take(xyz, tnp.argsort(tf.linalg.norm(xyz, axis=1)), axis=0)
        ns = tf.shape(shift_idx)[0]
        nall = ns * nloc
        shift_vec = tnp.einsum("sd,fdk->fsk", shift_idx, cell)
        # shift_vec = tnp.tensordot(shift_idx, cell, axes=([1], [1]))
        # shift_vec = tnp.transpose(shift_vec, (1, 0, 2))
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        extend_atype = tnp.tile(atype[:, :, tnp.newaxis], (1, ns, 1))
        extend_aidx = tnp.tile(aidx[:, :, tnp.newaxis], (1, ns, 1))

    return (
        tnp.reshape(extend_coord, (nf, nall * 3)),
        tnp.reshape(extend_atype, (nf, nall)),
        tnp.reshape(extend_aidx, (nf, nall)),
    )
