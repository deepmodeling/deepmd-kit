# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from deepmd.dpmodel.array_api import (
    xp_take_along_axis,
)

from .region import (
    normalize_coord,
    to_face_distance,
)


def extend_input_and_build_neighbor_list(
    coord,
    atype,
    rcut: float,
    sel: list[int],
    mixed_types: bool = False,
    box: Optional[np.ndarray] = None,
):
    xp = array_api_compat.array_namespace(coord, atype)
    nframes, nloc = atype.shape[:2]
    if box is not None:
        coord_normalized = normalize_coord(
            xp.reshape(coord, (nframes, nloc, 3)),
            xp.reshape(box, (nframes, 3, 3)),
        )
    else:
        coord_normalized = coord
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, box, rcut
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=(not mixed_types),
    )
    extended_coord = xp.reshape(extended_coord, (nframes, -1, 3))
    return extended_coord, extended_atype, mapping, nlist


## translated from torch implementation by chatgpt
def build_neighbor_list(
    coord: np.ndarray,
    atype: np.ndarray,
    nloc: int,
    rcut: float,
    sel: Union[int, list[int]],
    distinguish_types: bool = True,
) -> np.ndarray:
    """Build neighbor list for a single frame. keeps nsel neighbors.

    Parameters
    ----------
    coord : np.ndarray
        exptended coordinates of shape [batch_size, nall x 3]
    atype : np.ndarray
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
    neighbor_list : np.ndarray
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
    xp = array_api_compat.array_namespace(coord, atype)
    batch_size = coord.shape[0]
    coord = xp.reshape(coord, (batch_size, -1))
    nall = coord.shape[1] // 3
    # fill virtual atoms with large coords so they are not neighbors of any
    # real atom.
    if coord.size > 0:
        xmax = xp.max(coord) + 2.0 * rcut
    else:
        xmax = 2.0 * rcut
    # nf x nall
    is_vir = atype < 0
    coord1 = xp.where(
        is_vir[:, :, None], xmax, xp.reshape(coord, (batch_size, nall, 3))
    )
    coord1 = xp.reshape(coord1, (batch_size, nall * 3))
    if isinstance(sel, int):
        sel = [sel]
    nsel = sum(sel)
    coord0 = coord1[:, : nloc * 3]
    diff = (
        xp.reshape(coord1, [batch_size, -1, 3])[:, None, :, :]
        - xp.reshape(coord0, [batch_size, -1, 3])[:, :, None, :]
    )
    assert list(diff.shape) == [batch_size, nloc, nall, 3]
    rr = xp.linalg.vector_norm(diff, axis=-1)
    # if central atom has two zero distances, sorting sometimes can not exclude itself
    rr -= xp.eye(nloc, nall, dtype=diff.dtype)[xp.newaxis, :, :]
    nlist = xp.argsort(rr, axis=-1)
    rr = xp.sort(rr, axis=-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]
    nnei = rr.shape[2]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = xp.concatenate(
            [rr, xp.ones([batch_size, nloc, nsel - nnei], dtype=rr.dtype) + rcut],
            axis=-1,
        )
        nlist = xp.concatenate(
            [nlist, xp.ones([batch_size, nloc, nsel - nnei], dtype=nlist.dtype)],
            axis=-1,
        )
    assert list(nlist.shape) == [batch_size, nloc, nsel]
    nlist = xp.where(
        xp.logical_or((rr > rcut), is_vir[:, :nloc, None]),
        xp.full_like(nlist, -1),
        nlist,
    )

    if distinguish_types:
        return nlist_distinguish_types(nlist, atype, sel)
    else:
        return nlist


def nlist_distinguish_types(
    nlist: np.ndarray,
    atype: np.ndarray,
    sel: list[int],
):
    """Given a nlist that does not distinguish atom types, return a nlist that
    distinguish atom types.

    """
    xp = array_api_compat.array_namespace(nlist, atype)
    nf, nloc, _ = nlist.shape
    ret_nlist = []
    tmp_atype = xp.tile(atype[:, None, :], (1, nloc, 1))
    mask = nlist == -1
    tnlist_0 = xp.where(mask, xp.zeros_like(nlist), nlist)
    tnlist = xp_take_along_axis(tmp_atype, tnlist_0, axis=2)
    tnlist = xp.where(mask, xp.full_like(tnlist, -1), tnlist)
    for ii, ss in enumerate(sel):
        pick_mask = xp.astype(tnlist == ii, xp.int32)
        sorted_indices = xp.argsort(-pick_mask, stable=True, axis=-1)
        pick_mask_sorted = -xp.sort(-pick_mask, axis=-1)
        inlist = xp_take_along_axis(nlist, sorted_indices, axis=2)
        inlist = xp.where(
            ~xp.astype(pick_mask_sorted, xp.bool), xp.full_like(inlist, -1), inlist
        )
        ret_nlist.append(inlist[..., :ss])
    ret = xp.concat(ret_nlist, axis=-1)
    return ret


def get_multiple_nlist_key(rcut: float, nsel: int) -> str:
    return str(rcut) + "_" + str(nsel)


## translated from torch implementation by chatgpt
def build_multiple_neighbor_list(
    coord: np.ndarray,
    nlist: np.ndarray,
    rcuts: list[float],
    nsels: list[int],
) -> dict[str, np.ndarray]:
    """Input one neighbor list, and produce multiple neighbor lists with
    different cutoff radius and numbers of selection out of it.  The
    required rcuts and nsels should be smaller or equal to the input nlist.

    Parameters
    ----------
    coord : np.ndarray
        exptended coordinates of shape [batch_size, nall x 3]
    nlist : np.ndarray
        Neighbor list of shape [batch_size, nloc, nsel], the neighbors
        should be stored in an ascending order.
    rcuts : list[float]
        list of cut-off radius in ascending order.
    nsels : list[int]
        maximal number of neighbors in ascending order.

    Returns
    -------
    nlist_dict : dict[str, np.ndarray]
        A dict of nlists, key given by get_multiple_nlist_key(rc, nsel)
        value being the corresponding nlist.

    """
    xp = array_api_compat.array_namespace(coord, nlist)
    assert len(rcuts) == len(nsels)
    if len(rcuts) == 0:
        return {}
    nb, nloc, nsel = nlist.shape
    if nsel < nsels[-1]:
        pad = -1 * xp.ones((nb, nloc, nsels[-1] - nsel), dtype=nlist.dtype)
        nlist = xp.concat([nlist, pad], axis=-1)
        nsel = nsels[-1]
    coord1 = xp.reshape(coord, (nb, -1, 3))
    nall = coord1.shape[1]
    coord0 = coord1[:, :nloc, :]
    nlist_mask = nlist == -1
    tnlist_0 = xp.where(nlist_mask, xp.zeros_like(nlist), nlist)
    index = xp.tile(xp.reshape(tnlist_0, (nb, nloc * nsel, 1)), (1, 1, 3))
    coord2 = xp_take_along_axis(coord1, index, axis=1)
    coord2 = xp.reshape(coord2, (nb, nloc, nsel, 3))
    diff = coord2 - coord0[:, :, None, :]
    rr = xp.linalg.vector_norm(diff, axis=-1)
    rr = xp.where(nlist_mask, xp.full_like(rr, float("inf")), rr)
    nlist0 = nlist
    ret = {}
    for rc, ns in zip(rcuts[::-1], nsels[::-1]):
        tnlist_1 = nlist0[:, :, :ns]
        tnlist_1 = xp.where(rr[:, :, :ns] > rc, xp.full_like(tnlist_1, -1), tnlist_1)
        ret[get_multiple_nlist_key(rc, ns)] = tnlist_1
    return ret


## translated from torch implementation by chatgpt
def extend_coord_with_ghosts(
    coord: np.ndarray,
    atype: np.ndarray,
    cell: Optional[np.ndarray],
    rcut: float,
):
    """Extend the coordinates of the atoms by appending peridoc images.
    The number of images is large enough to ensure all the neighbors
    within rcut are appended.

    Parameters
    ----------
    coord : np.ndarray
        original coordinates of shape [-1, nloc*3].
    atype : np.ndarray
        atom type of shape [-1, nloc].
    cell : np.ndarray
        simulation cell tensor of shape [-1, 9].
    rcut : float
        the cutoff radius

    Returns
    -------
    extended_coord: np.ndarray
        extended coordinates of shape [-1, nall*3].
    extended_atype: np.ndarray
        extended atom type of shape [-1, nall].
    index_mapping: np.ndarray
        mapping extended index to the local index

    """
    xp = array_api_compat.array_namespace(coord, atype)
    nf, nloc = atype.shape
    # int64 for index
    aidx = xp.tile(xp.arange(nloc, dtype=xp.int64)[xp.newaxis, :], (nf, 1))
    if cell is None:
        nall = nloc
        extend_coord = coord
        extend_atype = atype
        extend_aidx = aidx
    else:
        coord = xp.reshape(coord, (nf, nloc, 3))
        cell = xp.reshape(cell, (nf, 3, 3))
        to_face = to_face_distance(cell)
        nbuff = xp.astype(xp.ceil(rcut / to_face), xp.int64)
        nbuff = xp.max(nbuff, axis=0)
        xi = xp.arange(-int(nbuff[0]), int(nbuff[0]) + 1, 1, dtype=xp.int64)
        yi = xp.arange(-int(nbuff[1]), int(nbuff[1]) + 1, 1, dtype=xp.int64)
        zi = xp.arange(-int(nbuff[2]), int(nbuff[2]) + 1, 1, dtype=xp.int64)
        xyz = xp.linalg.outer(xi, xp.asarray([1, 0, 0]))[:, xp.newaxis, xp.newaxis, :]
        xyz = (
            xyz
            + xp.linalg.outer(yi, xp.asarray([0, 1, 0]))[xp.newaxis, :, xp.newaxis, :]
        )
        xyz = (
            xyz
            + xp.linalg.outer(zi, xp.asarray([0, 0, 1]))[xp.newaxis, xp.newaxis, :, :]
        )
        xyz = xp.reshape(xyz, (-1, 3))
        xyz = xp.astype(xyz, coord.dtype)
        shift_idx = xp.take(xyz, xp.argsort(xp.linalg.vector_norm(xyz, axis=1)), axis=0)
        ns, _ = shift_idx.shape
        nall = ns * nloc
        # shift_vec = xp.einsum("sd,fdk->fsk", shift_idx, cell)
        shift_vec = xp.tensordot(shift_idx, cell, axes=([1], [1]))
        shift_vec = xp.permute_dims(shift_vec, (1, 0, 2))
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        extend_atype = xp.tile(atype[:, :, xp.newaxis], (1, ns, 1))
        extend_aidx = xp.tile(aidx[:, :, xp.newaxis], (1, ns, 1))

    return (
        xp.reshape(extend_coord, (nf, nall * 3)),
        xp.reshape(extend_atype, (nf, nall)),
        xp.reshape(extend_aidx, (nf, nall)),
    )
