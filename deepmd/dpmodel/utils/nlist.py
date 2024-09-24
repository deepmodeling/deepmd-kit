# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import array_api_compat
import numpy as np

from .region import (
    normalize_coord,
    to_face_distance,
)


def extend_input_and_build_neighbor_list(
    coord,
    atype,
    rcut: float,
    sel: List[int],
    mixed_types: bool = False,
    box: Optional[np.ndarray] = None,
):
    nframes, nloc = atype.shape[:2]
    if box is not None:
        coord_normalized = normalize_coord(
            coord.reshape(nframes, nloc, 3),
            box.reshape(nframes, 3, 3),
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
    extended_coord = extended_coord.reshape(nframes, -1, 3)
    return extended_coord, extended_atype, mapping, nlist


## translated from torch implemantation by chatgpt
def build_neighbor_list(
    coord: np.ndarray,
    atype: np.ndarray,
    nloc: int,
    rcut: float,
    sel: Union[int, List[int]],
    distinguish_types: bool = True,
) -> np.ndarray:
    """Build neightbor list for a single frame. keeps nsel neighbors.

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
    sel : int or List[int]
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
            [rr, xp.ones([batch_size, nloc, nsel - nnei]) + rcut],  # pylint: disable=no-explicit-dtype
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
    sel: List[int],
):
    """Given a nlist that does not distinguish atom types, return a nlist that
    distinguish atom types.

    """
    xp = array_api_compat.array_namespace(nlist, atype)
    nf, nloc, _ = nlist.shape
    ret_nlist = []
    tmp_atype = xp.tile(atype[:, None], [1, nloc, 1])
    mask = nlist == -1
    tnlist_0 = nlist.copy()
    tnlist_0[mask] = 0
    tnlist = xp.take_along_axis(tmp_atype, tnlist_0, axis=2).squeeze()
    tnlist = xp.where(mask, -1, tnlist)
    snsel = tnlist.shape[2]
    for ii, ss in enumerate(sel):
        pick_mask = (tnlist == ii).astype(xp.int32)
        sorted_indices = xp.argsort(-pick_mask, kind="stable", axis=-1)
        pick_mask_sorted = -xp.sort(-pick_mask, axis=-1)
        inlist = xp.take_along_axis(nlist, sorted_indices, axis=2)
        inlist = xp.where(~pick_mask_sorted.astype(bool), -1, inlist)
        ret_nlist.append(xp.split(inlist, [ss, snsel - ss], axis=-1)[0])
    ret = xp.concatenate(ret_nlist, axis=-1)
    return ret


def get_multiple_nlist_key(rcut: float, nsel: int) -> str:
    return str(rcut) + "_" + str(nsel)


## translated from torch implemantation by chatgpt
def build_multiple_neighbor_list(
    coord: np.ndarray,
    nlist: np.ndarray,
    rcuts: List[float],
    nsels: List[int],
) -> Dict[str, np.ndarray]:
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
    rcuts : List[float]
        list of cut-off radius in ascending order.
    nsels : List[int]
        maximal number of neighbors in ascending order.

    Returns
    -------
    nlist_dict : Dict[str, np.ndarray]
        A dict of nlists, key given by get_multiple_nlist_key(rc, nsel)
        value being the corresponding nlist.

    """
    assert len(rcuts) == len(nsels)
    if len(rcuts) == 0:
        return {}
    nb, nloc, nsel = nlist.shape
    if nsel < nsels[-1]:
        pad = -1 * np.ones((nb, nloc, nsels[-1] - nsel), dtype=nlist.dtype)
        nlist = np.concatenate([nlist, pad], axis=-1)
        nsel = nsels[-1]
    coord1 = coord.reshape(nb, -1, 3)
    nall = coord1.shape[1]
    coord0 = coord1[:, :nloc, :]
    nlist_mask = nlist == -1
    tnlist_0 = nlist.copy()
    tnlist_0[nlist_mask] = 0
    index = np.tile(tnlist_0.reshape(nb, nloc * nsel, 1), [1, 1, 3])
    coord2 = np.take_along_axis(coord1, index, axis=1).reshape(nb, nloc, nsel, 3)
    diff = coord2 - coord0[:, :, None, :]
    rr = np.linalg.norm(diff, axis=-1)
    rr = np.where(nlist_mask, float("inf"), rr)
    nlist0 = nlist
    ret = {}
    for rc, ns in zip(rcuts[::-1], nsels[::-1]):
        tnlist_1 = np.copy(nlist0[:, :, :ns])
        tnlist_1[rr[:, :, :ns] > rc] = -1
        ret[get_multiple_nlist_key(rc, ns)] = tnlist_1
    return ret


## translated from torch implemantation by chatgpt
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
        maping extended index to the local index

    """
    xp = array_api_compat.array_namespace(coord, atype)
    nf, nloc = atype.shape
    aidx = xp.tile(xp.arange(nloc)[xp.newaxis, :], (nf, 1))  # pylint: disable=no-explicit-dtype
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
        xi = xp.arange(-int(nbuff[0]), int(nbuff[0]) + 1, 1)  # pylint: disable=no-explicit-dtype
        yi = xp.arange(-int(nbuff[1]), int(nbuff[1]) + 1, 1)  # pylint: disable=no-explicit-dtype
        zi = xp.arange(-int(nbuff[2]), int(nbuff[2]) + 1, 1)  # pylint: disable=no-explicit-dtype
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
