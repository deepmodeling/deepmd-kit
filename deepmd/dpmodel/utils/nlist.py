# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
    Optional,
    Union,
)

import numpy as np

from .region import (
    to_face_distance,
)


## translated from torch implemantation by chatgpt
def build_neighbor_list(
    coord1: np.ndarray,
    atype: np.ndarray,
    nloc: int,
    rcut: float,
    sel: Union[int, List[int]],
    distinguish_types: bool = True,
) -> np.ndarray:
    """Build neightbor list for a single frame. keeps nsel neighbors.

    Parameters
    ----------
    coord1 : np.ndarray
        exptended coordinates of shape [batch_size, nall x 3]
    atype : np.ndarray
        extended atomic types of shape [batch_size, nall]
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

    """
    batch_size = coord1.shape[0]
    coord1 = coord1.reshape(batch_size, -1)
    nall = coord1.shape[1] // 3
    if isinstance(sel, int):
        sel = [sel]
    nsel = sum(sel)
    coord0 = coord1[:, : nloc * 3]
    diff = (
        coord1.reshape([batch_size, -1, 3])[:, None, :, :]
        - coord0.reshape([batch_size, -1, 3])[:, :, None, :]
    )
    assert list(diff.shape) == [batch_size, nloc, nall, 3]
    rr = np.linalg.norm(diff, axis=-1)
    nlist = np.argsort(rr, axis=-1)
    rr = np.sort(rr, axis=-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]
    nnei = rr.shape[2]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = np.concatenate(
            [rr, np.ones([batch_size, nloc, nsel - nnei]) + rcut], axis=-1
        )
        nlist = np.concatenate(
            [nlist, np.ones([batch_size, nloc, nsel - nnei], dtype=nlist.dtype)],
            axis=-1,
        )
    assert list(nlist.shape) == [batch_size, nloc, nsel]
    nlist = np.where((rr > rcut), -1, nlist)

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
    nf, nloc, _ = nlist.shape
    ret_nlist = []
    tmp_atype = np.tile(atype[:, None], [1, nloc, 1])
    mask = nlist == -1
    tnlist_0 = nlist.copy()
    tnlist_0[mask] = 0
    tnlist = np.take_along_axis(tmp_atype, tnlist_0, axis=2).squeeze()
    tnlist = np.where(mask, -1, tnlist)
    snsel = tnlist.shape[2]
    for ii, ss in enumerate(sel):
        pick_mask = (tnlist == ii).astype(np.int32)
        sorted_indices = np.argsort(-pick_mask, kind="stable", axis=-1)
        pick_mask_sorted = -np.sort(-pick_mask, axis=-1)
        inlist = np.take_along_axis(nlist, sorted_indices, axis=2)
        inlist = np.where(~pick_mask_sorted.astype(bool), -1, inlist)
        ret_nlist.append(np.split(inlist, [ss, snsel - ss], axis=-1)[0])
    ret = np.concatenate(ret_nlist, axis=-1)
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
        tnlist_1[rr[:, :, :ns] > rc] = int(-1)
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
    nf, nloc = atype.shape
    aidx = np.tile(np.arange(nloc)[np.newaxis, :], (nf, 1))
    if cell is None:
        nall = nloc
        extend_coord = coord.copy()
        extend_atype = atype.copy()
        extend_aidx = aidx.copy()
    else:
        coord = coord.reshape((nf, nloc, 3))
        cell = cell.reshape((nf, 3, 3))
        to_face = to_face_distance(cell)
        nbuff = np.ceil(rcut / to_face).astype(int)
        nbuff = np.max(nbuff, axis=0)
        xi = np.arange(-nbuff[0], nbuff[0] + 1, 1)
        yi = np.arange(-nbuff[1], nbuff[1] + 1, 1)
        zi = np.arange(-nbuff[2], nbuff[2] + 1, 1)
        xyz = np.outer(xi, np.array([1, 0, 0]))[:, np.newaxis, np.newaxis, :]
        xyz = xyz + np.outer(yi, np.array([0, 1, 0]))[np.newaxis, :, np.newaxis, :]
        xyz = xyz + np.outer(zi, np.array([0, 0, 1]))[np.newaxis, np.newaxis, :, :]
        xyz = xyz.reshape(-1, 3)
        shift_idx = xyz[np.argsort(np.linalg.norm(xyz, axis=1))]
        ns, _ = shift_idx.shape
        nall = ns * nloc
        shift_vec = np.einsum("sd,fdk->fsk", shift_idx, cell)
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        extend_atype = np.tile(atype[:, :, np.newaxis], (1, ns, 1))
        extend_aidx = np.tile(aidx[:, :, np.newaxis], (1, ns, 1))

    return (
        extend_coord.reshape((nf, nall * 3)),
        extend_atype.reshape((nf, nall)),
        extend_aidx.reshape((nf, nall)),
    )
