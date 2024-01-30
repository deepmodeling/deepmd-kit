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
    if nsel != nnei:
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

    if not distinguish_types:
        return nlist
    else:
        ret_nlist = []
        tmp_atype = np.tile(atype[:, None], [1, nloc, 1])
        mask = nlist == -1
        tnlist_0 = nlist
        tnlist_0[mask] = 0
        tnlist = np.take_along_axis(tmp_atype, tnlist_0[:, :, None], axis=2).squeeze()
        tnlist = np.where(mask, -1, tnlist)
        snsel = tnlist.shape[2]
        for ii, ss in enumerate(sel):
            pick_mask = (tnlist == ii).astype(np.int32)
            pick_mask_sorted_indices = np.argsort(-pick_mask, kind="stable", axis=-1)
            inlist = np.take_along_axis(nlist, pick_mask_sorted_indices, axis=2)
            inlist = np.where(pick_mask.astype(bool), inlist, -1)
            ret_nlist.append(np.split(inlist, [ss, snsel - ss], axis=-1)[0])
        return np.concatenate(ret_nlist, axis=-1)


def get_multiple_nlist_key(rcut: float, nsel: int) -> str:
    return str(rcut) + "_" + str(nsel)


def build_multiple_neighbor_list(
    coord: np.ndarray, nlist: np.ndarray, rcuts: List[float], nsels: List[int]
) -> Dict[str, np.ndarray]:
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
    tnlist_0 = nlist
    tnlist_0[nlist_mask] = 0
    index = np.tile(tnlist_0.reshape(nb, nloc * nsel, 1), [1, 1, 3])
    coord2 = np.take_along_axis(coord1, index, axis=1).reshape(nb, nloc, nsel, 3)
    diff = coord2 - coord0[:, :, None, :]
    rr = np.linalg.norm(diff, axis=-1)
    rr = np.where(nlist_mask, float("inf"), rr)
    nlist0 = nlist
    ret = {}
    for rc, ns in zip(rcuts[::-1], nsels[::-1]):
        tnlist_1 = nlist0[:, :, :ns]
        tnlist_1[rr[:, :, :ns] > rc] = int(-1)
        nlist0 = tnlist_1
        ret[get_multiple_nlist_key(rc, ns)] = nlist0
    return ret


def extend_coord_with_ghosts(
    coord: np.ndarray, atype: np.ndarray, cell: Optional[np.ndarray], rcut: float
):
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
