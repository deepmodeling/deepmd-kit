# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
    Union,
)

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.region import (
    normalize_coord,
    to_face_distance,
)


def extend_input_and_build_neighbor_list(
    coord,
    atype,
    rcut: float,
    sel: list[int],
    mixed_types: bool = False,
    box: Optional[torch.Tensor] = None,
):
    nframes, nloc = atype.shape[:2]
    if box is not None:
        box_gpu = box.to(coord.device, non_blocking=True)
        coord_normalized = normalize_coord(
            coord.view(nframes, nloc, 3),
            box_gpu.reshape(nframes, 3, 3),
        )
    else:
        box_gpu = None
        coord_normalized = coord.clone()
    extended_coord, extended_atype, mapping = extend_coord_with_ghosts(
        coord_normalized, atype, box_gpu, rcut, box
    )
    nlist = build_neighbor_list(
        extended_coord,
        extended_atype,
        nloc,
        rcut,
        sel,
        distinguish_types=(not mixed_types),
    )
    extended_coord = extended_coord.view(nframes, -1, 3)
    return extended_coord, extended_atype, mapping, nlist


def build_neighbor_list(
    coord: torch.Tensor,
    atype: torch.Tensor,
    nloc: int,
    rcut: float,
    sel: Union[int, list[int]],
    distinguish_types: bool = True,
) -> torch.Tensor:
    """Build neighbor list for a single frame. keeps nsel neighbors.

    Parameters
    ----------
    coord : torch.Tensor
        exptended coordinates of shape [batch_size, nall x 3]
    atype : torch.Tensor
        extended atomic types of shape [batch_size, nall]
        if type < 0 the atom is treat as virtual atoms.
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
    neighbor_list : torch.Tensor
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
    batch_size = coord.shape[0]
    nall = coord.shape[1] // 3
    # fill virtual atoms with large coords so they are not neighbors of any
    # real atom.
    if coord.numel() > 0:
        xmax = torch.max(coord) + 2.0 * rcut
    else:
        xmax = torch.zeros(1, dtype=coord.dtype, device=coord.device) + 2.0 * rcut

    coord_xyz = coord.view(batch_size, nall, 3)
    # nf x nall
    is_vir = atype < 0
    # batch_size x nall x 3
    vcoord_xyz = torch.where(is_vir[:, :, None], xmax, coord_xyz)
    if isinstance(sel, int):
        sel = [sel]

    # Get the coordinates for the local atoms (first nloc atoms)
    # batch_size x nloc x 3
    vcoord_local_xyz = vcoord_xyz[:, :nloc, :]

    # Calculate displacement vectors.
    diff = vcoord_xyz.unsqueeze(1) - vcoord_local_xyz.unsqueeze(2)
    assert diff.shape == (batch_size, nloc, nall, 3)
    # nloc x nall
    rr = torch.linalg.norm(diff, dim=-1)
    # if central atom has two zero distances, sorting sometimes can not exclude itself
    # The following operation makes rr[b, i, i] = -1.0 (assuming original self-distance is 0)
    # so that self-atom is sorted first.
    diag_len = min(nloc, nall)
    idx = torch.arange(diag_len, device=rr.device, dtype=torch.int)
    rr[:, idx, idx] -= 1.0

    nsel = sum(sel)
    nnei = rr.shape[-1]
    top_k = min(nsel + 1, nnei)
    rr, nlist = torch.topk(rr, top_k, largest=False)

    # nloc x (nall-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]

    return _trim_mask_distinguish_nlist(
        is_vir, atype, rr, nlist, rcut, sel, distinguish_types
    )


def _trim_mask_distinguish_nlist(
    is_vir_cntl: torch.Tensor,
    atype_neig: torch.Tensor,
    rr: torch.Tensor,
    nlist: torch.Tensor,
    rcut: float,
    sel: list[int],
    distinguish_types: bool,
) -> torch.Tensor:
    """Trim the size of nlist, mask if any central atom is virtual, distinguish types if necessary."""
    nsel = sum(sel)
    # nloc x nsel
    batch_size, nloc, nnei = rr.shape
    assert batch_size == is_vir_cntl.shape[0]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = torch.cat(
            [
                rr,
                torch.ones(
                    [batch_size, nloc, nsel - nnei], device=rr.device, dtype=rr.dtype
                )
                + rcut,
            ],
            dim=-1,
        )
        nlist = torch.cat(
            [
                nlist,
                torch.ones(
                    [batch_size, nloc, nsel - nnei], dtype=nlist.dtype, device=rr.device
                ),
            ],
            dim=-1,
        )
    assert list(nlist.shape) == [batch_size, nloc, nsel]
    nlist = torch.where(
        torch.logical_or((rr > rcut), is_vir_cntl[:, :nloc, None]), -1, nlist
    )
    if distinguish_types:
        return nlist_distinguish_types(nlist, atype_neig, sel)
    else:
        return nlist


def build_directional_neighbor_list(
    coord_cntl: torch.Tensor,
    atype_cntl: torch.Tensor,
    coord_neig: torch.Tensor,
    atype_neig: torch.Tensor,
    rcut: float,
    sel: Union[int, list[int]],
    distinguish_types: bool = True,
) -> torch.Tensor:
    """Build directional neighbor list.

    With each central atom, all the neighbor atoms in the cut-off radius will
    be recorded in the neighbor list. The maximum neighbors is nsel. If the real
    number of neighbors is larger than nsel, the neighbors will be sorted with the
    distance and the first nsel neighbors are kept.

    Important: the central and neighboring atoms are assume to be different atoms.

    Parameters
    ----------
    coord_central : torch.Tensor
        coordinates of central atoms. assumed to be local atoms.
        shape [batch_size, nloc_central x 3]
    atype_central : torch.Tensor
        atomic types of central atoms. shape [batch_size, nloc_central]
        if type < 0 the atom is treated as virtual atoms.
    coord_neighbor : torch.Tensor
        extended coordinates of neighbors atoms. shape [batch_size, nall_neighbor x 3]
    atype_central : torch.Tensor
        extended atomic types of neighbors atoms. shape [batch_size, nall_neighbor]
        if type < 0 the atom is treated as virtual atoms.
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
    neighbor_list : torch.Tensor
        Neighbor list of shape [batch_size, nloc_central, nsel], the neighbors
        are stored in an ascending order. If the number of neighbors is less than nsel,
        the positions are masked with -1. The neighbor list of an atom looks like
        |------ nsel ------|
        xx xx xx xx -1 -1 -1
        if distinguish_types==True and we have two types
        |---- nsel[0] -----| |---- nsel[1] -----|
        xx xx xx xx -1 -1 -1 xx xx xx -1 -1 -1 -1
        For virtual atoms all neighboring positions are filled with -1.
    """
    batch_size = coord_cntl.shape[0]
    coord_cntl = coord_cntl.view(batch_size, -1)
    nloc_cntl = coord_cntl.shape[1] // 3
    coord_neig = coord_neig.view(batch_size, -1)
    nall_neig = coord_neig.shape[1] // 3
    # fill virtual atoms with large coords so they are not neighbors of any
    # real atom.
    if coord_neig.numel() > 0:
        xmax = torch.max(coord_cntl) + 2.0 * rcut
    else:
        xmax = (
            torch.zeros(1, dtype=coord_neig.dtype, device=coord_neig.device)
            + 2.0 * rcut
        )
    # nf x nloc
    is_vir_cntl = atype_cntl < 0
    # nf x nall
    is_vir_neig = atype_neig < 0
    # nf x nloc x 3
    coord_cntl = coord_cntl.view(batch_size, nloc_cntl, 3)
    # nf x nall x 3
    coord_neig = torch.where(
        is_vir_neig[:, :, None], xmax, coord_neig.view(batch_size, nall_neig, 3)
    ).view(batch_size, nall_neig, 3)
    # nsel
    if isinstance(sel, int):
        sel = [sel]
    # nloc x nall x 3
    diff = coord_neig[:, None, :, :] - coord_cntl[:, :, None, :]
    assert list(diff.shape) == [batch_size, nloc_cntl, nall_neig, 3]
    # nloc x nall
    rr = torch.linalg.norm(diff, dim=-1)
    rr, nlist = torch.sort(rr, dim=-1)

    # We assume that the central and neighbor atoms are different,
    # thus we do not need to exclude self-neighbors.
    # # if central atom has two zero distances, sorting sometimes can not exclude itself
    # rr -= torch.eye(nloc_cntl, nall_neig, dtype=rr.dtype, device=rr.device).unsqueeze(0)
    # rr, nlist = torch.sort(rr, dim=-1)
    # # nloc x (nall-1)
    # rr = rr[:, :, 1:]
    # nlist = nlist[:, :, 1:]

    return _trim_mask_distinguish_nlist(
        is_vir_cntl, atype_neig, rr, nlist, rcut, sel, distinguish_types
    )


def nlist_distinguish_types(
    nlist: torch.Tensor,
    atype: torch.Tensor,
    sel: list[int],
):
    """Given a nlist that does not distinguish atom types, return a nlist that
    distinguish atom types.

    """
    nf, nloc, nnei = nlist.shape
    ret_nlist = []
    # nloc x nall
    tmp_atype = torch.tile(atype.unsqueeze(1), [1, nloc, 1])
    mask = nlist == -1
    # nloc x s(nsel)
    tnlist = torch.gather(
        tmp_atype,
        2,
        nlist.masked_fill(mask, 0),
    )
    tnlist = tnlist.masked_fill(mask, -1)
    snsel = tnlist.shape[2]
    for ii, ss in enumerate(sel):
        # nloc x s(nsel)
        # to int because bool cannot be sort on GPU
        pick_mask = (tnlist == ii).to(torch.int32)
        # nloc x s(nsel), stable sort, nearer neighbors first
        pick_mask, imap = torch.sort(pick_mask, dim=-1, descending=True, stable=True)
        # nloc x s(nsel)
        inlist = torch.gather(nlist, 2, imap)
        inlist = inlist.masked_fill(~(pick_mask.to(torch.bool)), -1)
        # nloc x nsel[ii]
        ret_nlist.append(inlist[..., :ss])
    return torch.concat(ret_nlist, dim=-1)


# build_neighbor_list = torch.vmap(
#   build_neighbor_list_lower,
#   in_dims=(0,0,None,None,None),
#   out_dims=(0),
# )


def get_multiple_nlist_key(
    rcut: float,
    nsel: int,
) -> str:
    return str(rcut) + "_" + str(nsel)


def build_multiple_neighbor_list(
    coord: torch.Tensor,
    nlist: torch.Tensor,
    rcuts: list[float],
    nsels: list[int],
) -> dict[str, torch.Tensor]:
    """Input one neighbor list, and produce multiple neighbor lists with
    different cutoff radius and numbers of selection out of it.  The
    required rcuts and nsels should be smaller or equal to the input nlist.

    Parameters
    ----------
    coord : torch.Tensor
        exptended coordinates of shape [batch_size, nall x 3]
    nlist : torch.Tensor
        Neighbor list of shape [batch_size, nloc, nsel], the neighbors
        should be stored in an ascending order.
    rcuts : list[float]
        list of cut-off radius in ascending order.
    nsels : list[int]
        maximal number of neighbors in ascending order.

    Returns
    -------
    nlist_dict : dict[str, torch.Tensor]
        A dict of nlists, key given by get_multiple_nlist_key(rc, nsel)
        value being the corresponding nlist.

    """
    assert len(rcuts) == len(nsels)
    if len(rcuts) == 0:
        return {}
    nb, nloc, nsel = nlist.shape
    if nsel < nsels[-1]:
        pad = -1 * torch.ones(
            [nb, nloc, nsels[-1] - nsel],
            dtype=nlist.dtype,
            device=nlist.device,
        )
        # nb x nloc x nsel
        nlist = torch.cat([nlist, pad], dim=-1)
        nsel = nsels[-1]
    # nb x nall x 3
    coord1 = coord.view(nb, -1, 3)
    nall = coord1.shape[1]
    # nb x nloc x 3
    coord0 = coord1[:, :nloc, :]
    nlist_mask = nlist == -1
    # nb x (nloc x nsel) x 3
    index = (
        nlist.masked_fill(nlist_mask, 0)
        .view(nb, nloc * nsel)
        .unsqueeze(-1)
        .expand(-1, -1, 3)
    )
    # nb x nloc x nsel x 3
    coord2 = torch.gather(coord1, dim=1, index=index).view(nb, nloc, nsel, 3)
    # nb x nloc x nsel x 3
    diff = coord2 - coord0[:, :, None, :]
    # nb x nloc x nsel
    rr = torch.linalg.norm(diff, dim=-1)
    rr.masked_fill(nlist_mask, float("inf"))
    nlist0 = nlist
    ret = {}
    for rc, ns in zip(rcuts[::-1], nsels[::-1]):
        nlist0 = nlist0[:, :, :ns].masked_fill(rr[:, :, :ns] > rc, -1)
        ret[get_multiple_nlist_key(rc, ns)] = nlist0
    return ret


def extend_coord_with_ghosts(
    coord: torch.Tensor,
    atype: torch.Tensor,
    cell: Optional[torch.Tensor],
    rcut: float,
    cell_cpu: Optional[torch.Tensor] = None,
):
    """Extend the coordinates of the atoms by appending peridoc images.
    The number of images is large enough to ensure all the neighbors
    within rcut are appended.

    Parameters
    ----------
    coord : torch.Tensor
        original coordinates of shape [-1, nloc*3].
    atype : torch.Tensor
        atom type of shape [-1, nloc].
    cell : torch.Tensor
        simulation cell tensor of shape [-1, 9].
    rcut : float
        the cutoff radius
    cell_cpu : torch.Tensor
        cell on cpu for performance

    Returns
    -------
    extended_coord: torch.Tensor
        extended coordinates of shape [-1, nall*3].
    extended_atype: torch.Tensor
        extended atom type of shape [-1, nall].
    index_mapping: torch.Tensor
        mapping extended index to the local index

    """
    device = coord.device
    nf, nloc = atype.shape
    # int64 for index
    aidx = torch.tile(
        torch.arange(nloc, device=device, dtype=torch.int64).unsqueeze(0), [nf, 1]
    )
    if cell is None:
        nall = nloc
        extend_coord = coord.clone()
        extend_atype = atype.clone()
        extend_aidx = aidx.clone()
    else:
        coord = coord.view([nf, nloc, 3])
        cell = cell.view([nf, 3, 3])
        cell_cpu = cell_cpu.view([nf, 3, 3]) if cell_cpu is not None else cell
        # nf x 3
        to_face = to_face_distance(cell_cpu)
        # nf x 3
        # *2: ghost copies on + and - directions
        # +1: central cell
        nbuff = torch.ceil(rcut / to_face).to(torch.int64)
        # 3
        nbuff = torch.amax(nbuff, dim=0)  # faster than torch.max
        nbuff_cpu = nbuff.cpu()
        xi = torch.arange(
            -nbuff_cpu[0], nbuff_cpu[0] + 1, 1, device="cpu", dtype=torch.int64
        )
        yi = torch.arange(
            -nbuff_cpu[1], nbuff_cpu[1] + 1, 1, device="cpu", dtype=torch.int64
        )
        zi = torch.arange(
            -nbuff_cpu[2], nbuff_cpu[2] + 1, 1, device="cpu", dtype=torch.int64
        )
        eye_3 = torch.eye(3, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device="cpu")
        xyz = xi.view(-1, 1, 1, 1) * eye_3[0]
        xyz = xyz + yi.view(1, -1, 1, 1) * eye_3[1]
        xyz = xyz + zi.view(1, 1, -1, 1) * eye_3[2]
        xyz = xyz.view(-1, 3)
        xyz = xyz.to(device=device, non_blocking=True)
        # ns x 3
        shift_idx = xyz[torch.argsort(torch.linalg.norm(xyz, dim=-1))]
        ns, _ = shift_idx.shape
        nall = ns * nloc
        # nf x ns x 3
        shift_vec = torch.einsum("sd,fdk->fsk", shift_idx, cell)
        # nf x ns x nloc x 3
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        # nf x ns x nloc
        extend_atype = torch.tile(atype.unsqueeze(-2), [1, ns, 1])
        # nf x ns x nloc
        extend_aidx = torch.tile(aidx.unsqueeze(-2), [1, ns, 1])
    return (
        extend_coord.reshape([nf, nall * 3]).to(device),
        extend_atype.view([nf, nall]).to(device),
        extend_aidx.view([nf, nall]).to(device),
    )
