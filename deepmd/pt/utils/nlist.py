# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Dict,
    List,
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
    sel: List[int],
    mixed_types: bool = False,
    box: Optional[torch.Tensor] = None,
):
    nframes, nloc = atype.shape[:2]
    if box is not None:
        coord_normalized = normalize_coord(
            coord.view(nframes, nloc, 3),
            box.reshape(nframes, 3, 3),
        )
    else:
        coord_normalized = coord.clone()
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
    extended_coord = extended_coord.view(nframes, -1, 3)
    return extended_coord, extended_atype, mapping, nlist


def build_neighbor_list(
    coord1: torch.Tensor,
    atype: torch.Tensor,
    nloc: int,
    rcut: float,
    sel: Union[int, List[int]],
    distinguish_types: bool = True,
) -> torch.Tensor:
    """Build neightbor list for a single frame. keeps nsel neighbors.

    Parameters
    ----------
    coord1 : torch.Tensor
        exptended coordinates of shape [batch_size, nall x 3]
    atype : torch.Tensor
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

    """
    batch_size = coord1.shape[0]
    coord1 = coord1.view(batch_size, -1)
    nall = coord1.shape[1] // 3
    if isinstance(sel, int):
        sel = [sel]
    nsel = sum(sel)
    # nloc x 3
    coord0 = coord1[:, : nloc * 3]
    # nloc x nall x 3
    diff = coord1.view([batch_size, -1, 3]).unsqueeze(1) - coord0.view(
        [batch_size, -1, 3]
    ).unsqueeze(2)
    assert list(diff.shape) == [batch_size, nloc, nall, 3]
    # nloc x nall
    rr = torch.linalg.norm(diff, dim=-1)
    rr, nlist = torch.sort(rr, dim=-1)
    # nloc x (nall-1)
    rr = rr[:, :, 1:]
    nlist = nlist[:, :, 1:]
    # nloc x nsel
    nnei = rr.shape[2]
    if nsel <= nnei:
        rr = rr[:, :, :nsel]
        nlist = nlist[:, :, :nsel]
    else:
        rr = torch.cat(
            [rr, torch.ones([batch_size, nloc, nsel - nnei], device=rr.device) + rcut],
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
    nlist = nlist.masked_fill((rr > rcut), -1)

    if distinguish_types:
        return nlist_distinguish_types(nlist, atype, sel)
    else:
        return nlist


def nlist_distinguish_types(
    nlist: torch.Tensor,
    atype: torch.Tensor,
    sel: List[int],
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
        ret_nlist.append(torch.split(inlist, [ss, snsel - ss], dim=-1)[0])
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
    rcuts: List[float],
    nsels: List[int],
) -> Dict[str, torch.Tensor]:
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
    rcuts : List[float]
        list of cut-off radius in ascending order.
    nsels : List[int]
        maximal number of neighbors in ascending order.

    Returns
    -------
    nlist_dict : Dict[str, torch.Tensor]
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
        nlist0 = nlist0[:, :, :ns].masked_fill(rr[:, :, :ns] > rc, int(-1))
        ret[get_multiple_nlist_key(rc, ns)] = nlist0
    return ret


def extend_coord_with_ghosts(
    coord: torch.Tensor,
    atype: torch.Tensor,
    cell: Optional[torch.Tensor],
    rcut: float,
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

    Returns
    -------
    extended_coord: torch.Tensor
        extended coordinates of shape [-1, nall*3].
    extended_atype: torch.Tensor
        extended atom type of shape [-1, nall].
    index_mapping: torch.Tensor
        maping extended index to the local index

    """
    device = coord.device
    nf, nloc = atype.shape
    aidx = torch.tile(torch.arange(nloc, device=device).unsqueeze(0), [nf, 1])
    if cell is None:
        nall = nloc
        extend_coord = coord.clone()
        extend_atype = atype.clone()
        extend_aidx = aidx.clone()
    else:
        coord = coord.view([nf, nloc, 3])
        cell = cell.view([nf, 3, 3])
        # nf x 3
        to_face = to_face_distance(cell)
        # nf x 3
        # *2: ghost copies on + and - directions
        # +1: central cell
        nbuff = torch.ceil(rcut / to_face).to(torch.long)
        # 3
        nbuff = torch.max(nbuff, dim=0, keepdim=False).values
        xi = torch.arange(-nbuff[0], nbuff[0] + 1, 1, device=device)
        yi = torch.arange(-nbuff[1], nbuff[1] + 1, 1, device=device)
        zi = torch.arange(-nbuff[2], nbuff[2] + 1, 1, device=device)
        xyz = xi.view(-1, 1, 1, 1) * torch.tensor(
            [1, 0, 0], dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=device
        )
        xyz = xyz + yi.view(1, -1, 1, 1) * torch.tensor(
            [0, 1, 0], dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=device
        )
        xyz = xyz + zi.view(1, 1, -1, 1) * torch.tensor(
            [0, 0, 1], dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=device
        )
        xyz = xyz.view(-1, 3)
        # ns x 3
        shift_idx = xyz[torch.argsort(torch.norm(xyz, dim=1))]
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
