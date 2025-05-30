# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Optional,
)

import torch


@torch.jit.script
def aggregate(
    data: torch.Tensor,
    owners: torch.Tensor,
    average: bool = True,
    num_owner: Optional[int] = None,
) -> torch.Tensor:
    """
    Aggregate rows in data by specifying the owners.

    Parameters
    ----------
    data : data tensor to aggregate [n_row, feature_dim]
    owners : specify the owner of each row [n_row, 1]
    average : if True, average the rows, if False, sum the rows.
        Default = True
    num_owner : the number of owners, this is needed if the
        max idx of owner is not presented in owners tensor
        Default = None

    Returns
    -------
    output: [num_owner, feature_dim]
    """
    if num_owner is None or average:
        # requires bincount
        bin_count = torch.bincount(owners)
        bin_count = bin_count.where(bin_count != 0, bin_count.new_ones(1))
        if (num_owner is not None) and (bin_count.shape[0] != num_owner):
            difference = num_owner - bin_count.shape[0]
            bin_count = torch.cat([bin_count, bin_count.new_ones(difference)])
        else:
            num_owner = bin_count.shape[0]
    else:
        bin_count = None

    output = data.new_zeros([num_owner, data.shape[1]])
    output = output.index_add_(0, owners, data)
    if average:
        assert bin_count is not None
        output = (output.T / bin_count).T
    return output


@torch.jit.script
def get_graph_index(
    nlist: torch.Tensor,
    nlist_mask: torch.Tensor,
    a_nlist_mask: torch.Tensor,
    nall: int,
    use_loc_mapping: bool = True,
):
    """
    Get the index mapping for edge graph and angle graph, ready in `aggregate` or `index_select`.

    Parameters
    ----------
    nlist : nf x nloc x nnei
        Neighbor list. (padded neis are set to 0)
    nlist_mask : nf x nloc x nnei
        Masks of the neighbor list. real nei 1 otherwise 0
    a_nlist_mask : nf x nloc x a_nnei
        Masks of the neighbor list for angle. real nei 1 otherwise 0
    nall
        The number of extended atoms.

    Returns
    -------
    edge_index : n_edge x 2
        n2e_index : n_edge
            Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
        n_ext2e_index : n_edge
            Broadcast indices from extended node(j) to edge(ij).
    angle_index : n_angle x 3
        n2a_index : n_angle
            Broadcast indices from extended node(j) to angle(ijk).
        eij2a_index : n_angle
            Broadcast indices from extended edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
        eik2a_index : n_angle
            Broadcast indices from extended edge(ik) to angle(ijk).
    """
    nf, nloc, nnei = nlist.shape
    _, _, a_nnei = a_nlist_mask.shape
    # nf x nloc x nnei x nnei
    # nlist_mask_3d = nlist_mask[:, :, :, None] & nlist_mask[:, :, None, :]
    a_nlist_mask_3d = a_nlist_mask[:, :, :, None] & a_nlist_mask[:, :, None, :]
    n_edge = nlist_mask.sum().item()
    # n_angle = a_nlist_mask_3d.sum().item()

    # following: get n2e_index, n_ext2e_index, n2a_index, eij2a_index, eik2a_index

    # 1. atom graph
    # node(i) to edge(ij) index_select; edge(ij) to node aggregate
    nlist_loc_index = torch.arange(0, nf * nloc, dtype=nlist.dtype, device=nlist.device)
    # nf x nloc x nnei
    n2e_index = nlist_loc_index.reshape(nf, nloc, 1).expand(-1, -1, nnei)
    # n_edge
    n2e_index = n2e_index[nlist_mask]  # graph node index, atom_graph[:, 0]

    # node_ext(j) to edge(ij) index_select
    frame_shift = torch.arange(0, nf, dtype=nlist.dtype, device=nlist.device) * (
        nall if not use_loc_mapping else nloc
    )
    shifted_nlist = nlist + frame_shift[:, None, None]
    # n_edge
    n_ext2e_index = shifted_nlist[nlist_mask]  # graph neighbor index, atom_graph[:, 1]

    # 2. edge graph
    # node(i) to angle(ijk) index_select
    n2a_index = nlist_loc_index.reshape(nf, nloc, 1, 1).expand(-1, -1, a_nnei, a_nnei)
    # n_angle
    n2a_index = n2a_index[a_nlist_mask_3d]

    # edge(ij) to angle(ijk) index_select; angle(ijk) to edge(ij) aggregate
    edge_id = torch.arange(0, n_edge, dtype=nlist.dtype, device=nlist.device)
    # nf x nloc x nnei
    edge_index = torch.zeros([nf, nloc, nnei], dtype=nlist.dtype, device=nlist.device)
    edge_index[nlist_mask] = edge_id
    # only cut a_nnei neighbors, to avoid nnei x nnei
    edge_index = edge_index[:, :, :a_nnei]
    edge_index_ij = edge_index.unsqueeze(-1).expand(-1, -1, -1, a_nnei)
    # n_angle
    eij2a_index = edge_index_ij[a_nlist_mask_3d]

    # edge(ik) to angle(ijk) index_select
    edge_index_ik = edge_index.unsqueeze(-2).expand(-1, -1, a_nnei, -1)
    # n_angle
    eik2a_index = edge_index_ik[a_nlist_mask_3d]

    return torch.cat(
        [n2e_index.unsqueeze(-1), n_ext2e_index.unsqueeze(-1)], dim=-1
    ), torch.cat(
        [n2a_index.unsqueeze(-1), eij2a_index.unsqueeze(-1), eik2a_index.unsqueeze(-1)],
        dim=-1,
    )
