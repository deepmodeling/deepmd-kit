# SPDX-License-Identifier: LGPL-3.0-or-later

import torch


def get_graph_index_flat(
    nlist_flat: torch.Tensor,
    extended_batch: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
    a_nlist_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get the index mapping for edge graph and angle graph in flat format.

    Parameters
    ----------
    nlist_flat : torch.Tensor
        Neighbor list in flat format [total_atoms, nnei].
        Indices refer to positions in extended_coord_flat.
        Padded neighbors are marked as -1.
    extended_batch : torch.Tensor
        Frame assignment for extended atoms [total_extended_atoms].
    batch : torch.Tensor
        Frame assignment for local atoms [total_atoms].
    ptr : torch.Tensor
        Frame boundaries [nframes + 1].
    a_sel : int
        Maximum number of angle neighbors (subset of edge neighbors).

    Returns
    -------
    edge_index : torch.Tensor [2, n_edge]
        n2e_index : n_edge
            Broadcast indices from node(i) to edge(ij), or reduction indices from edge(ij) to node(i).
            These are flat indices in range [0, total_atoms).
        n_ext2e_index : n_edge
            Broadcast indices from extended node(j) to edge(ij).
            These are flat indices in range [0, total_extended_atoms).
    angle_index : torch.Tensor [3, n_angle]
        n2a_index : n_angle
            Broadcast indices from node(i) to angle(ijk).
            These are flat indices in range [0, total_atoms).
        eij2a_index : n_angle
            Broadcast indices from edge(ij) to angle(ijk), or reduction indices from angle(ijk) to edge(ij).
            These are edge indices in range [0, n_edge).
        eik2a_index : n_angle
            Broadcast indices from edge(ik) to angle(ijk).
            These are edge indices in range [0, n_edge).
    """
    total_atoms = nlist_flat.shape[0]
    nnei = nlist_flat.shape[1]
    device = nlist_flat.device
    dtype = nlist_flat.dtype
    a_sel = a_nlist_mask.shape[1]

    # Create mask for valid neighbors (not -1)
    nlist_mask = nlist_flat >= 0  # [total_atoms, nnei]

    # Count edges
    n_edge = nlist_mask.sum().item()

    # Angle mask: both neighbors must be valid
    a_nlist_mask_3d = a_nlist_mask[:, :, None] & a_nlist_mask[:, None, :]  # [total_atoms, a_sel, a_sel]

    # 1. Build edge_index
    # n2e_index: for each edge, which local atom does it belong to
    atom_indices = torch.arange(total_atoms, dtype=dtype, device=device)  # [total_atoms]
    n2e_index = atom_indices[:, None].expand(-1, nnei)[nlist_mask]  # [n_edge]

    # n_ext2e_index: for each edge, which extended atom is the neighbor
    n_ext2e_index = nlist_flat[nlist_mask]  # [n_edge]

    edge_index = torch.stack([n2e_index, n_ext2e_index], dim=0)  # [2, n_edge]
    
    # 2. Build angle_index
    # n2a_index: for each angle, which local atom does it belong to
    n2a_index = atom_indices[:, None, None].expand(-1, a_sel, a_sel)[a_nlist_mask_3d]  # [n_angle]

    # Create edge_id mapping: (atom_idx, neighbor_idx) -> edge_id
    edge_id = torch.arange(n_edge, dtype=dtype, device=device)
    edge_lookup = torch.full((total_atoms, nnei), -1, dtype=dtype, device=device)
    edge_lookup[nlist_mask] = edge_id
    # Only consider first a_sel neighbors for angles
    edge_lookup_a = edge_lookup[:, :a_sel]  # [total_atoms, a_sel]

    # eij2a_index: for each angle (i,j,k), the edge id of (i,j)
    edge_lookup_ij = edge_lookup_a[:, :, None].expand(-1, -1, a_sel)  # [total_atoms, a_sel, a_sel]
    eij2a_index = edge_lookup_ij[a_nlist_mask_3d]  # [n_angle]

    # eik2a_index: for each angle (i,j,k), the edge id of (i,k)
    edge_lookup_ik = edge_lookup_a[:, None, :].expand(-1, a_sel, -1)  # [total_atoms, a_sel, a_sel]
    eik2a_index = edge_lookup_ik[a_nlist_mask_3d]  # [n_angle]

    angle_index = torch.stack([n2a_index, eij2a_index, eik2a_index], dim=0)  # [3, n_angle]
    return edge_index, angle_index


def convert_nlist_to_flat_format(
    nlist_batch: torch.Tensor,
    extended_batch: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
) -> torch.Tensor:
    """
    Convert neighbor list from batch format to flat format.

    In batch format, nlist contains indices relative to each frame's extended atoms.
    In flat format, nlist should contain global indices into extended_coord_flat.

    Parameters
    ----------
    nlist_batch : torch.Tensor
        Neighbor list in batch format [nframes, nloc, nnei].
        Indices are relative to each frame (0 to nall-1 per frame).
    extended_batch : torch.Tensor
        Frame assignment for extended atoms [total_extended_atoms].
    batch : torch.Tensor
        Frame assignment for local atoms [total_atoms].
    ptr : torch.Tensor
        Frame boundaries for local atoms [nframes + 1].

    Returns
    -------
    nlist_flat : torch.Tensor
        Neighbor list in flat format [total_atoms, nnei].
        Indices are global (0 to total_extended_atoms-1).
        Padded neighbors remain as -1.
    """
    nframes, nloc_max, nnei = nlist_batch.shape
    device = nlist_batch.device
    dtype = nlist_batch.dtype

    # Compute extended_ptr: frame boundaries for extended atoms
    extended_ptr = torch.zeros(nframes + 1, dtype=torch.long, device=device)
    for i in range(nframes):
        extended_ptr[i + 1] = extended_ptr[i] + (extended_batch == i).sum()

    # Initialize flat nlist
    total_atoms = ptr[-1].item()
    nlist_flat = torch.full((total_atoms, nnei), -1, dtype=dtype, device=device)

    # Convert frame-relative indices to global indices
    for i in range(nframes):
        start_local = ptr[i].item()
        end_local = ptr[i + 1].item()
        nloc_i = end_local - start_local

        start_extended = extended_ptr[i].item()

        # Get this frame's nlist
        frame_nlist = nlist_batch[i, :nloc_i, :]  # [nloc_i, nnei]

        # Convert: add offset for extended atoms, keep -1 as -1
        frame_nlist_global = torch.where(
            frame_nlist >= 0,
            frame_nlist + start_extended,
            torch.tensor(-1, dtype=dtype, device=device)
        )

        nlist_flat[start_local:end_local, :] = frame_nlist_global

    return nlist_flat
