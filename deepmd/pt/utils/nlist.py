# SPDX-License-Identifier: LGPL-3.0-or-later

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.region import (
    normalize_coord,
    to_face_distance,
)


def extend_input_and_build_neighbor_list(
    coord: torch.Tensor,
    atype: torch.Tensor,
    rcut: float,
    sel: list[int],
    mixed_types: bool = False,
    box: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def extend_input_and_build_neighbor_list_flat(
    coord: torch.Tensor,
    atype: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
    rcut: float,
    sel: list[int],
    mixed_types: bool = False,
    box: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extend coordinates and build neighbor list for flat mixed-batch format.

    This function handles mixed batches where different frames have different numbers
    of atoms. Each frame's periodic boundary conditions are applied correctly using
    its own box.

    Parameters
    ----------
    coord : torch.Tensor
        Flattened coordinates with shape [total_atoms, 3].
    atype : torch.Tensor
        Flattened atom types with shape [total_atoms].
    batch : torch.Tensor
        Atom-to-frame assignment with shape [total_atoms].
    ptr : torch.Tensor
        Frame boundaries with shape [nframes + 1].
    rcut : float
        Cut-off radius.
    sel : list[int]
        Maximal number of neighbors (of each type).
    mixed_types : bool
        Whether to use mixed types (no type distinction in neighbor list).
    box : torch.Tensor | None
        Simulation boxes with shape [nframes, 9]. If None, no PBC is applied.

    Returns
    -------
    extended_coord : torch.Tensor
        Extended coordinates with shape [total_extended_atoms, 3].
    extended_atype : torch.Tensor
        Extended atom types with shape [total_extended_atoms].
    extended_batch : torch.Tensor
        Frame assignment for extended atoms with shape [total_extended_atoms].
    mapping : torch.Tensor
        Extended atom -> local flat index mapping with shape [total_extended_atoms].
    nlist : torch.Tensor
        Neighbor list with shape [total_atoms, nnei], using flat indices.

    Notes
    -----
    This function processes each frame independently, then concatenates the results.
    The neighbor list indices are adjusted to reference the flat extended coordinate array.
    """
    device = coord.device
    nframes = ptr.numel() - 1

    # Lists to collect results from each frame
    extended_coords_list = []
    extended_atypes_list = []
    extended_batches_list = []
    mappings_list = []
    nlists_list = []

    # Track cumulative offsets for adjusting indices
    extended_offset = 0

    for i in range(nframes):
        # Extract frame data
        start_idx = int(ptr[i].item())
        end_idx = int(ptr[i + 1].item())
        nloc = end_idx - start_idx

        # Get frame coordinates and types
        frame_coord = coord[start_idx:end_idx].reshape(1, nloc, 3)  # [1, nloc, 3]
        frame_atype = atype[start_idx:end_idx].reshape(1, nloc)      # [1, nloc]

        # Get frame box if available
        frame_box = None
        if box is not None:
            frame_box = box[i:i+1]  # [1, 9]

        # Normalize coordinates if box is provided
        if frame_box is not None:
            box_gpu = frame_box.to(device, non_blocking=True)
            coord_normalized = normalize_coord(
                frame_coord,
                box_gpu.reshape(1, 3, 3),
            )
        else:
            box_gpu = None
            coord_normalized = frame_coord.clone()

        # Extend coordinates with ghosts (periodic images)
        frame_extended_coord, frame_extended_atype, frame_mapping = extend_coord_with_ghosts(
            coord_normalized.reshape(1, -1),
            frame_atype,
            box_gpu,
            rcut,
            frame_box,
        )

        # Build neighbor list for this frame
        frame_nlist = build_neighbor_list(
            frame_extended_coord,
            frame_extended_atype,
            nloc,
            rcut,
            sel,
            distinguish_types=(not mixed_types),
        )

        # Reshape to remove batch dimension
        frame_extended_coord = frame_extended_coord.view(-1, 3)  # [nall_frame, 3]
        frame_extended_atype = frame_extended_atype.view(-1)     # [nall_frame]
        frame_mapping = frame_mapping.view(-1)                   # [nall_frame]
        frame_nlist = frame_nlist.view(nloc, -1)                 # [nloc, nnei]

        nall_frame = frame_extended_coord.shape[0]

        # Adjust mapping to reference flat coordinate indices
        frame_mapping_flat = frame_mapping + start_idx

        # Adjust neighbor list indices to reference flat extended coordinate array
        # frame_nlist currently has indices in [0, nall_frame)
        # We need to shift them by extended_offset
        frame_nlist_adjusted = torch.where(
            frame_nlist >= 0,
            frame_nlist + extended_offset,
            frame_nlist,
        )

        # Create batch assignment for extended atoms
        frame_extended_batch = torch.full(
            (nall_frame,), i, dtype=torch.long, device=device
        )

        # Collect results
        extended_coords_list.append(frame_extended_coord)
        extended_atypes_list.append(frame_extended_atype)
        extended_batches_list.append(frame_extended_batch)
        mappings_list.append(frame_mapping_flat)
        nlists_list.append(frame_nlist_adjusted)

        # Update offset for next frame
        extended_offset += nall_frame

    # Concatenate all frames
    extended_coord = torch.cat(extended_coords_list, dim=0)      # [total_extended_atoms, 3]
    extended_atype = torch.cat(extended_atypes_list, dim=0)      # [total_extended_atoms]
    extended_batch = torch.cat(extended_batches_list, dim=0)     # [total_extended_atoms]
    mapping = torch.cat(mappings_list, dim=0)                    # [total_extended_atoms]
    nlist = torch.cat(nlists_list, dim=0)                        # [total_atoms, nnei]

    return extended_coord, extended_atype, extended_batch, mapping, nlist


def build_precomputed_flat_graph(
    coord: torch.Tensor,
    atype: torch.Tensor,
    batch: torch.Tensor,
    ptr: torch.Tensor,
    rcut: float,
    sel: list[int],
    a_rcut: float,
    a_sel: int,
    mixed_types: bool = False,
    box: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    device = coord.device
    nframes = ptr.numel() - 1
    extended_coords_list = []
    extended_atypes_list = []
    extended_batches_list = []
    extended_images_list = []
    mappings_list = []
    nlists_ext_list = []
    central_indices_list = []
    extended_ptr = torch.zeros(nframes + 1, dtype=torch.long, device=device)
    extended_offset = 0

    for frame_idx in range(nframes):
        start_idx = int(ptr[frame_idx].item())
        end_idx = int(ptr[frame_idx + 1].item())
        nloc = end_idx - start_idx
        frame_coord = coord[start_idx:end_idx].reshape(1, nloc, 3)
        frame_atype = atype[start_idx:end_idx].reshape(1, nloc)
        frame_box = box[frame_idx : frame_idx + 1] if box is not None else None

        if frame_box is not None:
            box_device = frame_box.to(device, non_blocking=True)
            coord_normalized = normalize_coord(
                frame_coord,
                box_device.reshape(1, 3, 3),
            )
        else:
            box_device = None
            coord_normalized = frame_coord.clone()

        (
            frame_extended_coord,
            frame_extended_atype,
            frame_mapping,
            frame_extended_image,
        ) = extend_coord_with_ghosts_with_images(
            coord_normalized.reshape(1, -1),
            frame_atype,
            box_device,
            rcut,
            frame_box,
        )
        frame_nlist_ext = build_neighbor_list(
            frame_extended_coord,
            frame_extended_atype,
            nloc,
            rcut,
            sel,
            distinguish_types=(not mixed_types),
        )

        frame_extended_coord = frame_extended_coord.view(-1, 3)
        frame_extended_atype = frame_extended_atype.view(-1)
        frame_mapping = frame_mapping.view(-1)
        frame_extended_image = frame_extended_image.view(-1, 3)
        frame_nlist_ext = frame_nlist_ext.view(nloc, -1)
        nall_frame = frame_extended_coord.shape[0]

        central_indices_list.append(
            torch.arange(
                extended_offset,
                extended_offset + nloc,
                dtype=torch.long,
                device=device,
            )
        )
        nlists_ext_list.append(
            torch.where(
                frame_nlist_ext >= 0,
                frame_nlist_ext + extended_offset,
                frame_nlist_ext,
            )
        )
        extended_coords_list.append(frame_extended_coord)
        extended_atypes_list.append(frame_extended_atype)
        extended_batches_list.append(
            torch.full((nall_frame,), frame_idx, dtype=torch.long, device=device)
        )
        extended_images_list.append(frame_extended_image)
        mappings_list.append(frame_mapping + start_idx)
        extended_offset += nall_frame
        extended_ptr[frame_idx + 1] = extended_offset

    extended_coord = torch.cat(extended_coords_list, dim=0)
    extended_atype = torch.cat(extended_atypes_list, dim=0)
    extended_batch = torch.cat(extended_batches_list, dim=0)
    extended_image = torch.cat(extended_images_list, dim=0)
    mapping = torch.cat(mappings_list, dim=0)
    central_ext_index = torch.cat(central_indices_list, dim=0)
    nlist_ext = torch.cat(nlists_ext_list, dim=0)
    nlist_mask = nlist_ext >= 0

    nall = extended_coord.shape[0]
    nlist_ext_clamped = torch.clamp(nlist_ext, min=0, max=nall - 1)
    nlist = torch.where(
        nlist_mask,
        mapping[nlist_ext_clamped],
        torch.tensor(-1, dtype=nlist_ext.dtype, device=device),
    )

    coord_central = extended_coord[central_ext_index]
    coord_pad = torch.cat([extended_coord, extended_coord[-1:, :] + rcut], dim=0)
    nlist_safe = torch.where(nlist_mask, nlist_ext, torch.tensor(nall, device=device))
    index = nlist_safe.view(-1).unsqueeze(-1).expand(-1, 3)
    coord_nei = torch.gather(coord_pad, 0, index).view(nlist_ext.shape[0], -1, 3)
    dist = torch.linalg.norm(coord_nei - coord_central[:, None, :], dim=-1)
    a_dist_mask = (dist[:, :a_sel] < a_rcut) & nlist_mask[:, :a_sel]
    a_nlist_ext = torch.where(
        a_dist_mask,
        nlist_ext[:, :a_sel],
        torch.tensor(-1, dtype=nlist_ext.dtype, device=device),
    )
    a_nlist_mask = a_nlist_ext >= 0
    a_nlist_ext_clamped = torch.clamp(a_nlist_ext, min=0, max=nall - 1)
    a_nlist = torch.where(
        a_nlist_mask,
        mapping[a_nlist_ext_clamped],
        torch.tensor(-1, dtype=nlist_ext.dtype, device=device),
    )

    from deepmd.pt.model.network.graph_utils_flat import get_graph_index_flat

    edge_index, angle_index = get_graph_index_flat(
        nlist,
        extended_batch,
        batch,
        ptr,
        a_nlist_mask,
    )
    return {
        "extended_atype": extended_atype,
        "extended_batch": extended_batch,
        "extended_image": extended_image,
        "extended_ptr": extended_ptr,
        "mapping": mapping,
        "central_ext_index": central_ext_index,
        "nlist": nlist,
        "nlist_ext": nlist_ext,
        "a_nlist": a_nlist,
        "a_nlist_ext": a_nlist_ext,
        "nlist_mask": nlist_mask,
        "a_nlist_mask": a_nlist_mask,
        "edge_index": edge_index,
        "angle_index": angle_index,
    }


def rebuild_extended_coord_from_flat_graph(
    coord: torch.Tensor,
    box: torch.Tensor | None,
    mapping: torch.Tensor,
    extended_batch: torch.Tensor,
    extended_image: torch.Tensor,
) -> torch.Tensor:
    if box is None:
        return coord[mapping]
    cell = box.reshape(-1, 3, 3)
    atom_cell = cell[extended_batch]
    rec_cell, _ = torch.linalg.inv_ex(atom_cell)
    coord_inter = torch.einsum("ni,nij->nj", coord[mapping], rec_cell)
    coord_wrapped = torch.einsum(
        "ni,nij->nj",
        torch.remainder(coord_inter, 1.0),
        atom_cell,
    )
    image = extended_image.to(dtype=box.dtype, device=box.device)
    shift_vec = torch.einsum("ni,nij->nj", image, atom_cell)
    return coord_wrapped + shift_vec


def get_central_ext_index(
    extended_batch: torch.Tensor,
    ptr: torch.Tensor,
) -> torch.Tensor:
    nframes = ptr.numel() - 1
    extended_counts = torch.bincount(extended_batch, minlength=nframes)
    extended_ptr = torch.cat(
        [
            torch.zeros(1, dtype=extended_counts.dtype, device=extended_counts.device),
            torch.cumsum(extended_counts, dim=0),
        ]
    )
    extended_index = torch.arange(
        extended_batch.shape[0], dtype=extended_batch.dtype, device=extended_batch.device
    )
    frame_local_index = extended_index - extended_ptr[extended_batch]
    nloc_per_frame = (ptr[1:] - ptr[:-1]).to(extended_batch.device)
    central_mask = frame_local_index < nloc_per_frame[extended_batch]
    return torch.nonzero(central_mask, as_tuple=False).view(-1)


def extend_input_and_build_neighbor_list_with_images(
    coord: torch.Tensor,
    atype: torch.Tensor,
    rcut: float,
    sel: list[int],
    mixed_types: bool = False,
    box: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like ``extend_input_and_build_neighbor_list`` but also returns lattice images.

    This helper is intended for sidecar graph precomputation workflows that need a
    stable, replayable description of how extended atoms are generated without
    changing the existing training path.

    Returns
    -------
    extended_coord
        Extended coordinates with shape ``[nf, nall, 3]``.
    extended_atype
        Extended atom types with shape ``[nf, nall]``.
    mapping
        Extended atom -> local atom index mapping with shape ``[nf, nall]``.
    extended_image
        Integer lattice image for each extended atom with shape ``[nf, nall, 3]``.
    nlist
        Neighbor list with shape ``[nf, nloc, nnei]``.
    """
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
    extended_coord, extended_atype, mapping, extended_image = (
        extend_coord_with_ghosts_with_images(
            coord_normalized,
            atype,
            box_gpu,
            rcut,
            box,
        )
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
    return extended_coord, extended_atype, mapping, extended_image, nlist


def build_neighbor_list(
    coord: torch.Tensor,
    atype: torch.Tensor,
    nloc: int,
    rcut: float,
    sel: int | list[int],
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
    sel: int | list[int],
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
) -> torch.Tensor:
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
    cell: torch.Tensor | None,
    rcut: float,
    cell_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    extend_coord, extend_atype, extend_aidx = _extend_coord_with_ghosts_impl(
        coord,
        atype,
        cell,
        rcut,
        cell_cpu=cell_cpu,
        return_image=False,
    )
    return extend_coord, extend_atype, extend_aidx


def extend_coord_with_ghosts_with_images(
    coord: torch.Tensor,
    atype: torch.Tensor,
    cell: torch.Tensor | None,
    rcut: float,
    cell_cpu: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extend coordinates and additionally return the integer lattice image.

    The returned image tensor records which periodic image each extended atom
    belongs to. This is useful for sidecar graph serialization where extended
    coordinates should be recoverable from the original local coordinates and
    the simulation cell.
    """
    extend_coord, extend_atype, extend_aidx, extend_image = (
        _extend_coord_with_ghosts_impl(
            coord,
            atype,
            cell,
            rcut,
            cell_cpu=cell_cpu,
            return_image=True,
        )
    )
    return extend_coord, extend_atype, extend_aidx, extend_image


def _extend_coord_with_ghosts_impl(
    coord: torch.Tensor,
    atype: torch.Tensor,
    cell: torch.Tensor | None,
    rcut: float,
    cell_cpu: torch.Tensor | None = None,
    return_image: bool = False,
) -> tuple[torch.Tensor, ...]:
    device = coord.device
    nf, nloc = atype.shape
    aidx = torch.tile(
        torch.arange(nloc, device=device, dtype=torch.int64).unsqueeze(0), [nf, 1]
    )
    if cell is None:
        nall = nloc
        extend_coord = coord.clone()
        extend_atype = atype.clone()
        extend_aidx = aidx.clone()
        extend_image = (
            torch.zeros((nf, nloc, 3), device=device, dtype=torch.int64)
            if return_image
            else None
        )
    else:
        coord = coord.view([nf, nloc, 3])
        cell = cell.view([nf, 3, 3])
        cell_cpu = cell_cpu.view([nf, 3, 3]) if cell_cpu is not None else cell
        to_face = to_face_distance(cell_cpu)
        nbuff = torch.ceil(rcut / to_face).to(torch.int64)
        nbuff = torch.amax(nbuff, dim=0)
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
        shift_idx = xyz[torch.argsort(torch.linalg.norm(xyz, dim=-1))]
        # Convert shift_idx to the same dtype as cell to avoid type mismatch
        shift_idx = shift_idx.to(dtype=cell.dtype)
        ns, _ = shift_idx.shape
        nall = ns * nloc
        shift_vec = torch.einsum("sd,fdk->fsk", shift_idx, cell)
        extend_coord = coord[:, None, :, :] + shift_vec[:, :, None, :]
        extend_atype = torch.tile(atype.unsqueeze(-2), [1, ns, 1])
        extend_aidx = torch.tile(aidx.unsqueeze(-2), [1, ns, 1])
        extend_image = (
            torch.tile(shift_idx.view(1, ns, 1, 3), [nf, 1, nloc, 1])
            if return_image
            else None
        )
    result = [
        extend_coord.reshape([nf, nall * 3]).to(device),
        extend_atype.view([nf, nall]).to(device),
        extend_aidx.view([nf, nall]).to(device),
    ]
    if return_image:
        assert extend_image is not None
        result.append(extend_image.view([nf, nall, 3]).to(device))
    return tuple(result)
