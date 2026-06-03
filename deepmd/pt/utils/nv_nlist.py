# SPDX-License-Identifier: LGPL-3.0-or-later
"""Toolkit-Ops (``nvalchemiops``) neighbor-list strategy.

A :class:`~deepmd.dpmodel.utils.neighbor_list.NeighborList` implementation that
builds the extended representation ``(extended_coord, extended_atype, nlist,
mapping)`` using the device-resident O(N) cell list in ``nvalchemiops``, intended
for large periodic systems.

Toolkit-Ops returns a dense ``[total_atoms, max_neighbors]`` neighbor matrix over
the flattened batch. The matrix is converted to the DeePMD extended-atom contract
by materializing each unique ghost ``(frame, src_local, shift)`` once; the
candidate list is then distance-sorted and truncated to ``sum(sel)`` so the
returned neighbor count is fixed. The search is non-differentiable and runs on
detached coordinates, while ``extended_coord`` is rebuilt from the input
coordinates so force and virial gradients propagate unchanged.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch

from deepmd.dpmodel.utils.neighbor_list import (
    NeighborList,
)
from deepmd.pt.utils.region import (
    normalize_coord,
)

NV_CELL_LIST_THRESHOLD = 1024


def is_nv_available() -> bool:
    """Whether the ``nvalchemiops`` Toolkit-Ops neighbor list is importable."""
    try:
        import nvalchemiops.torch.neighbors  # noqa: F401
    except ImportError:
        return False
    return True


def choose_nv_nlist_method(nloc: int) -> str:
    """Choose the Toolkit-Ops neighbor method for a homogeneous batch.

    Parameters
    ----------
    nloc
        Number of local atoms per frame.

    Returns
    -------
    str
        Toolkit-Ops method name.
    """
    if nloc >= NV_CELL_LIST_THRESHOLD:
        return "batch_cell_list"
    return "batch_naive"


class NvNeighborList(NeighborList):
    """O(N) neighbor-list strategy using the ``nvalchemiops`` cell list.

    Implements the :class:`~deepmd.dpmodel.utils.neighbor_list.NeighborList`
    interface on torch tensors; the search runs on the device of the input
    coordinates.  A periodic ``box`` is required -- the cell list needs a cell to
    wrap periodic images.
    """

    def build(
        self,
        coord: Any,
        atype: Any,
        box: Any,
        rcut: float,
        sel: list[int],
    ) -> tuple[Any, Any, Any, Any]:
        """Build the extended system and neighbor list.

        See :meth:`deepmd.dpmodel.utils.neighbor_list.NeighborList.build`. The
        returned ``nlist`` is distance-sorted and truncated to ``sum(sel)``. A
        periodic ``box`` is required, as the cell list operates on a periodic cell.
        """
        from nvalchemiops.torch.neighbors import (
            neighbor_list,
        )

        if box is None:
            raise ValueError("NvNeighborList requires a periodic box; got box=None.")

        nf, nloc = atype.shape[:2]
        device = coord.device
        target_neighbors = int(sum(sel))
        search_capacity = target_neighbors
        total_atoms = nf * nloc
        cell = box.reshape(nf, 3, 3).to(device=device, dtype=coord.dtype)
        coord = normalize_coord(coord.reshape(nf, nloc, 3), cell)
        positions_for_nlist = coord.reshape(total_atoms, 3).detach()
        pbc = torch.ones((nf, 3), dtype=torch.bool, device=device)
        batch_idx = torch.arange(
            nf, dtype=torch.int32, device=device
        ).repeat_interleave(nloc)
        batch_ptr = torch.arange(nf + 1, dtype=torch.int32, device=device) * nloc
        method = choose_nv_nlist_method(nloc)

        # Grow the search capacity until all neighbors fit so the distance-sort
        # below selects the true nearest ``sum(sel)``.
        while True:
            neighbor_matrix, num_neighbors, shifts = neighbor_list(
                positions_for_nlist,
                float(rcut),
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                method=method,
                max_neighbors=int(search_capacity),
                return_neighbor_list=False,
                wrap_positions=False,
            )
            max_found = (
                int(num_neighbors.max().item()) if num_neighbors.numel() > 0 else 0
            )
            if max_found <= search_capacity:
                break
            search_capacity = max(max_found, _grow_search_capacity(search_capacity))

        extended_coord, extended_atype, mapping, nlist = _matrix_to_extended_inputs(
            coord=coord,
            atype=atype,
            cell=cell,
            nloc=nloc,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            shifts=shifts,
        )
        nlist = _truncate_to_sel_compiled(
            extended_coord, nlist, target_neighbors, float(rcut)
        )
        return extended_coord, extended_atype, nlist, mapping


def _grow_search_capacity(capacity: int) -> int:
    """Increase Toolkit-Ops capacity by 1.25x, rounded up."""
    return (capacity * 5 + 3) // 4


@torch.no_grad()
def _truncate_to_sel(
    extended_coord: torch.Tensor,
    nlist: torch.Tensor,
    nsel: int,
    rcut: float,
) -> torch.Tensor:
    """Distance-sort the candidate neighbor list and keep the nearest ``nsel``
    within ``rcut``, padding with ``-1`` when fewer neighbors exist.

    The Toolkit-Ops search capacity may exceed ``sum(sel)`` on dense systems; this
    fixes the returned neighbor count at ``nsel``.

    The output is the integer ``nlist``; ``extended_coord`` is only read to rank
    candidates and is returned unchanged by the caller. The routine is therefore
    non-differentiable and runs under ``no_grad`` so it never participates in the
    autograd graph (forward, backward, or the second-order pass used to train
    forces), which also avoids retaining the distance temporaries for backward.
    """
    nf, nloc, nnei = nlist.shape
    if nnei < nsel:
        pad = torch.full(
            (nf, nloc, nsel - nnei), -1, dtype=nlist.dtype, device=nlist.device
        )
        return torch.cat([nlist, pad], dim=-1)
    if nnei == nsel:
        return nlist
    real_neighbor = nlist >= 0
    safe_nlist = torch.where(real_neighbor, nlist, torch.zeros_like(nlist))
    coord0 = extended_coord[:, :nloc, :]
    index = safe_nlist.view(nf, nloc * nnei, 1).expand(-1, -1, 3)
    coord1 = torch.gather(extended_coord, 1, index).view(nf, nloc, nnei, 3)
    rr = torch.linalg.norm(coord1 - coord0[:, :, None, :], dim=-1)
    rr = torch.where(real_neighbor, rr, float("inf"))
    rr, order = torch.sort(rr, dim=-1)
    sorted_nlist = torch.gather(safe_nlist, 2, order)
    sorted_nlist = torch.where(rr > rcut, -1, sorted_nlist)
    # ``.contiguous()`` is required: the bare ``[..., :nsel]`` slice keeps the
    # wider candidate stride, but the compiled lower interface freezes the nlist
    # sel axis and asserts a contiguous layout (``assert_size_stride``).
    return sorted_nlist[..., :nsel].contiguous()


# Lower the gather/distance-sort/mask pipeline of `_truncate_to_sel` into a single
# Inductor graph. ``dynamic=True`` keeps the per-system ``(nf, nloc, nnei)`` shapes
# on one compiled graph instead of recompiling per system size, and fusing the
# pipeline avoids materializing the full ``(nf, nloc, nnei, 3)`` distance
# temporaries, which lowers both this step's peak memory and its latency relative
# to eager. Compilation is lazy: it happens on first call, not at import.
_truncate_to_sel_compiled = torch.compile(_truncate_to_sel, dynamic=True)


def _matrix_to_extended_inputs(
    *,
    coord: torch.Tensor,
    atype: torch.Tensor,
    cell: torch.Tensor,
    nloc: int,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Toolkit-Ops matrix output to compact extended inputs.

    Toolkit-Ops returns neighbors as a dense matrix over flattened atoms:
    ``neighbor_matrix[dst_global, slot] = src_global`` and
    ``shifts[dst_global, slot] = (sx, sy, sz)``. Here ``dst_global`` and
    ``src_global`` are indices in the concatenated ``nf * nloc`` input.

    DeePMD lower paths use a different contract: ``nlist`` stores indices into
    ``extended_coord``. Local atoms occupy ``[0, nloc)`` in each frame, while
    shifted PBC images must be appended as ghost atoms. This conversion builds
    the minimal ghost set by materializing each unique
    ``(frame, src_local, shift)`` once, then redirects all shifted nlist entries
    to the corresponding compact ghost index.
    """
    nf = coord.shape[0]
    total_atoms, max_neighbors = neighbor_matrix.shape
    device = coord.device
    dtype = coord.dtype
    local_mapping = torch.arange(nloc, dtype=torch.long, device=device)
    local_mapping = local_mapping.unsqueeze(0).expand(nf, -1)
    nlist = torch.full((nf, nloc, max_neighbors), -1, dtype=torch.long, device=device)

    # === Step 1. Flatten valid Toolkit-Ops matrix slots ===
    # `edge_idx` indexes the flattened matrix layout `(total_atoms, max_neighbors)`.
    # This avoids constructing a full repeated destination tensor.
    slot = torch.arange(max_neighbors, dtype=torch.long, device=device).expand(
        total_atoms, max_neighbors
    )
    valid = (slot < num_neighbors.unsqueeze(1)).reshape(-1)
    edge_idx = torch.nonzero(valid, as_tuple=False).flatten()
    if edge_idx.numel() == 0:
        return coord, atype, local_mapping, nlist

    # Decode flattened edge slots:
    #   dst         : flattened center atom, in [0, nf * nloc)
    #   src         : flattened neighbor atom returned by Toolkit-Ops
    #   frame_idx   : batch frame/system containing both dst and src
    #   center_idx  : local center atom index inside the frame
    #   src_local   : local neighbor atom index before applying the PBC shift
    dst = edge_idx // max_neighbors
    src = neighbor_matrix.reshape(-1).index_select(0, edge_idx).to(dtype=torch.long)
    shift = shifts.reshape(-1, 3).index_select(0, edge_idx).to(dtype=torch.long)
    src_local = src % nloc
    frame_idx = dst // nloc
    center_idx = dst % nloc
    slot_idx = edge_idx % max_neighbors
    zero_shift = torch.all(shift == 0, dim=1)

    # === Step 2. Direct neighbors keep their local extended indices ===
    # Zero-shift neighbors already live in the leading local block of
    # `extended_coord`, so their DeePMD nlist value is simply `src_local`.
    direct_edge_idx = torch.nonzero(zero_shift, as_tuple=False).flatten()
    nlist[
        frame_idx.index_select(0, direct_edge_idx),
        center_idx.index_select(0, direct_edge_idx),
        slot_idx.index_select(0, direct_edge_idx),
    ] = src_local.index_select(0, direct_edge_idx)

    shifted_edge_idx = torch.nonzero(~zero_shift, as_tuple=False).flatten()
    if shifted_edge_idx.numel() == 0:
        return coord, atype, local_mapping, nlist

    # === Step 3. Materialize each unique shifted atom once per frame ===
    # A shifted source may appear in many center atoms' neighbor slots.  Dedup by
    # `(frame, src_local, shift)` so all such slots share one compact ghost atom.
    ghost_keys = torch.cat(
        [
            frame_idx.index_select(0, shifted_edge_idx).unsqueeze(1),
            src_local.index_select(0, shifted_edge_idx).unsqueeze(1),
            shift.index_select(0, shifted_edge_idx),
        ],
        dim=1,
    )
    unique_keys, inverse = torch.unique(ghost_keys, dim=0, return_inverse=True)
    ghost_frame = unique_keys[:, 0].to(dtype=torch.long)
    ghost_src = unique_keys[:, 1].to(dtype=torch.long)
    ghost_shift = unique_keys[:, 2:].to(dtype=dtype)

    # Assign per-frame compact ghost indices.  `ghost_rank` is the offset within
    # a frame's ghost block, so the final extended index is `nloc + ghost_rank`.
    # The `.item()` sync is only used to size the padded dense output.
    ghost_count = torch.bincount(ghost_frame, minlength=nf)
    max_extra = int(ghost_count.max().item())
    order = torch.argsort(ghost_frame)
    sorted_frame = ghost_frame.index_select(0, order)
    frame_start = torch.cumsum(ghost_count, dim=0) - ghost_count
    sorted_rank = torch.arange(
        unique_keys.shape[0], dtype=torch.long, device=device
    ) - frame_start.index_select(0, sorted_frame)
    ghost_rank = torch.empty_like(sorted_rank)
    ghost_rank[order] = sorted_rank
    ghost_index = nloc + ghost_rank

    extended_coord = torch.zeros((nf, nloc + max_extra, 3), dtype=dtype, device=device)
    extended_atype = torch.full(
        (nf, nloc + max_extra), -1, dtype=atype.dtype, device=device
    )
    mapping = torch.zeros((nf, nloc + max_extra), dtype=torch.long, device=device)
    extended_coord[:, :nloc] = coord
    extended_atype[:, :nloc] = atype
    mapping[:, :nloc] = local_mapping

    # Convert integer cell shifts to Cartesian ghost coordinates and record the
    # extended-to-local mapping used later to scatter forces/virials back.
    shift_cart = torch.bmm(
        ghost_shift.unsqueeze(1), cell.index_select(0, ghost_frame)
    ).squeeze(1)
    extended_coord[ghost_frame, ghost_index] = (
        coord[ghost_frame, ghost_src] + shift_cart
    )
    extended_atype[ghost_frame, ghost_index] = atype[ghost_frame, ghost_src]
    mapping[ghost_frame, ghost_index] = ghost_src

    # Redirect shifted neighbor slots to their compact ghost indices.  `inverse`
    # maps each shifted edge's key back to its row in `unique_keys`.
    shifted_nlist_values = ghost_index.index_select(0, inverse)
    shifted_frames = frame_idx.index_select(0, shifted_edge_idx)
    shifted_centers = center_idx.index_select(0, shifted_edge_idx)
    shifted_slots = slot_idx.index_select(0, shifted_edge_idx)
    nlist[shifted_frames, shifted_centers, shifted_slots] = shifted_nlist_values
    return extended_coord, extended_atype, mapping, nlist
