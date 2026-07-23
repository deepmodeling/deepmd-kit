# SPDX-License-Identifier: LGPL-3.0-or-later
"""Edge-vector neighbor-list helpers for SeZM-style models."""

from __future__ import (
    annotations,
)

import torch

from deepmd.dpmodel.utils.neighbor_list import (
    EdgeNeighborList,
)

_DUMMY_EDGE_COUNT = 2


def _append_dummy_edges(
    edge_index: torch.Tensor,
    edge_vec: torch.Tensor,
    edge_scatter_index: torch.Tensor,
) -> EdgeNeighborList:
    """Append masked in-range edges so exported graphs never see empty inputs."""
    device = edge_index.device
    dummy_index = torch.zeros(
        (2, _DUMMY_EDGE_COUNT),
        dtype=edge_index.dtype,
        device=device,
    )
    dummy_vec = torch.zeros(
        (_DUMMY_EDGE_COUNT, 3),
        dtype=edge_vec.dtype,
        device=device,
    )
    edge_index = torch.cat([edge_index, dummy_index], dim=1)
    edge_vec = torch.cat([edge_vec, dummy_vec], dim=0)
    edge_scatter_index = torch.cat([edge_scatter_index, dummy_index], dim=1)
    edge_mask = torch.cat(
        [
            torch.ones(
                edge_vec.shape[0] - _DUMMY_EDGE_COUNT, dtype=torch.bool, device=device
            ),
            torch.zeros(_DUMMY_EDGE_COUNT, dtype=torch.bool, device=device),
        ]
    )
    # ``coord`` and ``atype`` are filled by public constructors.
    return EdgeNeighborList(
        coord=torch.empty(0, dtype=edge_vec.dtype, device=device),
        atype=torch.empty(0, dtype=torch.long, device=device),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_scatter_index=edge_scatter_index,
        edge_mask=edge_mask,
    )


def edge_schema_from_extended(
    coord: torch.Tensor,
    atype: torch.Tensor,
    nlist: torch.Tensor,
    mapping: torch.Tensor | None,
    *,
    scatter_to_local: bool = False,
) -> EdgeNeighborList:
    """Build the unified edge schema from an extended-coordinate neighbor list.

    This is the zero-shift form used by callers that already have periodic-image
    coordinates, such as LAMMPS ghost atoms or the native fallback builder.

    Contract: ``nlist`` is assumed to be already truncated to the model cutoff
    before this conversion is used.  DeePMD's native builders select within
    ``rcut``; exported LAMMPS nlist paths keep cutoff filtering inside the
    traced model graph.
    Unlike the candidate-list builders (:func:`edge_schema_from_neighbor_matrix`,
    :func:`edge_schema_from_ij_shifts`), this function therefore applies no
    ``edge_len <= rcut`` upper bound -- doing so would be a redundant op on the
    native training hot path, and any residual out-of-range edge is already
    zeroed by the descriptor's smooth cutoff envelope.
    """
    nf, nloc, nsel = nlist.shape
    device = coord.device
    nall = coord.shape[1]

    neighbor_flat = nlist.reshape(-1)
    dst_actual = (
        torch.arange(neighbor_flat.shape[0], device=device, dtype=torch.long) // nsel
    )
    frame_idx = dst_actual // nloc
    dst_local = dst_actual % nloc
    valid_flat = neighbor_flat >= 0
    neighbor_safe = torch.where(
        valid_flat, neighbor_flat, torch.zeros_like(neighbor_flat)
    )
    neighbor_safe_2d = neighbor_safe.to(dtype=torch.long).view(nf, nloc * nsel)

    neighbor_coord = torch.gather(
        coord,
        1,
        neighbor_safe_2d.unsqueeze(-1).expand(-1, -1, 3),
    ).reshape(-1, 3)
    dst_coord = torch.gather(
        coord[:, :nloc, :],
        1,
        dst_local.view(nf, -1).unsqueeze(-1).expand(-1, -1, 3),
    ).reshape(-1, 3)
    edge_vec_all = neighbor_coord - dst_coord
    edge_len2 = torch.sum(edge_vec_all * edge_vec_all, dim=-1)

    if mapping is None:
        src_local = neighbor_safe.to(dtype=torch.long)
    else:
        src_local = torch.gather(mapping, 1, neighbor_safe_2d).reshape(-1)
    src_actual = frame_idx * nloc + src_local.to(dtype=torch.long)
    src_scatter = frame_idx * nall + neighbor_safe.to(dtype=torch.long)
    dst_scatter = frame_idx * nall + dst_local

    # No ``edge_len2 <= rcut**2`` upper bound here: ``nlist`` is contractually
    # cutoff-truncated by the caller (see the docstring). Only padding (-1),
    # ghost-only neighbours, and coincident pairs are dropped.
    edge_keep = valid_flat & (src_local >= 0) & (src_local < nloc) & (edge_len2 > 1e-10)
    valid_idx = torch.nonzero(edge_keep, as_tuple=False).flatten()
    edge_index = torch.stack(
        [
            src_actual.index_select(0, valid_idx),
            dst_actual.index_select(0, valid_idx),
        ],
        dim=0,
    )
    if scatter_to_local:
        edge_scatter_index = edge_index
    else:
        edge_scatter_index = torch.stack(
            [
                src_scatter.index_select(0, valid_idx),
                dst_scatter.index_select(0, valid_idx),
            ],
            dim=0,
        )
    schema = _append_dummy_edges(
        edge_index,
        edge_vec_all.index_select(0, valid_idx),
        edge_scatter_index,
    )
    schema.coord = coord[:, :nloc, :].contiguous() if scatter_to_local else coord
    # The local-atom slice is a stride-(nall, 1) view when nloc < nall (always so
    # with ghost atoms, and for the spin path where the source carries 2*nall
    # columns). The compiled core flattens ``atype`` via ``reshape(-1)``, which
    # ``torch.compile`` lowers to ``aten.view`` and rejects on a non-contiguous
    # layout under symbolic shapes. Materialize a contiguous copy here, mirroring
    # ``coord`` above.
    schema.atype = atype[:, :nloc].contiguous()
    return schema


def edge_schema_from_neighbor_matrix(
    coord: torch.Tensor,
    atype: torch.Tensor,
    cell: torch.Tensor | None,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
    rcut: float,
) -> EdgeNeighborList:
    """Build edge schema from a dense neighbor matrix and integer shifts."""
    nf, nloc = atype.shape[:2]
    total_atoms, max_neighbors = neighbor_matrix.shape
    device = coord.device
    slot = torch.arange(max_neighbors, dtype=torch.long, device=device).expand(
        total_atoms, max_neighbors
    )
    valid = (slot < num_neighbors.unsqueeze(1)).reshape(-1)
    edge_idx = torch.nonzero(valid, as_tuple=False).flatten()
    if edge_idx.numel() == 0:
        empty = _append_dummy_edges(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0, 3), dtype=coord.dtype, device=device),
            torch.zeros((2, 0), dtype=torch.long, device=device),
        )
        empty.coord = coord
        empty.atype = atype
        return empty

    dst = edge_idx // max_neighbors
    src = neighbor_matrix.reshape(-1).index_select(0, edge_idx).to(dtype=torch.long)
    shift = shifts.reshape(-1, 3).index_select(0, edge_idx)
    src_local = src % nloc
    frame_idx = dst // nloc
    src_actual = frame_idx * nloc + src_local
    coord_flat = coord.reshape(nf * nloc, 3)
    edge_vec_all = coord_flat.index_select(0, src_actual) - coord_flat.index_select(
        0, dst
    )

    if cell is not None:
        shifted_idx = torch.nonzero(
            torch.any(shift != 0, dim=1), as_tuple=False
        ).flatten()
        if shifted_idx.numel() > 0:
            shift_cart = torch.bmm(
                shift.index_select(0, shifted_idx).to(dtype=coord.dtype).unsqueeze(1),
                cell.index_select(0, frame_idx.index_select(0, shifted_idx)),
            ).squeeze(1)
            edge_vec_all.index_add_(0, shifted_idx, shift_cart)

    edge_len2 = torch.sum(edge_vec_all * edge_vec_all, dim=-1)
    edge_keep = (edge_len2 > 1e-10) & (edge_len2 <= float(rcut) * float(rcut))
    valid_idx = torch.nonzero(edge_keep, as_tuple=False).flatten()
    schema = _append_dummy_edges(
        torch.stack(
            [
                src_actual.index_select(0, valid_idx),
                dst.index_select(0, valid_idx),
            ],
            dim=0,
        ),
        edge_vec_all.index_select(0, valid_idx),
        torch.stack(
            [
                src_actual.index_select(0, valid_idx),
                dst.index_select(0, valid_idx),
            ],
            dim=0,
        ),
    )
    schema.coord = coord
    schema.atype = atype
    return schema


def edge_schema_from_ij_shifts(
    positions: torch.Tensor,
    atype: torch.Tensor,
    cell: torch.Tensor | None,
    ii: torch.Tensor,
    jj: torch.Tensor,
    shifts: torch.Tensor,
    rcut: float,
) -> EdgeNeighborList:
    """Build a single-frame edge schema from ``vesin`` ``(i, j, S)`` output."""
    device = positions.device
    nloc = positions.shape[0]
    if ii.numel() == 0:
        empty = _append_dummy_edges(
            torch.zeros((2, 0), dtype=torch.long, device=device),
            torch.zeros((0, 3), dtype=positions.dtype, device=device),
            torch.zeros((2, 0), dtype=torch.long, device=device),
        )
        empty.coord = positions.reshape(1, nloc, 3)
        empty.atype = atype.reshape(1, nloc)
        return empty

    ii = ii.to(dtype=torch.long)
    jj = jj.to(dtype=torch.long)
    shifts = shifts.to(dtype=positions.dtype)
    edge_vec_all = positions.index_select(0, jj) - positions.index_select(0, ii)
    if cell is not None:
        shifted_idx = torch.nonzero(
            torch.any(shifts != 0, dim=1), as_tuple=False
        ).flatten()
        if shifted_idx.numel() > 0:
            # Image offset ``S @ cell`` as a broadcast multiply-reduce: the
            # (n_shift, 3) @ (3, 3) matmul otherwise dispatches to an fp64 GEMM
            # kernel whose length-3 contraction is catastrophically inefficient,
            # while this stays bit-identical.
            sel_shifts = shifts.index_select(0, shifted_idx)
            edge_vec_all.index_add_(
                0,
                shifted_idx,
                (sel_shifts[:, :, None] * cell).sum(1),
            )
    edge_len2 = torch.sum(edge_vec_all * edge_vec_all, dim=-1)
    edge_keep = (edge_len2 > 1e-10) & (edge_len2 <= float(rcut) * float(rcut))
    valid_idx = torch.nonzero(edge_keep, as_tuple=False).flatten()
    schema = _append_dummy_edges(
        torch.stack(
            [
                jj.index_select(0, valid_idx),
                ii.index_select(0, valid_idx),
            ],
            dim=0,
        ),
        edge_vec_all.index_select(0, valid_idx),
        torch.stack(
            [
                jj.index_select(0, valid_idx),
                ii.index_select(0, valid_idx),
            ],
            dim=0,
        ),
    )
    schema.coord = positions.reshape(1, nloc, 3)
    schema.atype = atype.reshape(1, nloc)
    return schema


def merge_frame_edge_schemas(frames: list[EdgeNeighborList]) -> EdgeNeighborList:
    """Merge per-frame local edge schemas into a batched flattened schema."""
    if not frames:
        raise ValueError("at least one frame schema is required")
    coord = torch.cat([frame.coord for frame in frames], dim=0)
    atype = torch.cat([frame.atype for frame in frames], dim=0)
    nloc = coord.shape[1]
    edge_indices: list[torch.Tensor] = []
    edge_vecs: list[torch.Tensor] = []
    scatter_indices: list[torch.Tensor] = []
    for frame_idx, frame in enumerate(frames):
        real = frame.edge_mask
        offset = frame_idx * nloc
        edge_indices.append(frame.edge_index[:, real] + offset)
        scatter_indices.append(frame.edge_scatter_index[:, real] + offset)
        edge_vecs.append(frame.edge_vec[real])
    edge_index = torch.cat(edge_indices, dim=1)
    edge_vec = torch.cat(edge_vecs, dim=0)
    edge_scatter_index = torch.cat(scatter_indices, dim=1)
    schema = _append_dummy_edges(edge_index, edge_vec, edge_scatter_index)
    schema.coord = coord
    schema.atype = atype
    return schema
