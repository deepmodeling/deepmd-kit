# SPDX-License-Identifier: LGPL-3.0-or-later
"""Device-resident neighbor-list builders for the pt_expt inference path.

These are torch-specific helpers (kept out of the array-API ``dpmodel``
backend) that build the extended system and neighbor list with the
``vesin.torch`` cell list.  The neighbor search runs on the device of the
input coordinates (CPU or CUDA), so on a GPU the whole build stays on the GPU
and avoids a host round-trip.  The neighbor *search* is non-differentiable, so
using an external library here does not affect the model's autograd graph.
"""

import torch

from deepmd.dpmodel.utils.nlist import (
    nlist_distinguish_types,
)


def is_vesin_torch_available() -> bool:
    """Whether the device-capable ``vesin.torch`` neighbor list is importable."""
    try:
        import vesin.torch  # noqa: F401
    except ImportError:
        return False
    return True


def build_neighbor_list_vesin_torch(
    coord: torch.Tensor,
    box: torch.Tensor | None,
    atype: torch.Tensor,
    rcut: float,
    sel: list[int],
    distinguish_types: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the extended system and neighbor list with ``vesin.torch``.

    Parameters
    ----------
    coord : torch.Tensor
        local atom coordinates, shape (nframes, nloc, 3).
    box : torch.Tensor or None
        simulation cell, shape (nframes, 3, 3). ``None`` for non-periodic.
    atype : torch.Tensor
        atom types, shape (nframes, nloc).
    rcut : float
        cutoff radius.
    sel : list[int]
        maximal number of selected neighbors (summed over types).
    distinguish_types : bool
        whether to reorder the neighbor list per atom type.

    Returns
    -------
    extended_coord : torch.Tensor, shape (nframes, nall, 3)
    extended_atype : torch.Tensor, shape (nframes, nall)
    nlist : torch.Tensor, shape (nframes, nloc, sum(sel))
    mapping : torch.Tensor, shape (nframes, nall)
    """
    device = coord.device
    nframes = coord.shape[0]
    frame_results = [
        _build_neighbor_list_vesin_torch_single(
            coord[ff],
            box[ff] if box is not None else None,
            atype[ff],
            rcut,
            sel,
            distinguish_types,
        )
        for ff in range(nframes)
    ]
    max_nall = max(ec.shape[0] for ec, _, _, _ in frame_results)
    ext_coords, ext_atypes, nlists, mappings = [], [], [], []
    for ec, ea, nl, mp in frame_results:
        pad = max_nall - ec.shape[0]
        if pad > 0:
            ec = torch.cat(
                [ec, torch.zeros((pad, 3), dtype=ec.dtype, device=device)], dim=0
            )
            ea = torch.cat(
                [ea, torch.full((pad,), -1, dtype=ea.dtype, device=device)], dim=0
            )
            mp = torch.cat(
                [mp, torch.zeros((pad,), dtype=mp.dtype, device=device)], dim=0
            )
        ext_coords.append(ec)
        ext_atypes.append(ea)
        nlists.append(nl)
        mappings.append(mp)
    return (
        torch.stack(ext_coords, dim=0),
        torch.stack(ext_atypes, dim=0),
        torch.stack(nlists, dim=0),
        torch.stack(mappings, dim=0),
    )


def _build_neighbor_list_vesin_torch_single(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    atype: torch.Tensor,
    rcut: float,
    sel: list[int],
    distinguish_types: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-frame variant of :func:`build_neighbor_list_vesin_torch`."""
    import vesin.torch

    device = positions.device
    nsel = sum(sel)
    nloc = positions.shape[0]
    periodic = cell is not None
    box = (
        cell if periodic else torch.zeros((3, 3), dtype=positions.dtype, device=device)
    )

    # Pin the default device to the input's device: vesin.torch allocates some
    # internal tensors on the ambient default device, which may be a fake/other
    # device in some contexts (e.g. tests set a placeholder CUDA default).
    nl = vesin.torch.NeighborList(cutoff=rcut, full_list=True)
    with torch.device(device):
        ii, jj, ss = nl.compute(
            points=positions, box=box, periodic=periodic, quantities="ijS"
        )
    ii = ii.to(torch.int64)
    jj = jj.to(torch.int64)
    ss = ss.to(positions.dtype)

    # ghost atoms: neighbors reached through a non-zero periodic shift
    out_mask = torch.any(ss != 0, dim=1)
    out_idx = jj[out_mask]
    out_coords = positions[out_idx] + ss[out_mask] @ box
    nghost = int(out_idx.shape[0])

    extended_coord = torch.cat([positions, out_coords], dim=0)
    extended_atype = torch.cat([atype, atype[out_idx]], dim=0)
    mapping = torch.cat(
        [torch.arange(nloc, dtype=torch.int64, device=device), out_idx], dim=0
    )

    # remap neighbor column indices: ghosts -> [nloc, nloc + nghost)
    neigh_idx = jj.clone()
    neigh_idx[out_mask] = torch.arange(
        nloc, nloc + nghost, dtype=torch.int64, device=device
    )

    # group pairs by center atom (vesin does not guarantee CSR ordering)
    counts = torch.bincount(ii, minlength=nloc)
    max_nn = int(counts.max()) if counts.numel() > 0 else 0
    order = torch.argsort(ii, stable=True)
    rows = ii[order]
    cols = torch.arange(ii.shape[0], dtype=torch.int64, device=device) - (
        torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    )
    dense_idx = torch.full((nloc, max_nn), -1, dtype=torch.int64, device=device)
    if ii.shape[0] > 0:
        dense_idx[rows, cols] = neigh_idx[order]

    # sort candidates by distance, keep the nsel nearest within rcut, pad with -1
    valid = dense_idx >= 0
    lookup = torch.where(valid, dense_idx, torch.zeros_like(dense_idx))
    dists = torch.linalg.norm(extended_coord[lookup] - positions[:, None, :], dim=-1)
    valid &= dists <= rcut
    dists = torch.where(valid, dists, torch.full_like(dists, float("inf")))
    sort_order = torch.argsort(dists, dim=-1)
    sorted_idx = torch.take_along_dim(dense_idx, sort_order, dim=-1)
    sorted_valid = torch.take_along_dim(valid, sort_order, dim=-1)

    nlist = torch.full((nloc, nsel), -1, dtype=torch.int64, device=device)
    keep = min(nsel, max_nn)
    if keep > 0:
        nlist[:, :keep] = torch.where(
            sorted_valid[:, :keep],
            sorted_idx[:, :keep],
            torch.full_like(sorted_idx[:, :keep], -1),
        )

    if distinguish_types:
        nlist = nlist_distinguish_types(nlist[None], extended_atype[None], sel)[0]

    return extended_coord, extended_atype, nlist, mapping
