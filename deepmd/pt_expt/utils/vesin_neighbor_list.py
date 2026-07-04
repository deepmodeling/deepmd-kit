# SPDX-License-Identifier: LGPL-3.0-or-later
"""Device-resident O(N) neighbor-list strategy backed by ``vesin.torch``.

This provides a :class:`~deepmd.dpmodel.utils.neighbor_list.NeighborList`
strategy that replaces the historical all-pairs ghost expansion (~27*N images +
a dense ``[N, 27N]`` distance matrix) with a cell list.  ``vesin.torch`` returns
an ``(i, j, S)`` edge list (local neighbor index ``j`` plus integer periodic
image ``S``); we materialize only the *real-neighbor* ghosts ``coord[j] + S@box``
and hand back the standard extended quartet ``(extended_coord, extended_atype,
nlist, mapping)``, so the rest of the model is unchanged.

The neighbor search runs on the device of the input coordinates (CPU or CUDA),
so on a GPU the whole build stays on the GPU.  When the inputs are numpy arrays
(the array-API ``dpmodel`` backend) the build is bridged through a CPU torch
tensor and the result converted back to numpy.  The search itself is
non-differentiable -- it runs on detached coordinates -- while the returned
``extended_coord`` is rebuilt from the (possibly grad-carrying) input
coordinates so autograd for forces/virials flows through unchanged.
"""

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from deepmd.dpmodel.utils.neighbor_list import (
    EdgeNeighborList,
    NeighborList,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask
from deepmd.pt_expt.utils.edge_schema import (
    edge_schema_from_ij_shifts,
    merge_frame_edge_schemas,
)


def is_vesin_torch_available() -> bool:
    """Whether the device-capable ``vesin.torch`` neighbor list is importable."""
    try:
        import vesin.torch  # noqa: F401
    except ImportError:
        return False
    return True


class VesinNeighborList(NeighborList):
    """O(N) neighbor-list strategy using the ``vesin.torch`` cell list.

    Implements the :class:`~deepmd.dpmodel.utils.neighbor_list.NeighborList`
    interface.  Works on torch tensors (on their own device) and on numpy arrays
    (bridged through a CPU torch tensor); the returned quartet matches the
    namespace and device of the input coordinates.
    """

    def build(
        self,
        coord: Any,
        atype: Any,
        box: Any,
        rcut: float,
        sel: list[int],
        return_mode: str = "extended",
        pair_excl: "PairExcludeMask | None" = None,
    ) -> tuple[Any, Any, Any, Any] | EdgeNeighborList:
        """Build the extended system + candidate neighbor list with vesin.

        See :meth:`deepmd.dpmodel.utils.neighbor_list.NeighborList.build`.  The
        returned ``nlist`` is distance-sorted and truncated to ``sum(sel)``
        (matching the default builder); the lower interface still re-formats /
        type-splits it.

        Parameters
        ----------
        pair_excl : PairExcludeMask or None, optional
            When provided, excluded type pairs are erased from the returned
            neighbor list (entries set to ``-1``) by
            :func:`~deepmd.dpmodel.utils.nlist.apply_pair_exclusion_nlist`.
            The atomic-model seam applies the same filter as an idempotent
            backstop, so passing ``pair_excl`` here is a build-time
            optimization that avoids re-scanning per forward call.
        """
        is_numpy = not isinstance(coord, torch.Tensor)
        # vesin runs on the device of the inputs: numpy (the dpmodel backend) is
        # bridged through CPU torch; torch tensors stay on their own device.  Pin
        # the ambient default device (cf. the ``with torch.device(...)`` guard
        # around ``nl.compute`` below) so ``as_tensor`` is not affected by a
        # placeholder default device -- e.g. tests set a CUDA default, under
        # which a device-less ``as_tensor`` triggers CUDA init even for CPU input.
        device = torch.device("cpu") if is_numpy else coord.device
        with torch.device(device):
            coord_t = torch.as_tensor(coord)
            atype_t = torch.as_tensor(atype).to(torch.int64)
            box_t = None if box is None else torch.as_tensor(box, dtype=coord_t.dtype)

        nframes = atype_t.shape[0]
        nloc = atype_t.shape[1]
        coord_t = coord_t.reshape(nframes, nloc, 3)
        if box_t is not None:
            box_t = box_t.reshape(nframes, 3, 3)

        if return_mode == "edges":
            frame_edges = [
                _build_single_edges(
                    coord_t[ff],
                    box_t[ff] if box_t is not None else None,
                    atype_t[ff],
                    rcut,
                    sel,
                )
                for ff in range(nframes)
            ]
            schema = merge_frame_edge_schemas(frame_edges)
            if is_numpy:
                return EdgeNeighborList(
                    coord=schema.coord.detach().cpu().numpy(),
                    atype=schema.atype.cpu().numpy(),
                    edge_index=schema.edge_index.cpu().numpy(),
                    edge_vec=schema.edge_vec.detach().cpu().numpy(),
                    edge_scatter_index=schema.edge_scatter_index.cpu().numpy(),
                    edge_mask=schema.edge_mask.cpu().numpy(),
                )
            return schema
        if return_mode != "extended":
            raise ValueError(f"Unsupported neighbor-list return_mode: {return_mode!r}")

        frame_results = [
            _build_single(
                coord_t[ff],
                box_t[ff] if box_t is not None else None,
                atype_t[ff],
                rcut,
                sel,
            )
            for ff in range(nframes)
        ]
        max_nall = max(ec.shape[0] for ec, _, _, _ in frame_results)
        device = coord_t.device
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
        extended_coord = torch.stack(ext_coords, dim=0)
        extended_atype = torch.stack(ext_atypes, dim=0)
        nlist = torch.stack(nlists, dim=0)
        mapping = torch.stack(mappings, dim=0)

        if pair_excl is not None:
            from deepmd.dpmodel.utils.nlist import (
                apply_pair_exclusion_nlist,
            )

            nlist = apply_pair_exclusion_nlist(nlist, extended_atype, pair_excl)

        if is_numpy:
            return (
                extended_coord.detach().cpu().numpy(),
                extended_atype.cpu().numpy(),
                nlist.cpu().numpy(),
                mapping.cpu().numpy(),
            )
        return extended_coord, extended_atype, nlist, mapping


def _build_single(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    atype: torch.Tensor,
    rcut: float,
    sel: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single-frame ``(i,j,S)`` -> minimal-extended conversion.

    The cell list runs on detached coordinates (the search is
    non-differentiable); the returned ``extended_coord`` is rebuilt from
    ``positions`` so gradients flow to the local atoms and box.
    """
    device = positions.device
    nsel = sum(sel)
    nloc = positions.shape[0]

    # Empty system: vesin rejects an empty `points` array ("NULL pointer").
    # Return an empty extended representation directly, matching the native
    # builder's handling of a zero-atom frame.
    if nloc == 0:
        return (
            positions,
            atype,
            torch.full((0, nsel), -1, dtype=torch.int64, device=device),
            torch.zeros((0,), dtype=torch.int64, device=device),
        )

    periodic = cell is not None
    box = (
        cell if periodic else torch.zeros((3, 3), dtype=positions.dtype, device=device)
    )

    # Delegate the raw search to the shared helper in vesin_graph_builder
    # (function-level import: legacy module depends on graph module lazily to
    # avoid a module-level cycle — vesin_graph_builder imports
    # is_vesin_torch_available from this module).
    from deepmd.pt_expt.utils.vesin_graph_builder import (
        vesin_search_ijs,
    )

    ii, jj, ss = vesin_search_ijs(
        positions.detach(), cell if periodic else None, periodic, rcut, device
    )
    # ss is int64 from the helper; cast to float here for later ``ss @ box`` math.
    ss = ss.to(positions.dtype)

    # ghost atoms: neighbors reached through a non-zero periodic shift.  Rebuild
    # their coordinates from the grad-carrying `positions`/`box` so autograd for
    # forces/virials flows through the extended coordinates unchanged.
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
    dists = torch.linalg.norm(
        extended_coord.detach()[lookup] - positions.detach()[:, None, :], dim=-1
    )
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

    return extended_coord, extended_atype, nlist, mapping


def _build_single_edges(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    atype: torch.Tensor,
    rcut: float,
    sel: list[int],
) -> EdgeNeighborList:
    """Single-frame ``vesin`` output converted directly to edge vectors."""
    device = positions.device
    nsel = sum(sel)
    nloc = positions.shape[0]
    if nloc == 0:
        return edge_schema_from_ij_shifts(
            positions,
            atype,
            cell,
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(0, dtype=torch.int64, device=device),
            torch.zeros(0, 3, dtype=positions.dtype, device=device),
            rcut,
        )

    periodic = cell is not None
    from deepmd.pt_expt.utils.vesin_graph_builder import (
        vesin_search_ijs,
    )

    ii, jj, ss = vesin_search_ijs(
        positions.detach(), cell if periodic else None, periodic, rcut, device
    )
    # ss is int64 from the helper; edge_schema_from_ij_shifts accepts int shifts.
    return edge_schema_from_ij_shifts(
        positions=positions,
        atype=atype,
        cell=cell,
        ii=ii,
        jj=jj,
        shifts=ss,
        rcut=rcut,
    )
