# SPDX-License-Identifier: LGPL-3.0-or-later
"""Carry-all NeighborGraph builder backed by vesin.torch (O(N) cell list).

World-2 counterpart of vesin_neighbor_list.py: instead of building the dense
quartet, it returns per-frame local (i, j, S), then delegates to the array-API
``neighbor_graph_from_ijs`` (which recomputes ``edge_vec`` differentiably from
the ORIGINAL grad-carrying coords). torch-only => lives in pt_expt.

Scope note: ``vesin.torch``'s API is single-system, so this builder LOOPS over
frames in Python (~1 ms/frame call overhead measured on GPU). It is intended
for ``nf == 1`` inference and CPU use. On CPU (and on CUDA when ``nv`` is
unavailable) it is selected by the shared
:func:`~deepmd.pt_expt.utils.neighbor_graph_method.resolve_auto_graph_builder`
default ladder; otherwise prefer ``nv`` (:mod:`.nv_graph_builder`) for batched
multi-frame GPU work, which batches all frames in one kernel. Export and
training compile still use synthetic dense graph inputs — the builder choice
does not affect ``.pt2`` artifacts.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

if TYPE_CHECKING:
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )

import array_api_compat
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    NeighborGraph,
    apply_pair_exclusion,
    attach_edge_csr,
    neighbor_graph_from_ijs,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
)


def vesin_search_ijs(
    positions: torch.Tensor,
    cell: torch.Tensor | None,
    periodic: bool,
    rcut: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw ``vesin.torch`` neighbor search returning ``(ii, jj, ss)`` as int64.

    The caller is responsible for ensuring ``vesin.torch`` is importable (check
    :func:`~deepmd.pt_expt.utils.vesin_neighbor_list.is_vesin_torch_available`
    before calling).  ``positions`` must be detached (the search is
    non-differentiable).

    Parameters
    ----------
    positions : (nloc, 3) local-frame coordinates, already detached.
    cell : (3, 3) box matrix for periodic systems, or ``None`` for non-periodic.
        For non-periodic systems a zero box is constructed internally.
    periodic : whether the system is periodic.
    rcut : neighbor cutoff radius.
    device : device to pin as the ambient default.  ``vesin.torch`` allocates
        some internal tensors on the ambient default device, which may be a
        fake/other device in some test contexts (e.g. a placeholder CUDA
        default); pinning it here prevents spurious CUDA initializations.

    Returns
    -------
    ii : (E,) int64 center local indices.
    jj : (E,) int64 neighbor local indices.
    ss : (E, 3) int64 periodic image shifts.
    """
    import vesin.torch as _vesin_torch

    box = (
        cell if periodic else torch.zeros((3, 3), dtype=positions.dtype, device=device)
    )
    nl = _vesin_torch.NeighborList(cutoff=float(rcut), full_list=True)
    with torch.device(device):
        ii, jj, ss = nl.compute(
            points=positions,
            box=box,
            periodic=periodic,
            quantities="ijS",
        )
    return ii.to(torch.int64), jj.to(torch.int64), ss.to(torch.int64).reshape(-1, 3)


def build_neighbor_graph_vesin(
    coord: Any,
    atype: Any,
    box: Any | None,
    rcut: float,
    layout: GraphLayout | None = None,
    *,
    with_csr: bool = False,
    canonicalize: bool = False,
    pair_excl: PairExcludeMask | None = None,
    compact: bool = False,
) -> NeighborGraph:
    """Build a CARRY-ALL NeighborGraph using vesin.torch's O(N) cell list.

    Mirrors :func:`deepmd.dpmodel.utils.neighbor_graph.build_neighbor_graph_ase`
    but runs on the input tensor's device via ``vesin.torch``.

    Parameters
    ----------
    coord : torch.Tensor
        Coordinates with shape ``(nf, nloc, 3)``.
    atype : torch.Tensor
        Atom types with shape ``(nf, nloc)``.
    box : torch.Tensor or None
        Simulation cells with shape ``(nf, 3, 3)``.
    rcut : float
        Cutoff radius.
    layout : GraphLayout or None
        Edge-axis length policy.
    with_csr : bool
        Whether to construct destination/source CSR views.
    canonicalize : bool
        Whether to reorder every edge field into destination-major form. Implies
        ``with_csr=True``.
    pair_excl
        Optional :class:`~deepmd.dpmodel.utils.neighbor_graph.graph.PairExcludeMask`
        for model-level ``pair_exclude_types``. When given,
        :func:`apply_pair_exclusion` is applied after the geometric search. ``None``
        (default) leaves all geometrically valid edges present.
    compact
        Passed to :func:`apply_pair_exclusion`; see that function for details.
        Ignored when ``pair_excl`` is ``None``.

    Returns
    -------
    graph
        The carry-all :class:`NeighborGraph` over the LOCAL atoms.
    """
    if not is_vesin_torch_available():
        raise ImportError(
            "build_neighbor_graph_vesin requires vesin[torch]; "
            "install with `pip install vesin[torch]` or use neighbor_graph_method='dense'."
        )

    xp = array_api_compat.array_namespace(coord)
    dev = array_api_compat.device(coord)
    nf = coord.shape[0] if coord.ndim == 3 else 1
    coord = xp.reshape(coord, (nf, -1, 3))
    nloc = coord.shape[1]
    periodic = box is not None
    if periodic:
        box = xp.reshape(box, (nf, 3, 3))

    if nloc == 0:
        empty_i = torch.zeros((0,), dtype=torch.int64, device=dev)
        empty_S = torch.zeros((0, 3), dtype=torch.int64, device=dev)
        empty_nf = torch.zeros((0,), dtype=torch.int64, device=dev)
        return neighbor_graph_from_ijs(
            empty_i,
            empty_i,
            empty_S,
            coord,
            box,
            empty_nf,
            nloc,
            layout=layout,
            with_csr=with_csr,
            canonicalize=canonicalize,
        )

    i_parts, j_parts, S_parts, nf_parts = [], [], [], []
    for f in range(nf):
        pts = coord[f].detach()
        cell_f = box[f].detach() if periodic else None
        ii, jj, ss = vesin_search_ijs(pts, cell_f, periodic, rcut, dev)
        i_parts.append(ii)
        j_parts.append(jj)
        S_parts.append(ss)
        nf_parts.append(torch.full((ii.shape[0],), f, dtype=torch.int64, device=dev))

    # guard torch.cat against empty part lists (nf == 0), mirroring ase_builder
    i_all = (
        torch.cat(i_parts)
        if i_parts
        else torch.zeros((0,), dtype=torch.int64, device=dev)
    )
    j_all = (
        torch.cat(j_parts)
        if j_parts
        else torch.zeros((0,), dtype=torch.int64, device=dev)
    )
    S_all = (
        torch.cat(S_parts)
        if S_parts
        else torch.zeros((0, 3), dtype=torch.int64, device=dev)
    )
    nf_all = (
        torch.cat(nf_parts)
        if nf_parts
        else torch.zeros((0,), dtype=torch.int64, device=dev)
    )

    # virtual atoms (atype < 0) are excluded as centers AND neighbors — the
    # World-2 builder contract shared with the dense reference builder; the
    # geometric search above cannot know about them.
    at = torch.as_tensor(atype, device=dev).reshape(nf, nloc)
    keep = (at[nf_all, i_all] >= 0) & (at[nf_all, j_all] >= 0)
    i_all, j_all, nf_all = i_all[keep], j_all[keep], nf_all[keep]
    S_all = S_all[keep]

    # i = center (dst), j = neighbor (src); pass ORIGINAL coord/box
    # (grad-carrying). Unlike the nv builder, vesin's cell list handles
    # out-of-cell (unwrapped) positions natively, so no normalize_coord is
    # needed and S is consistent with the original coords as searched.
    graph = neighbor_graph_from_ijs(
        i_all,
        j_all,
        S_all,
        coord,
        box,
        nf_all,
        nloc,
        layout=layout,
    )
    if pair_excl is not None:
        at_flat = torch.as_tensor(atype, device=dev).reshape(-1)
        graph = apply_pair_exclusion(graph, at_flat, pair_excl, compact=compact)
    if with_csr or canonicalize:
        graph = attach_edge_csr(graph, nf * nloc, canonicalize=canonicalize)
    return graph
