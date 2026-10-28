# SPDX-License-Identifier: LGPL-3.0-or-later
"""Vesin graph builders for differentiable and compact deployment inputs.

Generic graphs recompute displacements from local neighbor pairs and periodic
image shifts, preserving coordinate and cell derivatives. Compact canonical
graphs use Vesin's detached displacements for exported models with explicit
forces. Both layouts retain every active neighbor within the exact cutoff.

Vesin's API is single-system, so this builder loops over frames in Python.
It is intended for ``nf == 1`` inference and CPU use. Inference ``neighbor_graph_method="auto"``
(:func:`~deepmd.pt_expt.utils.graph_builder.resolve_auto_graph_builder`) prefers
vesin for single-frame CUDA inputs; multi-frame batches stay on ``nv``/``dense``.
Training auto never selects vesin. Prefer ``nv`` (:mod:`.nv_graph_builder`) for batched
multi-frame GPU work, which batches all frames in one kernel.
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
from deepmd.pt_expt.utils.canonical_graph import (
    CanonicalGraph,
    canonical_graph_from_pairs,
)
from deepmd.pt_expt.utils.vesin_neighbor_list import (
    is_vesin_torch_available,
)
from deepmd.pt_expt.utils.vesin_search import (
    VesinNeighborSearch,
)


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
    search: VesinNeighborSearch | None = None,
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
    search
        Optional persistent search shared across evaluations. Its skin only
        controls candidate reuse; the returned graph retains the exact cutoff.

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
            torch.full((nf,), nloc, dtype=torch.int64, device=dev),
            layout=layout,
            with_csr=with_csr,
            canonicalize=canonicalize,
        )

    if search is None:
        search = VesinNeighborSearch(
            sorted=(with_csr or canonicalize) and torch.device(dev).type != "cuda"
        )
    i_parts, j_parts, S_parts, nf_parts = [], [], [], []
    for f in range(nf):
        cell_f = box[f] if periodic else None
        ii, jj, ss = search.compute(coord[f], cell_f, rcut, frame=f)
        i_parts.append(ii)
        j_parts.append(jj)
        S_parts.append(ss)
        nf_parts.append(torch.full((ii.shape[0],), f, dtype=torch.int64, device=dev))

    # A single frame already owns contiguous edge arrays. Empty batches retain
    # the same empty-array contract as the backend-independent ASE builder.
    if nf == 1:
        i_all, j_all, S_all, nf_all = i_parts[0], j_parts[0], S_parts[0], nf_parts[0]
    elif nf > 1:
        i_all = torch.cat(i_parts)
        j_all = torch.cat(j_parts)
        S_all = torch.cat(S_parts)
        nf_all = torch.cat(nf_parts)
    else:
        i_all = torch.zeros((0,), dtype=torch.int64, device=dev)
        j_all = torch.zeros((0,), dtype=torch.int64, device=dev)
        S_all = torch.zeros((0, 3), dtype=torch.int64, device=dev)
        nf_all = torch.zeros((0,), dtype=torch.int64, device=dev)

    # virtual atoms (atype < 0) are excluded as centers AND neighbors — the
    # World-2 builder contract shared with the dense reference builder; the
    # geometric search above cannot know about them.
    at = torch.as_tensor(atype, device=dev).reshape(nf, nloc)
    keep = (at[nf_all, i_all] >= 0) & (at[nf_all, j_all] >= 0)
    indices = torch.nonzero(keep, as_tuple=False).flatten()
    i_all = i_all.index_select(0, indices)
    j_all = j_all.index_select(0, indices)
    nf_all = nf_all.index_select(0, indices)
    S_all = S_all.index_select(0, indices)

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
        torch.full((nf,), nloc, dtype=torch.int64, device=dev),
        layout=layout,
    )
    if pair_excl is not None:
        at_flat = torch.as_tensor(atype, device=dev).reshape(-1)
        graph = apply_pair_exclusion(graph, at_flat, pair_excl, compact=compact)
    if with_csr or canonicalize:
        graph = attach_edge_csr(
            graph,
            nf * nloc,
            canonicalize=canonicalize,
            destination_sorted=search.sorted and (pair_excl is None or compact),
        )
    return graph


def build_canonical_graph_vesin(
    coord: torch.Tensor,
    box: torch.Tensor | None,
    rcut: float,
    search: VesinNeighborSearch,
    node_mask: torch.Tensor | None = None,
) -> CanonicalGraph:
    """Build the compact deployment graph directly from Vesin displacements.

    Parameters
    ----------
    coord : torch.Tensor
        Coordinates in angstrom with shape (nf, nloc, 3).
    box : torch.Tensor or None
        Cells in angstrom with shape (nf, 3, 3), or non-periodic boundaries.
    rcut : float
        Neighbor cutoff radius in angstrom.
    search : VesinNeighborSearch
        Persistent search with independent candidate storage for each frame.
    node_mask : torch.Tensor or None
        Active-node mask with shape (nf, nloc), on the coordinate device.
        None includes all nodes. Inactive nodes remain in the node axis but
        have no incoming or outgoing edges.

    Returns
    -------
    CanonicalGraph
        Local graph with exact-cutoff float32 geometry and dual CSR. Geometry
        is detached; this builder serves exported models with explicit forces.
    """
    nf, nloc, _ = coord.shape
    centers, sources, vectors = [], [], []
    for frame in range(nf):
        if nloc == 0:
            break
        center, source, edge_vec = search.compute(
            coord[frame],
            box[frame] if box is not None else None,
            rcut,
            frame=frame,
            return_vectors=True,
        )
        if node_mask is not None:
            active = node_mask[frame, center] & node_mask[frame, source]
            indices = torch.nonzero(active, as_tuple=False).flatten()
            center = center.index_select(0, indices)
            source = source.index_select(0, indices)
            edge_vec = edge_vec.index_select(0, indices)
        if frame:
            center = center + frame * nloc
            source = source + frame * nloc
        centers.append(center)
        sources.append(source)
        vectors.append(edge_vec)
    if not centers:
        center = torch.empty(0, dtype=torch.int64, device=coord.device)
        source = torch.empty_like(center)
        edge_vec = coord.new_empty((0, 3))
    elif nf > 1:
        center = torch.cat(centers)
        source = torch.cat(sources)
        edge_vec = torch.cat(vectors)
    return canonical_graph_from_pairs(
        center,
        source,
        edge_vec,
        torch.full((nf,), nloc, dtype=torch.int64, device=coord.device),
        nf * nloc,
    )
