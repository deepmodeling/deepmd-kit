# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-agnostic edge-graph neighbor-list contract (NeighborGraph) and its
length policy (GraphLayout). See memory/spec_unified_edge_nlist.md.

Node validity (real vs padding) is NOT a stored field: it is derived as
``arange(N) < sum(n_node)`` because ``n_node`` already encodes the real-node
count and the layout is compact-prefix (real nodes first, padding suffix).
``edge_mask`` IS stored because there is no per-axis edge count to derive it from.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import array_api_compat

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import (
        Array,
    )


@dataclass
class NeighborGraph:
    """Edge-graph neighbor list. Node axis is flat ``N = sum(n_node)``.

    Geometry enters the model ONLY through ``edge_vec`` (the single autograd
    leaf). ``edge_index``/``angle_index`` use the SoA ``(2, .)`` layout so the
    src/dst index vectors are contiguous.
    """

    n_node: Array
    """(nf,) int  nodes per frame (single-rank: local atoms; multi-rank: local+halo)."""
    edge_index: Array
    """(2, E) int  [src, dst]; src = neighbor, dst = center; both in [0, N)."""
    edge_vec: Array
    """(E, 3) float  r_src - r_dst (neighbor - center); the only geometry / autograd leaf."""
    edge_mask: Array
    """(E,) bool  real (1) vs padding (0). Always stored (no n_edge to derive from)."""
    n_local: Array | None = None
    """(nf,) int  multi-rank owned-vs-halo split; owned = first n_local[f]. None = all local."""
    angle_index: Array | None = None
    """(2, A) int  [edge_a, edge_b] sharing a center; into [0, E). None if no angles."""
    angle_mask: Array | None = None
    """(A,) bool  real vs padding on the angle axis. None if no angles."""


@dataclass
class GraphLayout:
    """Length policy: the only torch/jax difference. None => dynamic axis (torch);
    int => static capacity (jax/paddle padding target).
    """

    edge_capacity: int | None = None
    angle_capacity: int | None = None
    node_capacity: int | None = None
    frame_capacity: int | None = None
    min_edges: int = 2


def pad_and_guard_edges(
    edge_index: Array,
    edge_vec: Array,
    capacity: int | None,
    min_edges: int = 2,
    pad_value: int = 0,
) -> tuple[Array, Array, Array]:
    """Append padding/guard edges as a contiguous suffix and build edge_mask.

    Real edges (``edge_index``/``edge_vec``) stay at the front (compact layout).
    - ``capacity is None`` (torch dynamic): append exactly ``min_edges`` masked
      dummy edges so the edge axis has a known lower bound and shape-stable
      guards for export.
    - ``capacity`` set (jax static): pad to ``E_max = capacity``; raise on overflow.
    Dummy edges point at node ``pad_value`` (in-range) with zero ``edge_vec``.
    """
    xp = array_api_compat.array_namespace(edge_index)
    dev = array_api_compat.device(edge_index)
    e_real = edge_index.shape[1]
    if capacity is None:
        target = e_real + min_edges
    else:
        if e_real > capacity:
            raise ValueError(
                f"edge overflow: {e_real} real edges > edge_capacity {capacity}"
            )
        target = capacity
    n_pad = target - e_real
    pad_idx = xp.full((2, n_pad), pad_value, dtype=edge_index.dtype, device=dev)
    pad_vec = xp.zeros((n_pad, 3), dtype=edge_vec.dtype, device=dev)
    ei = xp.concat([edge_index, pad_idx], axis=1)
    ev = xp.concat([edge_vec, pad_vec], axis=0)
    arange = xp.arange(target, dtype=edge_index.dtype, device=dev)
    edge_mask = arange < e_real
    return ei, ev, edge_mask


def node_validity_mask(n_node: Array, n_total: int) -> Array:
    """Derive the (n_total,) real-vs-padding node mask from per-frame counts.

    Compact-prefix layout: the first ``sum(n_node)`` nodes are real, the rest
    are padding. jit-safe (no Python ``int`` cast on the traced sum).
    """
    xp = array_api_compat.array_namespace(n_node)
    idx = xp.arange(n_total, dtype=n_node.dtype, device=array_api_compat.device(n_node))
    return idx < xp.sum(n_node)


def neighbor_graph_from_extended(
    extended_coord: Array,
    nlist: Array,
    mapping: Array,
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Convert the legacy extended quartet into a ghost-free NeighborGraph.

    This is the dpmodel/array-API adapter that REUSES deepmd's existing, tested,
    general-cell neighbor list (``build_neighbor_list`` / ``extend_coord_with_ghosts``)
    instead of re-deriving neighbors. For each valid neighbor slot it emits one
    edge with ``src = mapping[neighbor]`` (the neighbor's LOCAL owner -> ghost-free
    index), ``dst = center`` (local), and ``edge_vec = extended_coord[neighbor] -
    extended_coord[center]`` (the ghost coordinate already carries the periodic
    shift). Invalid slots (``nlist == -1``) are dropped. Nodes are flattened with a
    ``frame * nloc`` offset; the edge axis is padded/guarded via ``pad_and_guard_edges``.

    Because every neighbor maps to a LOCAL owner, the resulting graph is ghost-free:
    forces scatter to local atoms (periodic images of the same atom sum to one owner
    through the ``src`` index), so no ``edge_scatter_index`` is needed (single-rank).

    Parameters
    ----------
    extended_coord
        (nf, nall, 3) extended (local + ghost) coordinates.
    nlist
        (nf, nloc, nsel) neighbor list into the extended atoms; -1 is padding.
    mapping
        (nf, nall) extended -> local-owner index (local atoms map to themselves).
    layout
        edge-axis length policy; ``None`` => dynamic (torch) with ``min_edges`` guards.
    """
    if layout is None:
        layout = GraphLayout()
    xp = array_api_compat.array_namespace(extended_coord, nlist, mapping)
    dev = array_api_compat.device(extended_coord)
    nf = nlist.shape[0]
    nloc = nlist.shape[1]
    nsel = nlist.shape[2]
    n_node = xp.full((nf,), nloc, dtype=xp.int64, device=dev)
    src_parts: list[Array] = []
    dst_parts: list[Array] = []
    vec_parts: list[Array] = []
    center_full = xp.broadcast_to(
        xp.reshape(xp.arange(nloc, dtype=xp.int64, device=dev), (nloc, 1)),
        (nloc, nsel),
    )
    center_flat = xp.reshape(center_full, (nloc * nsel,))
    for ff in range(nf):
        nl_flat = xp.reshape(nlist[ff], (nloc * nsel,))
        keep = xp.reshape(xp.nonzero(nl_flat >= 0)[0], (-1,))
        j_ext = xp.take(nl_flat, keep, axis=0)  # extended neighbor indices
        dst = xp.take(center_flat, keep, axis=0)  # local center indices
        src = xp.take(mapping[ff], j_ext, axis=0)  # local owner of neighbor
        vec = xp.take(extended_coord[ff], j_ext, axis=0) - xp.take(
            extended_coord[ff], dst, axis=0
        )
        offset = ff * nloc
        src_parts.append(src + offset)
        dst_parts.append(dst + offset)
        vec_parts.append(vec)
    edge_index = xp.astype(
        xp.stack([xp.concat(src_parts), xp.concat(dst_parts)], axis=0), xp.int64
    )
    edge_vec = xp.concat(vec_parts, axis=0)
    edge_index, edge_vec, edge_mask = pad_and_guard_edges(
        edge_index, edge_vec, layout.edge_capacity, layout.min_edges
    )
    return NeighborGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
    )


def build_neighbor_graph(
    coord: Array,
    atype: Array,
    box: Array | None,
    rcut: float,
    sel: int | list[int],
    mixed_types: bool = True,
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Build a NeighborGraph by reusing the tested dense nlist (dpmodel default).

    Calls ``extend_input_and_build_neighbor_list`` (general-cell, tested) then
    :func:`neighbor_graph_from_extended`. With ``sel`` large enough that no real
    neighbor is truncated, the result is exactly the in-``rcut`` environment (the
    ``sel``-as-normalization regime; see memory/spec_unified_edge_nlist.md).
    """
    from deepmd.dpmodel.utils.nlist import (
        extend_input_and_build_neighbor_list,
    )

    extended_coord, _extended_atype, mapping, nlist = (
        extend_input_and_build_neighbor_list(
            coord, atype, rcut, sel, mixed_types=mixed_types, box=box
        )
    )
    return neighbor_graph_from_extended(extended_coord, nlist, mapping, layout)
