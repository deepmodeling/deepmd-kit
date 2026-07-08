# SPDX-License-Identifier: LGPL-3.0-or-later
"""3-body angle graph: pairs of edges sharing a center within a_rcut.

Angles reference EDGES (angle_index into [0,E)); edge_vec stays the only
geometry leaf. a_sel is normalization-only (not a truncation). Reuses PR-D's
center_edge_pairs; a_rcut filters the participating edges.
"""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
)

import array_api_compat

if TYPE_CHECKING:
    from deepmd.dpmodel.array_api import Array

import dataclasses

from deepmd.dpmodel.utils.safe_gradient import (
    safe_for_vector_norm,
)

from .graph import (
    GraphLayout,
    NeighborGraph,
    pad_and_guard_angles,
)
from .pairs import (
    center_edge_pairs,
)
from .segment import (
    segment_sum,
)


def build_angle_index(
    edge_index: Array,
    edge_vec: Array,
    edge_mask: Array,
    n_total: int,
    a_rcut: float,
    *,
    ordered: bool = False,
    include_self: bool = False,
    layout: GraphLayout | None = None,
) -> tuple[Array, Array]:
    """Build angle index for 3-body terms.

    Parameters
    ----------
    edge_index : Array
        Shape (2, E) [src, dst] SoA edge indices.
    edge_vec : Array
        Shape (E, 3) edge vectors (neighbor - center).
    edge_mask : Array
        Shape (E,) boolean validity mask for edges.
    n_total : int
        Total number of nodes.
    a_rcut : float
        Angle cutoff. Only edges with norm < a_rcut participate in angles.
    ordered : bool, optional
        If True, include both (a, b) and (b, a) pairs (ordered pairs).
    include_self : bool, optional
        If True, include self-angle pairs (a, a).
    layout : GraphLayout or None, optional
        If provided, uses layout.angle_capacity as static padding capacity.

    Returns
    -------
    angle_index : Array
        Shape (2, A) index pairs into the edge list.
    angle_mask : Array
        Shape (A,) boolean mask for valid angles.
    """
    xp = array_api_compat.array_namespace(edge_index)
    # a_rcut edge gate: only edges within a_rcut may participate in an angle.
    # Strict `<` (not the edge channel's `<= rcut`, builder.py:284) is
    # intentional: it mirrors dpa3's dense angle gate exactly
    # (`a_dist_mask = safe_for_vector_norm(diff, axis=-1) < self.a_rcut`,
    # repflows.py:598), so an edge sitting exactly at a_rcut is excluded from
    # angles the same way dense excludes it, even though it would still be kept
    # as an edge. `safe_for_vector_norm` (not plain `vector_norm`) matches dpa3
    # value-for-value and keeps the gate off the NaN-gradient path shared with
    # the normalization below.
    dist = safe_for_vector_norm(edge_vec, axis=-1)  # (E,)
    a_edge_mask = xp.astype(edge_mask, xp.bool) & (dist < a_rcut)
    # compact eager form only (static_nnei not exposed until angle export is
    # needed, PR-G). dst = edge_index[1, :] per the [src, dst] SoA convention.
    q_e, k_e, pair_mask = center_edge_pairs(
        edge_index[1, :],
        a_edge_mask,
        n_total,
        include_self=include_self,
        ordered=ordered,
    )
    # compact form returns all-True pair_mask, but NEVER discard it: the
    # shape-static form keeps filtered pairs and invalidates them only here.
    angle_index = xp.stack([q_e, k_e], axis=0)  # (2, A_real)
    cap = layout.angle_capacity if layout is not None else None
    ai, am = pad_and_guard_angles(angle_index, cap, min_angles=2)
    # fold pair_mask into the real-angle prefix of the padded mask
    pm_padded = xp.concat(
        [
            pair_mask,
            xp.zeros(
                (am.shape[0] - pair_mask.shape[0],),
                dtype=xp.bool,
                device=array_api_compat.device(pair_mask),
            ),
        ],
        axis=0,
    )
    return ai, am & pm_padded


def attach_angles(
    graph: NeighborGraph,
    a_rcut: float,
    *,
    ordered: bool = False,
    include_self: bool = False,
    layout: GraphLayout | None = None,
) -> NeighborGraph:
    """Attach angle_index/angle_mask to an existing edge-only NeighborGraph.

    Parameters
    ----------
    graph : NeighborGraph
        Input graph (edge fields must be populated).
    a_rcut : float
        Angle cutoff radius. Only edges with norm < a_rcut participate.
    ordered : bool, optional
        If True, include both (a, b) and (b, a) angle pairs.
    include_self : bool, optional
        If True, include self-angle pairs (a, a).
    layout : GraphLayout or None, optional
        If provided, uses layout.angle_capacity and layout.node_capacity.

    Returns
    -------
    NeighborGraph
        A new NeighborGraph with angle_index and angle_mask populated;
        all edge/node fields are unchanged.
    """
    xp = array_api_compat.array_namespace(graph.edge_index)
    if layout is not None and layout.node_capacity is not None:
        n_total = layout.node_capacity
    else:
        n_total = int(xp.sum(graph.n_node))
    ai, am = build_angle_index(
        graph.edge_index,
        graph.edge_vec,
        graph.edge_mask,
        n_total,
        a_rcut,
        ordered=ordered,
        include_self=include_self,
        layout=layout,
    )
    return dataclasses.replace(graph, angle_index=ai, angle_mask=am)


def graph_angle_cos(angle_index: Array, edge_vec: Array, eps: float = 1e-6) -> Array:
    """Per-angle cosine, mirroring dpa3 ``cosine_ij`` (repflows.py:632-644).

    Parameters
    ----------
    angle_index : Array
        Shape (2, A) index pairs into edge list. ``angle_index[0, a]`` is
        edge_a and ``angle_index[1, a]`` is edge_b for angle ``a``.
    edge_vec : Array
        Shape (E, 3) edge vectors (r_src - r_dst, i.e. neighbor - center).
    eps : float, optional
        Numerical stabiliser: norm denominators use ``||v|| + eps`` and the
        dot product is scaled by ``(1 - eps)``.  Mirrors the dpa3 dense
        channel exactly (repflows.py:643-649).

    Returns
    -------
    Array
        Shape (A,) cosine values, one per angle slot (valid and padding).
        Padding slots carry arbitrary values; mask with angle_mask before use.
    """
    xp = array_api_compat.array_namespace(edge_vec)
    va = xp.take(edge_vec, angle_index[0, :], axis=0)  # (A, 3)
    vb = xp.take(edge_vec, angle_index[1, :], axis=0)  # (A, 3)
    # safe_for_vector_norm (not plain vector_norm) mirrors dpa3 exactly
    # (repflows.py:642-643) and gives a 0 gradient at ||v||==0 instead of NaN;
    # edge_vec is the sole autograd leaf, so this removes a latent NaN-gradient
    # landmine on the geometry path (values are identical for real, non-zero
    # edges, so fp64 dense/se_t parity is unchanged).
    na = va / (safe_for_vector_norm(va, axis=-1, keepdims=True) + eps)
    nb = vb / (safe_for_vector_norm(vb, axis=-1, keepdims=True) + eps)
    return xp.sum(na * nb, axis=-1) * (1.0 - eps)


def angle_to_edge_sum(data: Array, angle_index: Array, num_edges: int) -> Array:
    """Aggregate per-angle data to the angle's query edge (edge_a).

    Unlike ``edge_force_virial``, this does NOT take an ``angle_mask`` and
    does not zero padding internally: guard angles point at edge
    ``pad_value`` (an in-range real edge, e.g. edge 0), so their ``data``
    lands on that edge unless already zeroed. Callers MUST zero ``data`` at
    padded angle slots (``data * angle_mask``) before calling this.

    Parameters
    ----------
    data : Array
        Shape (A,) or (A, ...) per-angle data to aggregate. Must already be
        zero at padded (``angle_mask == False``) slots.
    angle_index : Array
        Shape (2, A) angle index pairs into edges.
    num_edges : int
        Total number of edges (E).

    Returns
    -------
    Array
        Shape (E,) or (E, ...) aggregated per-edge data.
    """
    return segment_sum(data, angle_index[0, :], num_edges)


def angle_to_node_sum(
    data: Array, angle_index: Array, edge_index: Array, num_nodes: int
) -> Array:
    """Aggregate per-angle data to the shared center (dst of edge_a).

    Unlike ``edge_force_virial``, this does NOT take an ``angle_mask`` and
    does not zero padding internally: guard angles point at node
    ``edge_index[1, pad_value]`` (an in-range real node), so their ``data``
    lands on that node unless already zeroed. Callers MUST zero ``data`` at
    padded angle slots (``data * angle_mask``) before calling this.

    Parameters
    ----------
    data : Array
        Shape (A,) or (A, ...) per-angle data to aggregate. Must already be
        zero at padded (``angle_mask == False``) slots.
    angle_index : Array
        Shape (2, A) angle index pairs into edges.
    edge_index : Array
        Shape (2, E) edge indices [src, dst].
    num_nodes : int
        Total number of nodes (N).

    Returns
    -------
    Array
        Shape (N,) or (N, ...) aggregated per-node data.
    """
    xp = array_api_compat.array_namespace(data)
    center = xp.take(edge_index[1, :], angle_index[0, :], axis=0)
    return segment_sum(data, center, num_nodes)


def angle_padding_fraction(graph: NeighborGraph) -> float:
    """Return the fraction of angle slots that are padding (guard entries).

    Parameters
    ----------
    graph : NeighborGraph
        A graph with ``angle_mask`` set (i.e., after :func:`attach_angles`
        with a static ``GraphLayout.angle_capacity``).

    Returns
    -------
    float
        ``1 - A_real / A_max`` where ``A_real`` is the count of valid angles
        and ``A_max`` is ``angle_mask.shape[0]``.  Returns ``0.0`` when the
        mask is empty.
    """
    if graph.angle_mask is None:
        return 0.0
    xp = array_api_compat.array_namespace(graph.angle_mask)
    total = graph.angle_mask.shape[0]
    if total == 0:
        return 0.0
    real = int(xp.sum(xp.astype(graph.angle_mask, xp.int64)))
    return 1.0 - real / total
