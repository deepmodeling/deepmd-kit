# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pairs of edges sharing a center (``dst``) — the edge-pair axis.

Shared primitive: graph-native attention (NeighborGraph PR-D) uses
``(ordered=True, include_self=True)`` = the full transformer neighbor-pair
square per center; 3-body angles (PR-E) use ``(ordered=False,
include_self=False)``.

Two forms:

- **compact eager** (``static_nnei=None``): segment-based enumeration over the
  real edges only — sort edge ids by center, expand each center's Cartesian
  square via cumsum offsets. Dynamic ``P = sum(deg**2)``; memory ``O(P)``
  (same order as dense attention's ``O(nloc * nnei**2)``). Uses data-dependent
  shapes (``nonzero``) so it is EAGER-ONLY.
- **shape-static** (``static_nnei`` set): assumes the center-major static
  layout (``E = n_center * static_nnei``, edge ``c * static_nnei + m`` belongs
  to center ``c`` — the layout ``from_dense_quartet(compact=False)`` emits).
  Pure arange/reshape arithmetic, ``P = n_center * static_nnei**2`` with all
  pairs materialized and validity carried by ``pair_mask`` — no data-dependent
  ops, so it stays jit/export/make_fx-traceable.

A global ``(E, E)`` same-center boolean is deliberately NOT used: with
``E ~ N * nnei`` it costs ``O(N**2 * nnei**2)`` memory.
"""

from __future__ import annotations

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
    xp_add_at,
)


def center_edge_pairs(
    dst: Array,
    edge_mask: Array,
    n_total: int,
    *,
    include_self: bool = True,
    ordered: bool = True,
    static_nnei: int | None = None,
) -> tuple[Array, Array, Array]:
    """Enumerate pairs of edges sharing a center.

    Parameters
    ----------
    dst : Array
        (E,) int64 center of each edge (``edge_index[1]``).
    edge_mask : Array
        (E,) bool, real (True) vs padding (False) edges.
    n_total : int
        Number of centers (bounds ``dst``).
    include_self : bool
        Keep the ``m == n`` diagonal (transformer self-attention needs it).
    ordered : bool
        Keep both ``(m, n)`` and ``(n, m)`` (attention: yes, ``q_m . k_n`` is
        not symmetric). ``False`` keeps only ``n >= m`` (with
        ``include_self=False``: ``n > m`` — the angle set).
    static_nnei : int | None
        ``None`` -> compact eager form. Set -> shape-static form assuming the
        center-major layout ``E = n_center * static_nnei``.

    Returns
    -------
    query_edge : Array
        (P,) int64 edge index of the query (``m``).
    key_edge : Array
        (P,) int64 edge index of the key (``n``).
    pair_mask : Array
        (P,) bool; False where either edge is padding or the pair is filtered
        by the ``include_self`` / ``ordered`` policy (shape-static form; the
        compact form drops such pairs and returns all-True).
    """
    xp = array_api_compat.array_namespace(dst)
    dev = array_api_compat.device(dst)
    if static_nnei is not None:
        return _pairs_shape_static(
            xp, dev, dst, edge_mask, static_nnei, include_self, ordered
        )
    return _pairs_compact(xp, dev, dst, edge_mask, n_total, include_self, ordered)


def _pairs_shape_static(
    xp: Any,
    dev: Any,
    dst: Array,
    edge_mask: Array,
    nn: int,
    include_self: bool,
    ordered: bool,
) -> tuple[Array, Array, Array]:
    e_tot = dst.shape[0]
    # (E, nn): every edge queries the nn slots of its own center block
    eids = xp.arange(e_tot, dtype=xp.int64, device=dev)
    base = (eids // nn) * nn  # start of each edge's center block
    slots = xp.arange(nn, dtype=xp.int64, device=dev)
    q2 = xp.broadcast_to(eids[:, None], (e_tot, nn))
    k2 = base[:, None] + slots[None, :]
    query_edge = xp.reshape(q2, (-1,))
    key_edge = xp.reshape(k2, (-1,))
    pair_mask = xp.take(edge_mask, query_edge, axis=0) & xp.take(
        edge_mask, key_edge, axis=0
    )
    if not include_self:
        pair_mask = pair_mask & (query_edge != key_edge)
    if not ordered:
        pair_mask = pair_mask & (key_edge >= query_edge)
    return query_edge, key_edge, pair_mask


def _pairs_compact(
    xp: Any,
    dev: Any,
    dst: Array,
    edge_mask: Array,
    n_total: int,
    include_self: bool,
    ordered: bool,
) -> tuple[Array, Array, Array]:
    empty = (
        xp.zeros((0,), dtype=xp.int64, device=dev),
        xp.zeros((0,), dtype=xp.int64, device=dev),
        xp.zeros((0,), dtype=xp.bool, device=dev),
    )
    if dst.shape[0] == 0:
        return empty
    # real edges only, grouped by center (stable sort keeps original order
    # within a center — irrelevant for correctness, deterministic for tests)
    (real_idx,) = xp.nonzero(edge_mask)
    r_tot = real_idx.shape[0]
    if r_tot == 0:
        return empty
    d_real = xp.take(dst, real_idx, axis=0)
    order = xp.argsort(d_real, stable=True)
    eid = xp.take(real_idx, order, axis=0)  # (R,) edge ids, center-grouped
    ds = xp.take(d_real, order, axis=0)  # (R,) sorted centers
    # per-center degree and group start (over the sorted layout)
    ones = xp.ones((r_tot,), dtype=xp.int64, device=dev)
    counts = xp_add_at(
        xp.zeros((n_total,), dtype=xp.int64, device=dev), ds, ones
    )  # (n_total,)
    csum = xp.cumulative_sum(counts)
    start = csum - counts  # (n_total,) group start per center
    deg = xp.take(counts, ds, axis=0)  # (R,) degree of each edge's center
    # each sorted edge t emits deg[t] pairs; P = sum(deg**2)
    query_sorted = xp.repeat(xp.arange(r_tot, dtype=xp.int64, device=dev), deg)  # (P,)
    # within each query's block, a 0..deg-1 ramp indexes the key group
    pair_off = xp.cumulative_sum(deg) - deg  # (R,) exclusive prefix of deg
    p_tot = query_sorted.shape[0]
    ramp = xp.arange(p_tot, dtype=xp.int64, device=dev) - xp.take(
        pair_off, query_sorted, axis=0
    )
    key_sorted = xp.take(start, xp.take(ds, query_sorted, axis=0), axis=0) + ramp
    query_edge = xp.take(eid, query_sorted, axis=0)
    key_edge = xp.take(eid, key_sorted, axis=0)
    keep = xp.ones((p_tot,), dtype=xp.bool, device=dev)
    if not include_self:
        keep = keep & (query_edge != key_edge)
    if not ordered:
        keep = keep & (key_edge >= query_edge)
    (kept,) = xp.nonzero(keep)
    query_edge = xp.take(query_edge, kept, axis=0)
    key_edge = xp.take(key_edge, kept, axis=0)
    pair_mask = xp.ones((query_edge.shape[0],), dtype=xp.bool, device=dev)
    return query_edge, key_edge, pair_mask
