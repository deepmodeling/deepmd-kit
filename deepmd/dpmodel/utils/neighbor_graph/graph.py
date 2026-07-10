# SPDX-License-Identifier: LGPL-3.0-or-later
"""Backend-agnostic edge-graph neighbor-list contract (NeighborGraph) and its
length policy (GraphLayout). See the design discussion wanghan-iapcm/deepmd-kit#4.

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
    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
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
    Dummy edges point at node ``pad_value`` (in-range) with zero ``edge_vec``.

    Parameters
    ----------
    edge_index
        (2, E_real) ``[src, dst]`` node endpoints of the real edges.
    edge_vec
        (E_real, 3) per-edge displacement of the real edges.
    capacity
        Target edge-axis length ``E_max``. ``None`` (torch dynamic) appends
        exactly ``min_edges`` masked dummy edges so the axis has a known lower
        bound and shape-stable guards for export; an int (jax static) pads to
        ``E_max = capacity`` and raises ``ValueError`` on overflow.
    min_edges
        Number of dummy edges appended when ``capacity is None``.
    pad_value
        Node index the dummy edges point at (must be in range).

    Returns
    -------
    edge_index
        (2, target) padded edge endpoints.
    edge_vec
        (target, 3) padded edge displacements (dummy rows zero).
    edge_mask
        (target,) boolean mask, ``True`` for the real-edge prefix.
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


def pad_and_guard_angles(
    angle_index: Array,
    angle_capacity: int | None = None,
    min_angles: int = 2,
    pad_value: int = 0,
) -> tuple[Array, Array]:
    """Append padding/guard angles as a contiguous suffix and build angle_mask.

    Real angles (``angle_index``) stay at the front (compact layout).
    Dummy angles point at edge ``pad_value`` (in-range).

    Parameters
    ----------
    angle_index
        (2, A_real) ``[edge_a, edge_b]`` edge endpoints of the real angles.
    angle_capacity
        Target angle-axis length ``A_max``. ``None`` (torch dynamic) appends
        exactly ``min_angles`` masked dummy angles so the axis has a known lower
        bound and shape-stable guards for export; an int (jax static) pads to
        ``A_max = angle_capacity`` and raises ``ValueError`` on overflow.
    min_angles
        Number of dummy angles appended when ``angle_capacity is None``.
    pad_value
        Edge index the dummy angles point at (must be in range).

    Returns
    -------
    angle_index
        (2, target) padded angle endpoints.
    angle_mask
        (target,) boolean mask, ``True`` for the real-angle prefix.
    """
    xp = array_api_compat.array_namespace(angle_index)
    dev = array_api_compat.device(angle_index)
    a_real = angle_index.shape[1]
    if angle_capacity is None:
        target = a_real + min_angles
    else:
        if a_real > angle_capacity:
            raise ValueError(
                f"angle overflow: {a_real} real angles > angle_capacity {angle_capacity}"
            )
        target = angle_capacity
    n_pad = target - a_real
    pad_idx = xp.full((2, n_pad), pad_value, dtype=angle_index.dtype, device=dev)
    ai = xp.concat([angle_index, pad_idx], axis=1)
    arange = xp.arange(target, dtype=angle_index.dtype, device=dev)
    angle_mask = arange < a_real
    return ai, angle_mask


def frame_id_from_n_node(n_node: Array, n_total: int | None = None) -> Array:
    """Node->frame map for a flat node axis: ``repeat(arange(nf), n_node)``.

    Implemented via ``searchsorted(cumulative_sum(n_node), arange(N), side="right")``
    -- the same primitives used in ``edge_force_virial`` for per-frame virial.

    Parameters
    ----------
    n_node
        Per-frame node counts.  Shape ``(nf,)``.
    n_total
        Size of the (possibly padded) flat node axis ``N``.  ``None`` (the
        numpy/eager default) falls back to ``int(sum(n_node))``; pass a STATIC
        value to keep the function trace-friendly under jax.jit / export, where
        ``int()`` on the traced sum is not allowed (mirrors
        :func:`node_validity_mask`).  Padding nodes ``[sum(n_node), n_total)``
        are CLAMPED to the last frame (``nf - 1``) so a downstream
        ``segment_sum(..., num_segments=nf)`` stays in range; they carry no real
        edge, so this assignment is unused downstream.

    Returns
    -------
    frame_id
        Frame index of each flat node, compact-prefix frame-major.
        Shape ``(n_total,)`` int64 (``n_total = sum(n_node)`` when not padded).
    """
    xp = array_api_compat.array_namespace(n_node)
    dev = array_api_compat.device(n_node)
    if n_total is None:
        n_total = int(xp.sum(n_node))
    idx = xp.arange(n_total, dtype=n_node.dtype, device=dev)
    boundaries = xp.cumulative_sum(n_node)  # (nf,) upper bounds, exclusive
    frame_id = xp.astype(xp.searchsorted(boundaries, idx, side="right"), xp.int64)
    # padding nodes (idx >= sum(n_node)) land at frame ``nf`` (OOB); clamp them to
    # the last real frame so the per-frame scatter never indexes out of range.
    # Derive ``nf - 1`` as a RUNTIME 0-d tensor (sum of ones over the frame axis)
    # rather than ``xp.asarray(n_node.shape[0] - 1)``: under symbolic make_fx /
    # torch.export, ``shape[0]`` is a SymInt and materializing it into a constant
    # tensor SPECIALIZES the frame axis -- baking the trace-time frame count into
    # every downstream per-frame reduction and breaking dynamic-``nf`` inference.
    last_frame = xp.sum(xp.ones_like(n_node)) - 1  # 0-d int == nf - 1
    return xp.minimum(frame_id, xp.astype(last_frame, xp.int64))


def apply_pair_exclusion(
    graph: NeighborGraph,
    atype: Array,
    pair_excl: PairExcludeMask | None,
    *,
    compact: bool = False,
) -> NeighborGraph:
    """Canonical pair-type exclusion transform (decision #18).

    ANDs the per-edge type keep-mask into ``graph.edge_mask`` so excluded
    type pairs contribute exactly zero to every downstream ``segment_sum``.
    The search stays purely geometric; this transform is applied ONCE at the
    atomic-model seam (model-level ``pair_exclude_types``) and, for
    descriptor-level ``exclude_types``, inside the descriptor's graph
    forward. Identity (returns ``graph`` itself) when ``pair_excl`` is
    ``None`` or empty.

    Parameters
    ----------
    graph
        The neighbor graph; only ``edge_mask`` (and, if ``compact=True``,
        ``edge_index``, ``edge_vec``, ``angle_index``, ``angle_mask``) are
        replaced.
    atype
        (N,) flat node types, clamped >= 0 (virtual atoms already handled
        by the caller / the builders).
    pair_excl
        The ``PairExcludeMask`` holding the excluded (ti, tj) set.
    compact
        If ``False`` (default), only zero-out masked edges via ``edge_mask``
        (shape-static; the ONLY mode allowed in compiled / AOTI paths).
        If ``True``, additionally drop masked edges so the returned graph
        has no padding on the edge axis (data-dependent shape; eager /
        dynamic-nedge only). When the graph carries angle fields, angles are
        remapped onto the compacted edge axis and any angle whose constituent
        edges were excluded is dropped. ``angle_index`` and ``angle_mask`` must
        be a consistent pair (both set or both ``None``; matching ``A``);
        anything else raises ``ValueError``.

    Returns
    -------
    NeighborGraph
        A ``dataclasses.replace`` copy (or the original ``graph`` on early
        exit) with the exclusion applied.

    Notes
    -----
    The C++ inference-path mirror is ``applyPairExclusion`` in
    ``source/api_cc/include/commonPT.h``. It uses the same argument order
    (edge_index, edge_mask, atype, ...) and the same variable names
    (``type_ij``, ``keep``): it computes
    ``type_ij = atype[dst]*(ntypes+1) + atype[src]`` and ANDs the flat
    ``(ntypes+1)^2`` table lookup into ``edge_mask`` (mask-only mode; no
    compact variant on the compiled path).
    """
    import dataclasses

    if pair_excl is None or len(pair_excl.get_exclude_types()) == 0:
        return graph
    xp = array_api_compat.array_namespace(graph.edge_mask)
    keep = pair_excl.build_edge_exclude_mask(graph.edge_index, atype)
    out = dataclasses.replace(
        graph,
        edge_mask=xp.logical_and(graph.edge_mask, xp.astype(keep, xp.bool)),
    )
    if compact:
        # Angle fields are a coupled pair (produced together by the angle
        # builder): both present or both None. Fail fast on any inconsistent
        # state — a partial or shape-mismatched pair is a caller bug that would
        # otherwise remap silently wrong.
        has_ai = out.angle_index is not None
        has_am = out.angle_mask is not None
        if has_ai != has_am:
            raise ValueError(
                "apply_pair_exclusion(compact=True): angle_index and angle_mask "
                "must both be set or both be None; got "
                f"angle_index={'set' if has_ai else 'None'}, "
                f"angle_mask={'set' if has_am else 'None'}."
            )
        if has_ai:
            if out.angle_index.ndim != 2 or out.angle_index.shape[0] != 2:
                raise ValueError(
                    "apply_pair_exclusion(compact=True): angle_index must have "
                    f"shape (2, A); got {tuple(out.angle_index.shape)}."
                )
            if out.angle_index.shape[1] != out.angle_mask.shape[0]:
                raise ValueError(
                    "apply_pair_exclusion(compact=True): angle_index (2, A) and "
                    f"angle_mask (A,) disagree on A: {out.angle_index.shape[1]} "
                    f"vs {out.angle_mask.shape[0]}."
                )
        (keep_idx,) = xp.nonzero(out.edge_mask)
        fields = {
            "edge_index": out.edge_index[:, keep_idx],
            "edge_vec": xp.take(out.edge_vec, keep_idx, axis=0),
            "edge_mask": xp.take(out.edge_mask, keep_idx, axis=0),
        }
        if has_ai:
            # Angles reference PRE-compaction edge positions; remap them to the
            # compacted axis and drop any angle whose constituent edges were
            # excluded. ``new_pos`` maps old edge position -> new position via an
            # exclusive prefix sum over the survivors (-1 for dropped edges).
            surv = xp.astype(out.edge_mask, out.edge_index.dtype)  # (E,) 0/1
            rank = xp.cumulative_sum(surv, axis=0) - surv  # survivors before me
            new_pos = xp.where(out.edge_mask, rank, xp.full_like(rank, -1))
            a_new = xp.take(new_pos, out.angle_index[0, :], axis=0)
            b_new = xp.take(new_pos, out.angle_index[1, :], axis=0)
            both_survive = xp.logical_and(a_new >= 0, b_new >= 0)
            angle_keep = xp.logical_and(
                xp.astype(out.angle_mask, xp.bool), both_survive
            )
            (angle_keep_idx,) = xp.nonzero(angle_keep)
            fields["angle_index"] = xp.stack(
                [
                    xp.take(a_new, angle_keep_idx, axis=0),
                    xp.take(b_new, angle_keep_idx, axis=0),
                ],
                axis=0,
            )
            fields["angle_mask"] = xp.take(out.angle_mask, angle_keep_idx, axis=0)
        out = dataclasses.replace(out, **fields)
    return out


def node_validity_mask(n_node: Array, n_total: int) -> Array:
    """Derive the (n_total,) real-vs-padding node mask from per-frame counts.

    Compact-prefix layout: the first ``sum(n_node)`` nodes are real, the rest
    are padding. jit-safe (no Python ``int`` cast on the traced sum).

    Parameters
    ----------
    n_node
        (nf,) per-frame REAL node counts.
    n_total
        Size of the (possibly padded) flat node axis ``N``.

    Returns
    -------
    mask
        (n_total,) boolean mask, ``True`` for the real-node compact prefix.
    """
    xp = array_api_compat.array_namespace(n_node)
    idx = xp.arange(n_total, dtype=n_node.dtype, device=array_api_compat.device(n_node))
    return idx < xp.sum(n_node)
