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
    field,
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
    src/dst index vectors are contiguous. Destination/source CSR views address
    the current payload through permutations. Builders may preserve incoming
    order or apply a stable destination-major canonicalization. Consumers must
    apply ``edge_mask`` even inside a CSR row.
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
    destination_order: Array | None = field(default=None, kw_only=True)
    """(E,) edge permutation grouped by destination; same dtype as edge_index."""
    destination_row_ptr: Array | None = field(default=None, kw_only=True)
    """(N + 1,) int64 offsets into ``destination_order``."""
    source_row_ptr: Array | None = field(default=None, kw_only=True)
    """(N + 1,) int64 CSR offsets into ``source_order``."""
    source_order: Array | None = field(default=None, kw_only=True)
    """(E,) source-grouped edge permutation; same dtype as edge_index."""
    destination_sorted: bool = field(default=False, kw_only=True)
    """Whether the payload is destination-major and destination_order is identity."""


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


def node_ownership_mask(n_node: Array, n_local: Array, n_total: int) -> Array:
    """Return the owned-node mask for a local-plus-halo graph.

    Each frame occupies one contiguous block of ``n_node[f]`` nodes, with its
    ``n_local[f]`` owned nodes first and halo nodes after them.

    Parameters
    ----------
    n_node
        Total node counts per frame with shape ``(nf,)``.
    n_local
        Owned node counts per frame with shape ``(nf,)``.
    n_total
        Size of the flat node axis.

    Returns
    -------
    Array
        Boolean ownership mask with shape ``(n_total,)``.
    """
    xp = array_api_compat.array_namespace(n_node, n_local)
    device = array_api_compat.device(n_node)
    node_index = xp.arange(n_total, dtype=n_node.dtype, device=device)
    frame_id = frame_id_from_n_node(n_node, n_total=n_total)
    frame_end = xp.cumulative_sum(n_node)
    frame_start = frame_end - n_node
    index_in_frame = node_index - xp.take(frame_start, frame_id, axis=0)
    local_count = xp.take(n_local, frame_id, axis=0)
    return index_in_frame < local_count


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
