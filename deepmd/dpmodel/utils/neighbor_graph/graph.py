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


def frame_id_from_n_node(n_node: Array) -> Array:
    """Node->frame map for a flat node axis: ``repeat(arange(nf), n_node)``.

    Implemented via ``searchsorted(cumulative_sum(n_node), arange(N), side="right")``
    -- the same primitives used in ``edge_force_virial`` for per-frame virial.

    Parameters
    ----------
    n_node
        Per-frame node counts.  Shape ``(nf,)``.

    Returns
    -------
    frame_id
        Frame index of each flat node, compact-prefix frame-major.
        Shape ``(N,)`` int64, where ``N = sum(n_node)``.
    """
    xp = array_api_compat.array_namespace(n_node)
    dev = array_api_compat.device(n_node)
    n_total = int(xp.sum(n_node))
    idx = xp.arange(n_total, dtype=n_node.dtype, device=dev)
    boundaries = xp.cumulative_sum(n_node)  # (nf,) upper bounds, exclusive
    return xp.astype(xp.searchsorted(boundaries, idx, side="right"), xp.int64)


def node_validity_mask(n_node: Array, n_total: int) -> Array:
    """Derive the (n_total,) real-vs-padding node mask from per-frame counts.

    Compact-prefix layout: the first ``sum(n_node)`` nodes are real, the rest
    are padding. jit-safe (no Python ``int`` cast on the traced sum).
    """
    xp = array_api_compat.array_namespace(n_node)
    idx = xp.arange(n_total, dtype=n_node.dtype, device=array_api_compat.device(n_node))
    return idx < xp.sum(n_node)
