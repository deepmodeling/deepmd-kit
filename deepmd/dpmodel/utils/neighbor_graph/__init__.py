# SPDX-License-Identifier: LGPL-3.0-or-later
"""NeighborGraph: backend-agnostic edge-graph neighbor-list subsystem.

The unified edge/graph neighbor-list contract and its supporting machinery:
``graph`` (the ``NeighborGraph``/``GraphLayout`` contract + derived node-validity
+ edge padding), ``builder`` (reuse the tested dense nlist -> ``NeighborGraph``),
``segment`` (mask-aware segment-reduction toolkit), and ``derivatives`` (edge
force/virial assembly). See memory/spec_unified_edge_nlist.md.
"""

from .builder import (
    build_neighbor_graph,
    neighbor_graph_from_extended,
)
from .derivatives import (
    edge_force_virial,
)
from .graph import (
    GraphLayout,
    NeighborGraph,
    node_validity_mask,
    pad_and_guard_edges,
)
from .segment import (
    segment_mean,
    segment_sum,
)

__all__ = [
    "GraphLayout",
    "NeighborGraph",
    "build_neighbor_graph",
    "edge_force_virial",
    "neighbor_graph_from_extended",
    "node_validity_mask",
    "pad_and_guard_edges",
    "segment_mean",
    "segment_sum",
]
