# SPDX-License-Identifier: LGPL-3.0-or-later
"""NeighborGraph: backend-agnostic edge-graph neighbor-list subsystem.

The unified edge/graph neighbor-list contract and its supporting machinery:
``graph`` (the ``NeighborGraph``/``GraphLayout`` contract + derived node-validity
+ edge padding), ``builder`` (the carry-all ``build_neighbor_graph`` dispatcher +
the ``from_dense_quartet`` legacy converter), ``segment`` (mask-aware
segment-reduction toolkit), and ``derivatives`` (edge force/virial assembly).
See the design discussion wanghan-iapcm/deepmd-kit#4.
"""

from .ase_builder import (
    build_neighbor_graph_ase,
)
from .builder import (
    build_neighbor_graph,
    from_dense_quartet,
    graph_from_dense_quartet,
)
from .derivatives import (
    edge_force_virial,
)
from .env import (
    edge_env_mat,
)
from .from_ijs import (
    neighbor_graph_from_ijs,
)
from .graph import (
    GraphLayout,
    NeighborGraph,
    apply_pair_exclusion,
    frame_id_from_n_node,
    node_validity_mask,
    pad_and_guard_edges,
)
from .pairs import (
    center_edge_pairs,
)
from .segment import (
    segment_max,
    segment_mean,
    segment_softmax,
    segment_sum,
)

__all__ = [
    "GraphLayout",
    "NeighborGraph",
    "apply_pair_exclusion",
    "build_neighbor_graph",
    "build_neighbor_graph_ase",
    "center_edge_pairs",
    "edge_env_mat",
    "edge_force_virial",
    "frame_id_from_n_node",
    "from_dense_quartet",
    "graph_from_dense_quartet",
    "neighbor_graph_from_ijs",
    "node_validity_mask",
    "pad_and_guard_edges",
    "segment_max",
    "segment_mean",
    "segment_softmax",
    "segment_sum",
]
