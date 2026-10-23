# SPDX-License-Identifier: LGPL-3.0-or-later
"""NeighborGraph: backend-agnostic edge-graph neighbor-list subsystem.

The unified edge/graph neighbor-list contract and its supporting machinery:
``graph`` (the ``NeighborGraph``/``GraphLayout`` contract + derived node-validity
+ edge padding), ``csr`` (backend-agnostic CSR construction and canonicalization),
``builder`` (the carry-all ``build_neighbor_graph`` dispatcher + the
``from_dense_quartet`` legacy converter), ``pair_graph`` (the complete
intra-frame pair topology of interactions without a cutoff), ``segment``
(mask-aware segment-reduction toolkit), and ``derivatives`` (edge and pair
force/virial assembly).
See the design discussion wanghan-iapcm/deepmd-kit#4.
"""

from .angles import (
    angle_padding_fraction,
    angle_to_edge_sum,
    angle_to_node_sum,
    attach_angles,
    build_angle_index,
    graph_angle_cos,
)
from .ase_builder import (
    build_neighbor_graph_ase,
)
from .builder import (
    build_neighbor_graph,
    from_dense_quartet,
    graph_from_dense_quartet,
)
from .csr import (
    attach_edge_csr,
    build_edge_csr,
    canonicalize_neighbor_graph,
)
from .derivatives import (
    edge_force_virial,
    pair_force_virial,
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
    append_isolated_frames,
    apply_pair_exclusion,
    compact_nodes,
    expand_node_values,
    frame_id_from_n_node,
    node_ownership_mask,
    node_validity_mask,
    pad_and_guard_angles,
    pad_and_guard_edges,
)
from .pair_graph import (
    ALL_PAIRS,
    K_GRID,
    PME_MESH,
    KGrid,
    MeshGrid,
    PairGraph,
    build_fixed_k_grid,
    build_k_grid,
    build_mesh_grid,
    build_pair_graph,
    build_topologies,
    check_complete_frames,
    enumerate_k_grid_bounds,
    flatten_topologies,
    k_grid_bounds_of_cells,
    topology_flat_fields,
    unflatten_topologies,
    unordered_pair_index,
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
    "ALL_PAIRS",
    "K_GRID",
    "PME_MESH",
    "GraphLayout",
    "KGrid",
    "MeshGrid",
    "NeighborGraph",
    "PairGraph",
    "angle_padding_fraction",
    "angle_to_edge_sum",
    "angle_to_node_sum",
    "append_isolated_frames",
    "apply_pair_exclusion",
    "attach_angles",
    "attach_edge_csr",
    "build_angle_index",
    "build_edge_csr",
    "build_fixed_k_grid",
    "build_k_grid",
    "build_mesh_grid",
    "build_neighbor_graph",
    "build_neighbor_graph_ase",
    "build_pair_graph",
    "build_topologies",
    "canonicalize_neighbor_graph",
    "center_edge_pairs",
    "check_complete_frames",
    "compact_nodes",
    "edge_env_mat",
    "edge_force_virial",
    "enumerate_k_grid_bounds",
    "expand_node_values",
    "flatten_topologies",
    "frame_id_from_n_node",
    "from_dense_quartet",
    "graph_angle_cos",
    "graph_from_dense_quartet",
    "k_grid_bounds_of_cells",
    "neighbor_graph_from_ijs",
    "node_ownership_mask",
    "node_validity_mask",
    "pad_and_guard_angles",
    "pad_and_guard_edges",
    "pair_force_virial",
    "segment_max",
    "segment_mean",
    "segment_softmax",
    "segment_sum",
    "topology_flat_fields",
    "unflatten_topologies",
    "unordered_pair_index",
]
