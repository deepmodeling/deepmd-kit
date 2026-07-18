# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-route regression tests for the dpmodel DPA4 descriptor."""

import dataclasses

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)

from .test_dpa4_sparse_edges import (
    build_neighbor_list_np,  # noqa: F401  (re-exported for Task 10's fold-in)
    build_sparse_edges_from_nlist,
    make_descriptor,
    make_inputs,
)


def make_graph_from_nlist(coord, nlist):
    """Build a ghost-free NeighborGraph from a gas-phase local nlist."""
    nf, nloc, _ = nlist.shape
    edge_index, edge_vec = build_sparse_edges_from_nlist(coord, nlist)
    return NeighborGraph(
        n_node=np.full((nf,), nloc, dtype=np.int64),
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=np.ones(edge_index.shape[1], dtype=bool),
    )


def _run_graph(dd, coord, atype, nlist, permute_seed=None):
    nf, nloc = atype.shape
    graph = make_graph_from_nlist(coord, nlist)
    if permute_seed is not None:
        perm = np.random.default_rng(permute_seed).permutation(
            graph.edge_index.shape[1]
        )
        graph = dataclasses.replace(
            graph,
            edge_index=graph.edge_index[:, perm],
            edge_vec=graph.edge_vec[perm],
            edge_mask=graph.edge_mask[perm],
        )
    out, rot_mat = dd.call_graph(graph, atype.reshape(-1))
    assert rot_mat is None
    return np.asarray(out)


def _jitter_zero_arrays(node, rng: np.random.Generator) -> None:
    """Recursively replace exactly-zero float arrays with small noise.

    DPA4 deliberately zero-initializes several residual output projections
    (``SO2Convolution.post_focus_mix``, ``EquivariantFFN.so3_linear_2`` --
    both per-block and the top-level ``output_ffn`` -- see the "Zero-
    initialized so residual path starts near-identity" comments in
    ``dpa4_nn/so2.py``/``dpa4_nn/ffn.py``) so a freshly constructed,
    untrained descriptor is architecturally edge/message independent: its
    scalar read-out is exactly the type embedding regardless of geometry,
    neighbors, or ``exclude_types``. That makes a bare ``make_descriptor()``
    vacuous for an exclusion anti-vacuity check -- excluding pairs cannot
    change an output that never depended on edges. This jitters those (and
    only those -- non-zero arrays such as the learned type embedding are
    untouched) in a serialized parameter tree in place, in a fixed
    depth-first order, so two calls seeded identically stay bit-identical.
    """
    if isinstance(node, dict):
        for value in node.values():
            _jitter_zero_arrays(value, rng)
    elif isinstance(node, list):
        for value in node:
            _jitter_zero_arrays(value, rng)
    elif isinstance(node, np.ndarray):
        if node.dtype.kind == "f" and node.size > 0 and np.all(node == 0.0):
            node[...] = rng.normal(0.0, 0.05, size=node.shape)


def make_message_sensitive_descriptor(seed: int = 99) -> DescrptDPA4:
    """A ``make_descriptor()`` variant with its zero-init residuals jittered.

    Two calls with the same ``seed`` are bit-identical (deserialize is
    deterministic given the jittered parameter tree), so a pair of
    independently constructed descriptors used to isolate the effect of
    ``exclude_types`` still share every other weight.
    """
    data = make_descriptor().serialize()
    _jitter_zero_arrays(data, np.random.default_rng(seed))
    return DescrptDPA4.deserialize(data)


def test_call_graph_matches_dense() -> None:
    # Same physical edges (non-binding sel) => same descriptor within fp64
    # scatter-reassociation tolerance; output is flat (N, C).
    dd = make_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    out_dense = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
    out_graph = _run_graph(dd, coord, atype, nlist)
    assert out_graph.shape == (nf * nloc, dd.get_dim_out())
    np.testing.assert_allclose(
        out_graph.reshape(nf, nloc, -1), out_dense, rtol=1e-10, atol=1e-12
    )


def test_call_graph_matches_dense_permuted_edges() -> None:
    # Arbitrary edge order (the graph contract) must not change the result.
    dd = make_descriptor()
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    out_dense = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
    out_graph = _run_graph(dd, coord, atype, nlist, permute_seed=31)
    np.testing.assert_allclose(
        out_graph.reshape(nf, nloc, -1), out_dense, rtol=1e-10, atol=1e-12
    )


def test_call_graph_exclude_types_matches_dense() -> None:
    # Pair exclusion: canonical apply_pair_exclusion on the graph must equal
    # the dense build_type_exclude_mask route. Also pins the empty-exclusion
    # branch via the tests above.
    #
    # NOTE: uses make_message_sensitive_descriptor(), not the plain
    # make_descriptor() fixture, so the anti-vacuity assertion below is
    # meaningful -- see that helper's docstring. A bare make_descriptor()
    # is architecturally edge-independent (multiple zero-init residual
    # output projections), so exclude_types provably cannot change its
    # output; asserting non-vacuity against it would always fail regardless
    # of whether exclusion is correctly wired.
    dd = make_message_sensitive_descriptor()
    dd.reinit_exclude([(0, 1)])
    coord, atype, nlist = make_inputs()
    nf, nloc = atype.shape
    out_dense = np.asarray(dd.call(coord.reshape(nf, -1), atype, nlist)[0])
    out_graph = _run_graph(dd, coord, atype, nlist)
    np.testing.assert_allclose(
        out_graph.reshape(nf, nloc, -1), out_dense, rtol=1e-10, atol=1e-12
    )
    # anti-vacuity: exclusion must actually change the descriptor
    dd2 = make_message_sensitive_descriptor()
    out_noexcl = np.asarray(dd2.call(coord.reshape(nf, -1), atype, nlist)[0])
    assert not np.allclose(out_dense, out_noexcl, rtol=1e-6, atol=1e-8)


def test_call_graph_comm_dict_raises() -> None:
    dd = make_descriptor()
    coord, atype, nlist = make_inputs()
    graph = make_graph_from_nlist(coord, nlist)
    with pytest.raises(NotImplementedError, match="comm_dict"):
        dd.call_graph(graph, atype.reshape(-1), comm_dict={"dummy": None})


def test_capability_flags() -> None:
    dd = make_descriptor()
    assert dd.uses_graph_lower() is True
    assert dd.uses_compact_edge_pairs() is False
    assert dd.graph_type_embedding_table() is None
    dd.disable_graph_lower()
    assert dd.uses_graph_lower() is False


def test_uses_graph_lower_feature_gates() -> None:
    # Conditioning inputs that ride only the dense call signature must gate
    # the graph route off. Exercise each gate attribute directly (the
    # constructor kwargs for spin/charge-spin/bridging are heavyweight).
    for attr in ("charge_spin_embedding", "spin_embedding", "bridging_switch"):
        dd = make_descriptor()
        assert dd.uses_graph_lower() is True
        setattr(dd, attr, object())  # any non-None sentinel
        assert dd.uses_graph_lower() is False, attr
