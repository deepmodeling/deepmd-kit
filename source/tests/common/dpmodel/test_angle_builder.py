# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    angle_padding_fraction,
    angle_to_edge_sum,
    angle_to_node_sum,
    attach_angles,
    build_angle_index,
    build_neighbor_graph,
    edge_force_virial,
    pad_and_guard_angles,
)


def test_pad_angles_dynamic_appends_min_guard():
    ai = np.array([[0, 1], [1, 0]], dtype=np.int64)  # 2 real angles
    out_ai, out_mask = pad_and_guard_angles(ai, angle_capacity=None, min_angles=2)
    assert out_ai.shape == (2, 4)  # 2 real + 2 guard
    np.testing.assert_array_equal(out_mask, [True, True, False, False])


def test_pad_angles_static_capacity():
    ai = np.array([[0, 1], [1, 0]], dtype=np.int64)
    out_ai, out_mask = pad_and_guard_angles(ai, angle_capacity=5)
    assert out_ai.shape == (2, 5)
    assert int(out_mask.sum()) == 2


def test_pad_angles_overflow_raises():
    ai = np.zeros((2, 6), dtype=np.int64)
    with pytest.raises(ValueError):
        pad_and_guard_angles(ai, angle_capacity=4)


def _angle_oracle(dst, evnorm, mask, a_rcut, ordered, include_self):
    """All (edge_a, edge_b) sharing a center, both edges within a_rcut."""
    out = set()
    for a in range(len(dst)):
        if not mask[a] or evnorm[a] >= a_rcut:
            continue
        for b in range(len(dst)):
            if not mask[b] or evnorm[b] >= a_rcut or dst[a] != dst[b]:
                continue
            if not include_self and a == b:
                continue
            if not ordered and b < a:
                continue
            out.add((a, b))
    return out


def test_build_angle_index_matches_oracle():
    # 4 edges, all dst=0; norms [0.5, 0.9, 2.5, 0.7], a_rcut=1.0
    # DEFAULT = unordered, no-self => within a_rcut {0,1,3} give {(0,1),(0,3),(1,3)}
    edge_index = np.array([[1, 2, 3, 1], [0, 0, 0, 0]], dtype=np.int64)
    edge_vec = np.array([[0.5, 0, 0], [0.9, 0, 0], [2.5, 0, 0], [0.7, 0, 0]])
    edge_mask = np.array([True, True, True, True])
    ai, am = build_angle_index(edge_index, edge_vec, edge_mask, 4, a_rcut=1.0)
    got = {(int(ai[0, p]), int(ai[1, p])) for p in range(ai.shape[1]) if am[p]}
    evnorm = np.linalg.norm(edge_vec, axis=-1)
    assert got == _angle_oracle(
        [0, 0, 0, 0], evnorm, edge_mask, 1.0, ordered=False, include_self=False
    )
    assert got == {
        (0, 1),
        (0, 3),
        (1, 3),
    }  # unordered, no-self, edge 2 dropped (norm>a_rcut)
    assert all(
        2 not in (int(ai[0, p]), int(ai[1, p])) for p in range(ai.shape[1]) if am[p]
    )
    assert all(
        int(ai[0, p]) != int(ai[1, p]) for p in range(ai.shape[1]) if am[p]
    )  # no self


def test_build_angle_index_ordered_include_self():
    # ordered + include_self: (0,0),(0,1),(0,3),(1,0),(1,1),(1,3),(3,0),(3,1),(3,3)
    edge_index = np.array([[1, 2, 3, 1], [0, 0, 0, 0]], dtype=np.int64)
    edge_vec = np.array([[0.5, 0, 0], [0.9, 0, 0], [2.5, 0, 0], [0.7, 0, 0]])
    edge_mask = np.array([True, True, True, True])
    ai, am = build_angle_index(
        edge_index, edge_vec, edge_mask, 4, a_rcut=1.0, ordered=True, include_self=True
    )
    got = {(int(ai[0, p]), int(ai[1, p])) for p in range(ai.shape[1]) if am[p]}
    evnorm = np.linalg.norm(edge_vec, axis=-1)
    assert got == _angle_oracle(
        [0, 0, 0, 0], evnorm, edge_mask, 1.0, ordered=True, include_self=True
    )


def test_build_angle_index_masked_edge():
    # edge 1 masked out — should not appear in any angle
    edge_index = np.array([[1, 2, 3, 1], [0, 0, 0, 0]], dtype=np.int64)
    edge_vec = np.array([[0.5, 0, 0], [0.9, 0, 0], [0.3, 0, 0], [0.7, 0, 0]])
    edge_mask = np.array([True, False, True, True])
    ai, am = build_angle_index(edge_index, edge_vec, edge_mask, 4, a_rcut=1.0)
    got = {(int(ai[0, p]), int(ai[1, p])) for p in range(ai.shape[1]) if am[p]}
    evnorm = np.linalg.norm(edge_vec, axis=-1)
    assert got == _angle_oracle(
        [0, 0, 0, 0], evnorm, edge_mask, 1.0, ordered=False, include_self=False
    )
    assert all(
        1 not in (int(ai[0, p]), int(ai[1, p])) for p in range(ai.shape[1]) if am[p]
    )


def test_build_angle_index_torch_namespace():
    # Step 4b: torch-namespace smoke test (function-level import for TID253)
    import torch

    edge_index = np.array([[1, 2, 3, 1], [0, 0, 0, 0]], dtype=np.int64)
    edge_vec = np.array([[0.5, 0, 0], [0.9, 0, 0], [2.5, 0, 0], [0.7, 0, 0]])
    edge_mask = np.array([True, True, True, True])

    ai_np, am_np = build_angle_index(edge_index, edge_vec, edge_mask, 4, a_rcut=1.0)
    got_np = {
        (int(ai_np[0, p]), int(ai_np[1, p])) for p in range(ai_np.shape[1]) if am_np[p]
    }

    t_edge_index = torch.from_numpy(edge_index)
    t_edge_vec = torch.from_numpy(edge_vec)
    t_edge_mask = torch.from_numpy(edge_mask)
    ai_t, am_t = build_angle_index(t_edge_index, t_edge_vec, t_edge_mask, 4, a_rcut=1.0)
    got_t = {
        (int(ai_t[0, p].item()), int(ai_t[1, p].item()))
        for p in range(ai_t.shape[1])
        if am_t[p].item()
    }
    assert got_t == got_np


def test_build_angle_index_multi_center():
    # Edges with MIXED centers: dst=[0,1,0,1,2]; exercises the dst[a]!=dst[b] exclusion
    # src=[1,2,3,4,5], dst=[0,1,0,1,2], all norms within a_rcut
    # Expected angles per center:
    #  dst=0: edges {0,2} => {(0,2),(2,0)}
    #  dst=1: edges {1,3} => {(1,3),(3,1)}
    #  dst=2: edge {4} => no pairs
    edge_index = np.array([[1, 2, 3, 4, 5], [0, 1, 0, 1, 2]], dtype=np.int64)
    edge_vec = np.array(
        [[0.3, 0, 0], [0.4, 0, 0], [0.5, 0, 0], [0.6, 0, 0], [0.7, 0, 0]]
    )
    edge_mask = np.array([True, True, True, True, True])
    ai, am = build_angle_index(edge_index, edge_vec, edge_mask, 6, a_rcut=1.0)
    got = {(int(ai[0, p]), int(ai[1, p])) for p in range(ai.shape[1]) if am[p]}
    evnorm = np.linalg.norm(edge_vec, axis=-1)
    expected = _angle_oracle(
        [0, 1, 0, 1, 2], evnorm, edge_mask, 1.0, ordered=False, include_self=False
    )
    assert got == expected
    # Verify no cross-center angles
    assert all(
        edge_index[1, int(ai[0, p])] == edge_index[1, int(ai[1, p])]
        for p in range(ai.shape[1])
        if am[p]
    )


def test_build_angle_index_static_layout():
    # Test with static layout.angle_capacity; shape must be (2, capacity)
    edge_index = np.array([[1, 2, 3, 1], [0, 0, 0, 0]], dtype=np.int64)
    edge_vec = np.array([[0.5, 0, 0], [0.9, 0, 0], [2.5, 0, 0], [0.7, 0, 0]])
    edge_mask = np.array([True, True, True, True])
    layout = GraphLayout(edge_capacity=100, angle_capacity=10)
    ai, am = build_angle_index(
        edge_index, edge_vec, edge_mask, 4, a_rcut=1.0, layout=layout
    )
    # Check static shape
    assert ai.shape == (2, 10)
    assert am.shape == (10,)
    # Check real angles match the dynamic result
    got_static = {(int(ai[0, p]), int(ai[1, p])) for p in range(ai.shape[1]) if am[p]}
    ai_dyn, am_dyn = build_angle_index(edge_index, edge_vec, edge_mask, 4, a_rcut=1.0)
    got_dynamic = {
        (int(ai_dyn[0, p]), int(ai_dyn[1, p]))
        for p in range(ai_dyn.shape[1])
        if am_dyn[p]
    }
    assert got_static == got_dynamic
    # Check mask counts match
    assert int(am.sum()) == int(am_dyn.sum())


def test_build_angle_index_ordered_no_self():
    # Test ordered=True, include_self=False; should be symmetric pairs excluding diagonals
    edge_index = np.array([[1, 2, 3, 1], [0, 0, 0, 0]], dtype=np.int64)
    edge_vec = np.array([[0.5, 0, 0], [0.9, 0, 0], [2.5, 0, 0], [0.7, 0, 0]])
    edge_mask = np.array([True, True, True, True])
    ai, am = build_angle_index(
        edge_index, edge_vec, edge_mask, 4, a_rcut=1.0, ordered=True, include_self=False
    )
    got = {(int(ai[0, p]), int(ai[1, p])) for p in range(ai.shape[1]) if am[p]}
    evnorm = np.linalg.norm(edge_vec, axis=-1)
    expected = _angle_oracle(
        [0, 0, 0, 0], evnorm, edge_mask, 1.0, ordered=True, include_self=False
    )
    assert got == expected
    # Verify no self-angles and ordered includes both directions
    assert all(
        int(ai[0, p]) != int(ai[1, p]) for p in range(ai.shape[1]) if am[p]
    )  # no self
    for pair in got:
        a, b = pair
        # For ordered=True, include_self=False, we expect both (a,b) and (b,a)
        # if they are different and both within a_rcut and same center
        rev_pair = (b, a)
        if a != b and a not in [2] and b not in [2]:  # edge 2 is outside a_rcut
            assert rev_pair in got  # symmetric pairs should both exist


# ---------------------------------------------------------------------------
# Task 3: attach_angles tests
# ---------------------------------------------------------------------------


def test_attach_angles_sets_fields_and_preserves_edges():
    """attach_angles populates angle_index/mask; edge fields are unchanged."""
    coord = np.array([[[0.0, 0, 0], [0.8, 0, 0], [0, 0.8, 0]]])
    atype = np.array([[0, 0, 0]])  # (nf, nloc)
    ng = build_neighbor_graph(coord, atype, None, 2.0)
    # default carry-all builder leaves angles None
    assert ng.angle_index is None
    assert ng.angle_mask is None
    ng2 = attach_angles(ng, a_rcut=1.5)
    assert ng2.angle_index is not None and ng2.angle_mask is not None
    # edge fields must be identical (by value and shape)
    np.testing.assert_array_equal(np.asarray(ng2.edge_index), np.asarray(ng.edge_index))
    np.testing.assert_array_equal(np.asarray(ng2.edge_mask), np.asarray(ng.edge_mask))
    np.testing.assert_array_equal(np.asarray(ng2.edge_vec), np.asarray(ng.edge_vec))


def test_attach_angles_angle_shape_consistent():
    """angle_index has shape (2, A) and angle_mask has shape (A,)."""
    coord = np.array([[[0.0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]]])
    atype = np.array([[0, 0, 0]])  # (nf, nloc)
    ng = build_neighbor_graph(coord, atype, None, 2.0)
    ng2 = attach_angles(ng, a_rcut=1.5)
    assert ng2.angle_index.shape[0] == 2
    assert ng2.angle_mask.shape[0] == ng2.angle_index.shape[1]


def test_attach_angles_valid_angles_reference_valid_edges():
    """All valid angle pairs (q_e, k_e) must index edges that are within a_rcut."""
    coord = np.array([[[0.0, 0, 0], [0.6, 0, 0], [0, 0.6, 0]]])
    atype = np.array([[0, 0, 0]])  # (nf, nloc)
    ng = build_neighbor_graph(coord, atype, None, 2.0)
    ng2 = attach_angles(ng, a_rcut=1.0)
    ei = np.asarray(ng2.edge_index)
    ev = np.asarray(ng2.edge_vec)
    em = np.asarray(ng2.edge_mask)
    ai = np.asarray(ng2.angle_index)
    am = np.asarray(ng2.angle_mask)
    for p in range(am.shape[0]):
        if not am[p]:
            continue
        q, k = int(ai[0, p]), int(ai[1, p])
        # both referenced edges must be valid and within a_rcut
        assert em[q] and em[k]
        assert np.linalg.norm(ev[q]) < 1.0
        assert np.linalg.norm(ev[k]) < 1.0
        # both referenced edges must share the same center (dst)
        assert ei[1, q] == ei[1, k]


def test_attach_angles_with_layout():
    """Static layout.angle_capacity is respected."""
    coord = np.array([[[0.0, 0, 0], [0.6, 0, 0], [0, 0.6, 0]]])
    atype = np.array([[0, 0, 0]])  # (nf, nloc)
    ng = build_neighbor_graph(coord, atype, None, 2.0)
    layout = GraphLayout(edge_capacity=100, angle_capacity=20)
    ng2 = attach_angles(ng, a_rcut=1.5, layout=layout)
    assert ng2.angle_index.shape == (2, 20)
    assert ng2.angle_mask.shape == (20,)


def test_attach_angles_ordered_include_self():
    """ordered=True, include_self=True produces a superset of default pairs."""
    coord = np.array([[[0.0, 0, 0], [0.5, 0, 0], [0, 0.5, 0]]])
    atype = np.array([[0, 0, 0]])  # (nf, nloc)
    ng = build_neighbor_graph(coord, atype, None, 2.0)
    ng_default = attach_angles(ng, a_rcut=1.5)
    ng_full = attach_angles(ng, a_rcut=1.5, ordered=True, include_self=True)
    ai_def = np.asarray(ng_default.angle_index)
    am_def = np.asarray(ng_default.angle_mask)
    ai_full = np.asarray(ng_full.angle_index)
    am_full = np.asarray(ng_full.angle_mask)
    n_default = int(am_def.sum())
    n_full = int(am_full.sum())
    # ordered+include_self must produce at least as many angles as default
    assert n_full >= n_default


def test_attach_angles_with_layout_node_capacity():
    """layout.node_capacity branch: node_capacity is used as n_total."""
    coord = np.array([[[0.0, 0, 0], [0.6, 0, 0], [0, 0.6, 0]]])
    atype = np.array([[0, 0, 0]])  # (nf, nloc)
    ng = build_neighbor_graph(coord, atype, None, 2.0)
    layout = GraphLayout(edge_capacity=100, angle_capacity=20, node_capacity=10)
    ng2 = attach_angles(ng, a_rcut=1.5, layout=layout)
    assert ng2.angle_index is not None
    assert ng2.angle_mask.shape == (20,)
    # same real-angle set as the dynamic path (node_capacity only oversizes n_total)
    ng3 = attach_angles(ng, a_rcut=1.5)
    got = {
        (int(ng2.angle_index[0, p]), int(ng2.angle_index[1, p]))
        for p in range(ng2.angle_index.shape[1])
        if ng2.angle_mask[p]
    }
    ref = {
        (int(ng3.angle_index[0, p]), int(ng3.angle_index[1, p]))
        for p in range(ng3.angle_index.shape[1])
        if ng3.angle_mask[p]
    }
    assert got == ref


# ---------------------------------------------------------------------------
# Task 4: angle aggregation (angle_to_edge_sum / angle_to_node_sum)
# ---------------------------------------------------------------------------


def test_angle_aggregation():
    """Test angle->edge and angle->node aggregation."""
    # edges: dst=[0,0]; angles: (a=0,b=0),(a=0,b=1),(a=1,b=0)
    edge_index = np.array([[5, 5], [0, 0]], dtype=np.int64)
    angle_index = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)
    data = np.array([1.0, 2.0, 4.0])  # per-angle
    # angle->edge (group by edge_a): edge0 gets angles 0,1 => 3; edge1 gets angle2 => 4
    e = angle_to_edge_sum(data, angle_index, 2)
    np.testing.assert_allclose(e, [3.0, 4.0])
    # angle->node (center of edge_a): all 3 angles share center 0 => 7
    n = angle_to_node_sum(data, angle_index, edge_index, 1)
    np.testing.assert_allclose(n, [7.0])


def test_angle_aggregation_torch_namespace():
    """Step 4b: torch-namespace smoke test for angle aggregation."""
    import torch

    # edges: dst=[0,0]; angles: (a=0,b=0),(a=0,b=1),(a=1,b=0)
    edge_index = np.array([[5, 5], [0, 0]], dtype=np.int64)
    angle_index = np.array([[0, 0, 1], [0, 1, 0]], dtype=np.int64)
    data = np.array([1.0, 2.0, 4.0])  # per-angle

    # numpy reference
    e_np = angle_to_edge_sum(data, angle_index, 2)
    n_np = angle_to_node_sum(data, angle_index, edge_index, 1)

    # torch version
    t_edge_index = torch.from_numpy(edge_index)
    t_angle_index = torch.from_numpy(angle_index)
    t_data = torch.from_numpy(data)

    e_t = angle_to_edge_sum(t_data, t_angle_index, 2)
    n_t = angle_to_node_sum(t_data, t_angle_index, t_edge_index, 1)

    # compare
    np.testing.assert_allclose(np.asarray(e_t), e_np)
    np.testing.assert_allclose(np.asarray(n_t), n_np)


# ---------------------------------------------------------------------------
# Task 6: angle-force invariance + angle_padding_fraction
# ---------------------------------------------------------------------------


def _small_graph_with_angles(a_rcut: float, layout: GraphLayout | None = None):
    """Return (graph_no_angles, graph_with_angles) for a 3-atom, 1-frame system."""
    # 3 atoms: 0,1,2 in a line along x; shape (nf=1, nloc=3, 3) / (nf=1, nloc=3)
    coord = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    atype = np.array([[0, 1, 0]], dtype=np.int64)
    box = np.array([[[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]])
    g = build_neighbor_graph(coord, atype, box, rcut=3.0)
    g_with = attach_angles(g, a_rcut, layout=layout)
    return g, g_with


def test_edge_force_virial_ignores_angles():
    """edge_force_virial output must be bit-identical with or without angles.

    Angles add topology (angle_index/angle_mask) to the NeighborGraph but do
    NOT change edge_vec, edge_index, or edge_mask — the only inputs to
    edge_force_virial.  This test proves that the angle fields are truly
    transparent to the force/virial assembly.
    """
    g_bare, g_with_angles = _small_graph_with_angles(a_rcut=1.5)

    # Manufacture a fake per-edge gradient (same shape as edge_vec)
    rng = np.random.default_rng(42)
    n_edges = int(g_bare.edge_index.shape[1])
    g_e = rng.standard_normal((n_edges, 3))

    def run(graph):
        return edge_force_virial(
            g_e,
            graph.edge_vec,
            graph.edge_index,
            graph.edge_mask,
            graph.n_node,
        )

    force_bare, av_bare, vir_bare = run(g_bare)
    force_with, av_with, vir_with = run(g_with_angles)

    # Exact equality: same inputs → same computation → identical bits
    np.testing.assert_array_equal(force_bare, force_with)
    np.testing.assert_array_equal(av_bare, av_with)
    np.testing.assert_array_equal(vir_bare, vir_with)


def test_angle_padding_fraction():
    """angle_padding_fraction returns 1 - A_real/A_max for a static layout.

    We build with a fixed angle_capacity=A_max so the fraction is deterministic
    (not influenced by the dynamic min_angles guard of pad_and_guard_angles).
    """
    A_max = 20  # static capacity, larger than any real angle count
    layout = GraphLayout(angle_capacity=A_max)
    g_bare, g_with = _small_graph_with_angles(a_rcut=1.5, layout=layout)

    # Confirm angles are present
    assert g_with.angle_mask is not None
    A_real = int(np.sum(g_with.angle_mask))
    assert 0 < A_real <= A_max, f"Expected 0 < A_real <= {A_max}, got {A_real}"

    expected = 1.0 - A_real / A_max
    got = angle_padding_fraction(g_with)
    assert got == pytest.approx(expected), f"got {got}, expected {expected}"

    # No angles → fraction is 0.0
    assert angle_padding_fraction(g_bare) == 0.0


def test_angle_padding_fraction_total_zero():
    """angle_padding_fraction returns 0.0 when angle_mask.shape[0] == 0.

    This exercises the `if total == 0: return 0.0` branch in angle_padding_fraction.
    A graph with angle_capacity=0 and a_rcut too small for any angles produces
    angle_mask with shape (0,).
    """
    # Create a layout with angle_capacity=0 and use a_rcut=0.01 (too small
    # for any edge in the fixture to pass the distance gate)
    layout = GraphLayout(angle_capacity=0)
    g_bare, g_with = _small_graph_with_angles(a_rcut=0.01, layout=layout)

    # Verify angle_mask exists and has shape (0,)
    assert g_with.angle_mask is not None
    assert g_with.angle_mask.shape[0] == 0

    # angle_padding_fraction must return 0.0 for empty mask
    result = angle_padding_fraction(g_with)
    assert result == 0.0
