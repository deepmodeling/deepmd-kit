import numpy as np
import pytest

from deepmd.dpmodel.utils.neighbor_graph import (
    GraphLayout,
    build_angle_index,
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
