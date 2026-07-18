# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.dpmodel.utils.exclude_mask import (
    PairExcludeMask,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    apply_pair_exclusion,
    attach_edge_csr,
)


def _toy_graph():
    # 4 nodes, types [0, 1, 0, 1]; 5 edges incl. one already-masked pad edge.
    # edge_index rows: [src(neighbor), dst(center)]
    edge_index = np.array([[1, 2, 3, 0, 0], [0, 0, 1, 3, 0]], dtype=np.int64)
    edge_vec = np.ones((5, 3), dtype=np.float64)
    edge_mask = np.array([1, 1, 1, 1, 0], dtype=np.int32)  # last = padding
    n_node = np.array([4], dtype=np.int64)
    return NeighborGraph(
        n_node=n_node,
        edge_index=edge_index,
        edge_vec=edge_vec,
        edge_mask=edge_mask,
    )


def test_none_and_empty_are_identity() -> None:
    g = _toy_graph()
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    assert apply_pair_exclusion(g, atype, None) is g
    assert apply_pair_exclusion(g, atype, PairExcludeMask(2, [])) is g


def test_excluded_pairs_are_masked_and_padding_stays_masked() -> None:
    g = _toy_graph()
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    out = apply_pair_exclusion(g, atype, PairExcludeMask(2, [(0, 1)]))
    # edges (dst_t, src_t): e0 (0,1) excl, e1 (0,0) keep, e2 (1,1) keep,
    # e3 (1,0) excl (symmetric), e4 padding stays 0.
    np.testing.assert_array_equal(out.edge_mask, [0, 1, 1, 0, 0])
    # non-mask fields untouched, input not mutated
    np.testing.assert_array_equal(g.edge_mask, [1, 1, 1, 1, 0])
    assert out.edge_index is g.edge_index
    assert out.edge_vec is g.edge_vec


def test_exclusion_invalidates_derived_csr_views() -> None:
    graph = attach_edge_csr(_toy_graph(), 4)
    atype = np.array([0, 1, 0, 1], dtype=np.int64)

    out = apply_pair_exclusion(graph, atype, PairExcludeMask(2, [(0, 1)]))

    assert out.destination_order is None
    assert out.destination_row_ptr is None
    assert out.source_order is None
    assert out.source_row_ptr is None
    assert out.destination_sorted is False


def test_no_exclusion_empty_list_is_identity() -> None:
    """Cover PairExcludeMask with non-None but empty exclude list."""
    g = _toy_graph()
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    result = apply_pair_exclusion(g, atype, PairExcludeMask(2, []))
    assert result is g


def test_no_excluded_edges_in_graph() -> None:
    """Exclusion list non-empty but no edge matches — all real edges stay."""
    g = _toy_graph()
    atype = np.array([0, 0, 0, 0], dtype=np.int64)  # all same type
    out = apply_pair_exclusion(g, atype, PairExcludeMask(2, [(0, 1)]))
    # (0,1) never appears — all edges kept (except pre-existing padding)
    np.testing.assert_array_equal(out.edge_mask, [1, 1, 1, 1, 0])


def test_torch_namespace_smoke() -> None:
    torch = pytest.importorskip("torch")
    g = _toy_graph()
    gt = NeighborGraph(
        n_node=torch.from_numpy(g.n_node),
        edge_index=torch.from_numpy(g.edge_index),
        edge_vec=torch.from_numpy(g.edge_vec),
        edge_mask=torch.from_numpy(g.edge_mask),
    )
    atype = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    out = apply_pair_exclusion(gt, atype, PairExcludeMask(2, [(0, 1)]))
    np.testing.assert_array_equal(out.edge_mask.numpy(), [0, 1, 1, 0, 0])


# ---------------------------------------------------------------------------
# compact=True tests
# ---------------------------------------------------------------------------


def test_compact_drops_masked_edges() -> None:
    """compact=True must keep exactly the valid edges (after exclusion)."""
    g = _toy_graph()
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    out_mask = apply_pair_exclusion(g, atype, PairExcludeMask(2, [(0, 1)]))
    out_compact = apply_pair_exclusion(
        g, atype, PairExcludeMask(2, [(0, 1)]), compact=True
    )
    # Expected kept edges: indices 1 and 2 (mask-only has [0,1,1,0,0])
    assert out_compact.edge_index.shape[1] == 2
    assert out_compact.edge_vec.shape[0] == 2
    assert out_compact.edge_mask.shape[0] == 2
    # all remaining edge_mask entries must be 1
    np.testing.assert_array_equal(out_compact.edge_mask, [1, 1])
    # edge_index content matches kept edges from mask path
    np.testing.assert_array_equal(
        out_compact.edge_index, out_mask.edge_index[:, [1, 2]]
    )


def test_compact_drops_preexisting_padding_too() -> None:
    """Pre-existing padding (edge 4) must be dropped even with no exclusions."""
    g = _toy_graph()
    atype = np.array([0, 0, 0, 0], dtype=np.int64)  # no type exclusions
    # compact=True with empty exclusion list -> graph has no exclusion keep_idx change
    # The brief says compact on identity returns graph unchanged,
    # but with a non-empty excl list that matches nothing, out has same edge_mask as g.
    # Let's use a real exclusion that changes something so compact is non-trivial:
    out = apply_pair_exclusion(g, atype, PairExcludeMask(2, [(0, 1)]), compact=True)
    # (0,1) never appears in this graph (all types are 0) → all real edges kept
    # but pre-existing padding (edge 4) should be dropped
    assert out.edge_index.shape[1] == 4  # only 4 real edges, padding gone
    np.testing.assert_array_equal(out.edge_mask, [1, 1, 1, 1])


def test_compact_torch_smoke() -> None:
    torch = pytest.importorskip("torch")
    g = _toy_graph()
    gt = NeighborGraph(
        n_node=torch.from_numpy(g.n_node),
        edge_index=torch.from_numpy(g.edge_index),
        edge_vec=torch.from_numpy(g.edge_vec),
        edge_mask=torch.from_numpy(g.edge_mask),
    )
    atype = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    out = apply_pair_exclusion(gt, atype, PairExcludeMask(2, [(0, 1)]), compact=True)
    np.testing.assert_array_equal(out.edge_mask.numpy(), [1, 1])
    assert out.edge_index.shape[1] == 2


def test_compact_invariance_vs_mask_only() -> None:
    """Descriptor-level invariance: segment_sum over mask-only == compact.

    Masked edges contribute zero to the sum; dropping them should give identical
    results.
    """
    g = _toy_graph()
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    excl = PairExcludeMask(2, [(0, 1)])

    out_mask = apply_pair_exclusion(g, atype, excl, compact=False)
    out_compact = apply_pair_exclusion(g, atype, excl, compact=True)

    # Build fake per-edge values (like edge_env_mat output)
    vals_mask = np.arange(5, dtype=np.float64).reshape(5, 1) + 1.0
    vals_mask_valid = vals_mask * out_mask.edge_mask[:, None]

    # Map compact edge indices to the original edge values
    # Kept edges are 1 and 2 (mask [0,1,1,0,0])
    vals_compact = vals_mask[out_mask.edge_mask.astype(bool)]

    # segment_sum over dst (center) node axis: 4 nodes
    N = 4
    dst_mask = out_mask.edge_index[1]  # (5,)
    dst_compact = out_compact.edge_index[1]  # (2,)

    # manual segment_sum for mask path
    result_mask = np.zeros((N, 1), dtype=np.float64)
    for ei, v in zip(dst_mask, vals_mask_valid, strict=True):
        result_mask[ei] += v

    result_compact = np.zeros((N, 1), dtype=np.float64)
    for ei, v in zip(dst_compact, vals_compact, strict=True):
        result_compact[ei] += v

    np.testing.assert_allclose(result_mask, result_compact)


# ---------------------------------------------------------------------------
# compact=True with angle fields — remap onto the compacted edge axis
# ---------------------------------------------------------------------------


def _toy_graph_with_angles():
    """Same base graph as _toy_graph but with angle_index/angle_mask populated."""
    g = _toy_graph()
    import dataclasses

    # Two toy angles (pairs of edges sharing a center), into edge positions [0,5)
    # angle0 = (edge0, edge1);  angle1 = (edge1, edge2)
    angle_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    angle_mask = np.array([1, 1], dtype=np.int32)
    return dataclasses.replace(g, angle_index=angle_index, angle_mask=angle_mask)


def test_compact_drops_angles_touching_excluded_edges() -> None:
    """compact=True remaps angle_index and drops angles whose edges were excluded."""
    g = _toy_graph_with_angles()
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    # exclusion (0,1) masks edges 0 and 3 (see the mask-only test): survivors are
    # old edges [1, 2] -> new positions [0, 1]; padding edge 4 is dropped too.
    out = apply_pair_exclusion(g, atype, PairExcludeMask(2, [(0, 1)]), compact=True)
    # edges compacted to the two survivors
    assert out.edge_index.shape[1] == 2
    # angle0 = (edge0, edge1): edge0 excluded -> angle0 DROPPED.
    # angle1 = (edge1, edge2): both survive -> kept, remapped (edge1->0, edge2->1).
    np.testing.assert_array_equal(out.angle_index, [[0], [1]])
    np.testing.assert_array_equal(out.angle_mask, [1])


def test_compact_remaps_angles_when_no_angle_dropped() -> None:
    """When exclusion drops no referenced edge, angles are remapped, none lost."""
    g = _toy_graph_with_angles()
    atype = np.array([0, 0, 0, 0], dtype=np.int64)  # (0,1) matches nothing
    # all 4 real edges kept (new positions == old for [0,1,2,3]); padding dropped.
    out = apply_pair_exclusion(g, atype, PairExcludeMask(2, [(0, 1)]), compact=True)
    assert out.edge_index.shape[1] == 4
    # both angles survive; indices unchanged (survivor ranks equal old positions)
    np.testing.assert_array_equal(out.angle_index, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(out.angle_mask, [1, 1])


def test_compact_raises_when_only_angle_index_present() -> None:
    """compact=True fails fast when angle_index is set but angle_mask is None."""
    import dataclasses

    g = _toy_graph()
    angle_index = np.array([[0, 1], [1, 2]], dtype=np.int64)
    g2 = dataclasses.replace(g, angle_index=angle_index)  # angle_mask stays None
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="both be set or both be None"):
        apply_pair_exclusion(g2, atype, PairExcludeMask(2, [(0, 1)]), compact=True)


def test_compact_raises_when_only_angle_mask_present() -> None:
    """compact=True fails fast when angle_mask is set but angle_index is None."""
    import dataclasses

    g = _toy_graph()
    angle_mask = np.array([1], dtype=np.int32)
    g_with_mask = dataclasses.replace(g, angle_mask=angle_mask)
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="both be set or both be None"):
        apply_pair_exclusion(
            g_with_mask, atype, PairExcludeMask(2, [(0, 1)]), compact=True
        )


def test_compact_raises_on_angle_dim_mismatch() -> None:
    """compact=True fails fast when angle_index (2,A) and angle_mask (A,) disagree."""
    import dataclasses

    g = _toy_graph()
    angle_index = np.array([[0, 1], [1, 2]], dtype=np.int64)  # A == 2
    angle_mask = np.array([1], dtype=np.int32)  # A == 1  (mismatch)
    g2 = dataclasses.replace(g, angle_index=angle_index, angle_mask=angle_mask)
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="disagree on A"):
        apply_pair_exclusion(g2, atype, PairExcludeMask(2, [(0, 1)]), compact=True)


def test_compact_raises_on_bad_angle_index_shape() -> None:
    """compact=True fails fast when angle_index is not (2, A)."""
    import dataclasses

    g = _toy_graph()
    angle_index = np.array([[0, 1, 2]], dtype=np.int64)  # (1, 3), not (2, A)
    angle_mask = np.array([1, 1, 1], dtype=np.int32)
    g2 = dataclasses.replace(g, angle_index=angle_index, angle_mask=angle_mask)
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match=r"shape \(2, A\)"):
        apply_pair_exclusion(g2, atype, PairExcludeMask(2, [(0, 1)]), compact=True)


def test_compact_angle_torch_smoke() -> None:
    """compact-mode angle remap runs under the torch namespace."""
    torch = pytest.importorskip("torch")
    g = _toy_graph_with_angles()
    gt = NeighborGraph(
        n_node=torch.from_numpy(g.n_node),
        edge_index=torch.from_numpy(g.edge_index),
        edge_vec=torch.from_numpy(g.edge_vec),
        edge_mask=torch.from_numpy(g.edge_mask),
        angle_index=torch.from_numpy(g.angle_index),
        angle_mask=torch.from_numpy(g.angle_mask),
    )
    atype = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    out = apply_pair_exclusion(gt, atype, PairExcludeMask(2, [(0, 1)]), compact=True)
    np.testing.assert_array_equal(out.angle_index.numpy(), [[0], [1]])
    np.testing.assert_array_equal(out.angle_mask.numpy(), [1])


def test_compact_works_when_angle_fields_are_none() -> None:
    """compact=True must NOT raise when angle_index and angle_mask are both None."""
    g = _toy_graph()  # angle_index=None, angle_mask=None by default
    atype = np.array([0, 1, 0, 1], dtype=np.int64)
    # Should succeed; reuse the existing compact assertion
    out = apply_pair_exclusion(g, atype, PairExcludeMask(2, [(0, 1)]), compact=True)
    assert out.edge_index.shape[1] == 2
