# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.dpmodel.utils.exclude_mask import PairExcludeMask
from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
    apply_pair_exclusion,
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
