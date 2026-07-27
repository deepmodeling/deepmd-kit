# SPDX-License-Identifier: LGPL-3.0-or-later
"""center_edge_pairs: pairs of edges sharing a center (NeighborGraph PR-D/E)."""

import numpy as np

from deepmd.dpmodel.utils.neighbor_graph import (
    center_edge_pairs,
)


def _oracle(dst, mask, include_self, ordered):
    pairs = []
    for m in range(len(dst)):
        if not mask[m]:
            continue
        for n in range(len(dst)):
            if not mask[n] or dst[m] != dst[n]:
                continue
            if not include_self and m == n:
                continue
            if not ordered and n < m:
                continue
            pairs.append((m, n))
    return set(pairs)


def _got(q, k, pm):
    return {(int(q[p]), int(k[p])) for p in range(q.shape[0]) if pm[p]}


class TestCompact:
    def test_transformer_all_ordered_with_self(self) -> None:
        # 3 edges: dst = [0, 0, 1]; center 0 has edges {0,1}, center 1 has {2}
        dst = np.array([0, 0, 1], dtype=np.int64)
        edge_mask = np.array([True, True, True])
        q, k, pm = center_edge_pairs(dst, edge_mask, 2)
        assert _got(q, k, pm) == _oracle([0, 0, 1], [1, 1, 1], True, True)
        # center 0: (0,0),(0,1),(1,0),(1,1); center 1: (2,2) => 5 pairs
        assert len(_got(q, k, pm)) == 5

    def test_unordered_no_self_is_angle_set(self) -> None:
        dst = np.array([0, 0, 0], dtype=np.int64)
        edge_mask = np.array([True, True, True])
        q, k, pm = center_edge_pairs(
            dst, edge_mask, 1, include_self=False, ordered=False
        )
        assert _got(q, k, pm) == {(0, 1), (0, 2), (1, 2)}

    def test_ignores_padding_edges(self) -> None:
        dst = np.array([0, 0, 0], dtype=np.int64)
        edge_mask = np.array([True, True, False])  # 3rd is a guard edge
        q, k, pm = center_edge_pairs(dst, edge_mask, 1)
        assert _got(q, k, pm) == {(0, 0), (0, 1), (1, 0), (1, 1)}

    def test_non_contiguous_center_order(self) -> None:
        # edges NOT sorted by center: dst = [1, 0, 1, 0]
        dst = np.array([1, 0, 1, 0], dtype=np.int64)
        edge_mask = np.array([True, True, True, True])
        q, k, pm = center_edge_pairs(dst, edge_mask, 2)
        assert _got(q, k, pm) == _oracle([1, 0, 1, 0], [1] * 4, True, True)

    def test_empty(self) -> None:
        dst = np.zeros((0,), dtype=np.int64)
        edge_mask = np.zeros((0,), dtype=bool)
        q, k, pm = center_edge_pairs(dst, edge_mask, 3)
        assert q.shape[0] == 0 and k.shape[0] == 0 and pm.shape[0] == 0

    def test_random_vs_oracle(self) -> None:
        rng = np.random.default_rng(7)
        dst = rng.integers(0, 5, size=23).astype(np.int64)
        edge_mask = rng.random(23) > 0.3
        for include_self in (True, False):
            for ordered in (True, False):
                q, k, pm = center_edge_pairs(
                    dst, edge_mask, 5, include_self=include_self, ordered=ordered
                )
                assert _got(q, k, pm) == _oracle(
                    dst, edge_mask, include_self, ordered
                ), (include_self, ordered)

    def test_torch_matches_numpy(self) -> None:
        import torch

        dst = np.array([0, 0, 1, 1, 1], dtype=np.int64)
        edge_mask = np.array([True, False, True, True, True])
        ref = _got(*center_edge_pairs(dst, edge_mask, 2))
        q, k, pm = center_edge_pairs(
            torch.from_numpy(dst), torch.from_numpy(edge_mask), 2
        )
        assert _got(q.numpy(), k.numpy(), pm.numpy()) == ref


class TestShapeStatic:
    def test_matches_compact(self) -> None:
        # center-major static layout: 2 centers x static_nnei=3, edges 2,5 padded
        dst = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        edge_mask = np.array([True, True, False, True, True, False])
        qc, kc, pmc = center_edge_pairs(dst, edge_mask, 2)
        qs, ks, pms = center_edge_pairs(dst, edge_mask, 2, static_nnei=3)
        assert qs.shape[0] == 2 * 3 * 3  # static P, data-independent
        assert _got(qs, ks, pms) == _got(qc, kc, pmc)

    def test_flags_and_masking(self) -> None:
        dst = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        edge_mask = np.array([True, True, True, True, False, False])
        for include_self in (True, False):
            for ordered in (True, False):
                qs, ks, pms = center_edge_pairs(
                    dst,
                    edge_mask,
                    2,
                    include_self=include_self,
                    ordered=ordered,
                    static_nnei=3,
                )
                assert qs.shape[0] == 2 * 3 * 3  # P static regardless of flags
                assert _got(qs, ks, pms) == _oracle(
                    dst, edge_mask, include_self, ordered
                ), (include_self, ordered)

    def test_torch_matches_numpy(self) -> None:
        import torch

        dst = np.array([0, 0, 1, 1], dtype=np.int64)
        edge_mask = np.array([True, False, True, True])
        ref = _got(*center_edge_pairs(dst, edge_mask, 2, static_nnei=2))
        q, k, pm = center_edge_pairs(
            torch.from_numpy(dst),
            torch.from_numpy(edge_mask),
            2,
            static_nnei=2,
        )
        assert _got(q.numpy(), k.numpy(), pm.numpy()) == ref
