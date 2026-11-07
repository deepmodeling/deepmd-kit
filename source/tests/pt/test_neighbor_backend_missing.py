# SPDX-License-Identifier: LGPL-3.0-or-later
"""Native fallback retains every cutoff edge without optional search libraries."""

import unittest
import warnings
from unittest import (
    mock,
)

import torch

from deepmd.pt_expt.utils import (
    auto_neighbor_list,
)


class TestNeighborBackendMissing(unittest.TestCase):
    def setUp(self) -> None:
        for name in ("is_nv_available", "is_vesin_torch_available"):
            patcher = mock.patch.object(auto_neighbor_list, name, return_value=False)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_dense_fallback_reports_cost_without_repeating(self) -> None:
        coord = torch.zeros((1, 1, 3), dtype=torch.float64, device="cpu")
        atype = torch.zeros((1, 1), dtype=torch.int64, device="cpu")
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("default", RuntimeWarning)
            for _ in range(3):
                auto_neighbor_list.AdaptiveNeighborList(method="native").build(
                    coord, atype, None, 2.0, [0], return_mode="edges"
                )
        self.assertEqual(len(recorded), 1)
        self.assertIn("O(N^2)", str(recorded[0].message))
        self.assertIn("nvalchemi-toolkit-ops", str(recorded[0].message))

    def test_native_edges_are_not_truncated_to_sel(self) -> None:
        """Batched real atoms keep all edges and phantom atoms keep none."""
        coord = torch.tensor(
            [[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.9, 0.0], [0.1, 0.0, 0.0]]],
            dtype=torch.float64,
            device="cpu",
        )
        coord = coord.repeat(2, 1, 1).requires_grad_()
        atype = torch.tensor([[0, 1, 0, -1]], device="cpu").repeat(2, 1)
        for method in ("native", "auto"):
            with self.subTest(method=method):
                builder = auto_neighbor_list.make_neighbor_list(
                    method, coord.device, return_mode="edges"
                )
                edges = builder.build(
                    coord, atype, None, 2.0, [1, 0], return_mode="edges"
                )
                indices = edges.edge_index[:, edges.edge_mask]
                self.assertEqual(indices.shape[1], 12)
                self.assertTrue((atype.reshape(-1)[indices] >= 0).all())
                self.assertTrue((indices[0] // 4 == indices[1] // 4).all())
                grad = torch.autograd.grad(edges.edge_vec.square().sum(), coord)[0]
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().sum()), 0.0)

    def test_native_edges_keep_periodic_self_images(self) -> None:
        coord = torch.zeros((1, 1, 3), dtype=torch.float64, device="cpu")
        atype = torch.zeros((1, 1), dtype=torch.int64, device="cpu")
        box = torch.eye(3, dtype=coord.dtype, device="cpu").unsqueeze(0)
        builder = auto_neighbor_list.make_neighbor_list(
            "auto", coord.device, return_mode="edges"
        )
        edges = builder.build(coord, atype, box, 1.1, [0], return_mode="edges")
        self.assertEqual(int(edges.edge_mask.sum()), 6)
        self.assertTrue((edges.edge_index[:, edges.edge_mask] == 0).all())
        torch.testing.assert_close(
            edges.edge_vec[edges.edge_mask].square().sum(-1),
            torch.ones(6, dtype=coord.dtype, device="cpu"),
        )


if __name__ == "__main__":
    unittest.main()
