# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-native (edge-stream) environment-matrix Triton kernel.

The edge form (:func:`deepmd.kernels.triton.env_mat.edge_env_mat`) is the
slot-free analogue used only by the pt_expt graph lower: the relative vector
``edge_vec`` is given directly (no neighbor gather) and the backward
differentiates ``edge_vec`` (the graph-path force leaf), so no scatter is
needed. These tests check parity against the dpmodel reference
(:func:`deepmd.dpmodel.utils.neighbor_graph.env.edge_env_mat`) in fp32 and fp64,
including padding (zero-vector) edges, the optionally returned smooth switch
``sw`` and its gradient (the strip type-pair gate consumes it), and
composability under ``make_fx``.
"""

import os
import unittest

import torch

from deepmd.kernels.triton.env_mat import (
    TRITON_AVAILABLE,
    edge_env_mat,
)

_GPU = unittest.skipUnless(
    torch.cuda.is_available() and TRITON_AVAILABLE,
    "CUDA and Triton are required for the edge env_mat kernel",
)


def _make_edges(dtype, device, n_edge=4000, ntypes=3, seed=0):
    """Random edge stream with padding (zero-vector) edges, as on the graph path."""
    g = torch.Generator(device=device).manual_seed(seed)
    edge_vec = torch.randn(n_edge, 3, generator=g, device=device, dtype=dtype) * 2.0
    edge_mask = torch.rand(n_edge, generator=g, device=device) > 0.2
    edge_vec[~edge_mask] = 0.0
    center_type = torch.randint(0, ntypes, (n_edge,), generator=g, device=device)
    davg = torch.randn(ntypes, 4, generator=g, device=device, dtype=dtype)
    dstd = 0.5 + torch.rand(ntypes, 4, generator=g, device=device, dtype=dtype)
    return edge_vec, center_type, edge_mask, davg, dstd


@_GPU
class TestEdgeEnvMatTriton(unittest.TestCase):
    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")

    def tearDown(self) -> None:
        os.environ.pop("DP_TRITON_INFER", None)

    def test_forward_and_edge_grad_parity(self) -> None:
        from deepmd.dpmodel.utils.neighbor_graph.env import edge_env_mat as edge_ref

        rcut, rcut_smth = 6.0, 2.0
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                ev, ct, mask, davg, dstd = _make_edges(dtype, self.device)
                gout = torch.randn(ev.shape[0], 4, device=self.device, dtype=dtype)

                os.environ["DP_TRITON_INFER"] = "1"
                e1 = ev.clone().requires_grad_()
                r1 = edge_env_mat(e1, ct, davg, dstd, rcut, rcut_smth, edge_mask=mask)
                (g1,) = torch.autograd.grad((r1 * gout).sum(), e1)

                e0 = ev.clone().requires_grad_()
                r0 = edge_ref(e0, ct, davg, dstd, rcut, rcut_smth, edge_mask=mask)
                (g0,) = torch.autograd.grad((r0 * gout).sum(), e0)

                ftol = 1e-5 if dtype == torch.float32 else 1e-10
                gtol = 1e-3 if dtype == torch.float32 else 1e-8
                self.assertLess((r1 - r0).abs().max().item(), ftol)
                self.assertLess(
                    (g1 - g0).abs().max().item() / (g0.abs().max().item() + 1e-30),
                    gtol,
                )

    def test_return_sw_parity(self) -> None:
        # ``return_sw`` also emits the per-edge switch (strip gate input). Feed
        # independent cotangents into ``env`` and ``sw`` so the backward
        # exercises the env path, the switch path (the fused ``g_sw * s'`` term)
        # and their sum into the ``edge_vec`` leaf.
        from deepmd.dpmodel.utils.neighbor_graph.env import edge_env_mat as edge_ref

        rcut, rcut_smth = 6.0, 2.0
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                ev, ct, mask, davg, dstd = _make_edges(dtype, self.device)
                g_env = torch.randn(ev.shape[0], 4, device=self.device, dtype=dtype)
                g_sw = torch.randn(ev.shape[0], 1, device=self.device, dtype=dtype)

                os.environ["DP_TRITON_INFER"] = "1"
                e1 = ev.clone().requires_grad_()
                env1, sw1 = edge_env_mat(
                    e1, ct, davg, dstd, rcut, rcut_smth, edge_mask=mask, return_sw=True
                )
                loss1 = (env1 * g_env).sum() + (sw1 * g_sw).sum()
                (g1,) = torch.autograd.grad(loss1, e1)

                e0 = ev.clone().requires_grad_()
                env0, sw0 = edge_ref(
                    e0, ct, davg, dstd, rcut, rcut_smth, edge_mask=mask, return_sw=True
                )
                loss0 = (env0 * g_env).sum() + (sw0 * g_sw).sum()
                (g0,) = torch.autograd.grad(loss0, e0)

                ftol = 1e-5 if dtype == torch.float32 else 1e-10
                gtol = 1e-3 if dtype == torch.float32 else 1e-8
                self.assertEqual(sw1.shape, (ev.shape[0], 1))
                self.assertLess((sw1 - sw0).abs().max().item(), ftol)
                self.assertLess((env1 - env0).abs().max().item(), ftol)
                self.assertLess(
                    (g1 - g0).abs().max().item() / (g0.abs().max().item() + 1e-30),
                    gtol,
                )

    def test_make_fx_opaque_operator(self) -> None:
        from torch.fx.experimental.proxy_tensor import (
            make_fx,
        )

        os.environ["DP_TRITON_INFER"] = "1"
        ev, ct, mask, davg, dstd = _make_edges(torch.float32, self.device)

        def fn(edge_vec, center_type, edge_mask, avg, std):
            return edge_env_mat(
                edge_vec, center_type, avg, std, 6.0, 2.0, edge_mask=edge_mask
            ).sum()

        gm = make_fx(fn, tracing_mode="fake")(ev, ct, mask, davg, dstd)
        targets = [str(n.target) for n in gm.graph.nodes]
        self.assertTrue(any("edge_env_mat.default" in t for t in targets))


if __name__ == "__main__":
    unittest.main()
