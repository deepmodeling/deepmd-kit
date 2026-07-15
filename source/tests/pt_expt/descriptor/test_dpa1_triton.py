# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-lower Triton path of the pt_expt DPA1 descriptor.

The graph lower (``--lower-kind graph``) represents the neighbor list as a flat
edge stream. When ``DP_TRITON_INFER >= 1`` the attention-free block routes
``call_graph`` through the fused edge-parallel
:func:`~deepmd.kernels.triton.dpa1.edge_conv.edge_conv` operator, for both
tebd-input modes: ``concat`` (the type feature enters the embedding input, no
gate) and ``strip`` (the type feature factorizes into the type-pair gate
``gg = gg_s * (1 + tt[idx] * sw)``, fed by the per-edge switch from
``edge_env_mat(return_sw=True)``).

Two properties are covered:

1. Parity: the fused ``_call_graph_triton`` reproduces the dpmodel reference
   ``call_graph`` (descriptor grrg, rotation matrix and the edge_vec force
   gradient) across residual structures, one/two-side, activation and
   non-power-of-two widths, in both tebd-input modes (strip with the smooth
   switch on and off).
2. ``make_fx`` bake: ``edge_conv`` traces as a single opaque node (the pt_expt
   graph-form ``.pt2`` export target), and ``call_graph`` routes to it only at
   ``DP_TRITON_INFER >= 1``.
"""

import dataclasses
import os
import unittest

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.kernels.triton.dpa1.activation import (
    TRITON_AVAILABLE,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

_CUDA = torch.cuda.is_available()
_GPU = unittest.skipUnless(
    _CUDA and TRITON_AVAILABLE,
    "CUDA and Triton are required for the DPA1 graph edge kernel",
)


def _build_dpa1_expt(
    device,
    neuron,
    one_side=False,
    act="tanh",
    precision="float64",
    tebd_input_mode="concat",
    smooth=True,
):
    from deepmd.pt_expt.descriptor.dpa1 import (
        DescrptDPA1,
    )

    des = DescrptDPA1(
        rcut=6.0,
        rcut_smth=2.0,
        sel=40,
        ntypes=2,
        neuron=neuron,
        axis_neuron=4,
        tebd_dim=8,
        tebd_input_mode=tebd_input_mode,
        attn_layer=0,
        type_one_side=one_side,
        activation_function=act,
        precision=precision,
        smooth_type_embedding=smooth,
        seed=1,
    ).to(device)
    des.eval()
    return des


@_GPU
class TestDpa1GraphRouting(unittest.TestCase):
    """Graph-lower parity and make_fx bake of the fused ``edge_conv`` route."""

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        gen = torch.Generator(device=self.device).manual_seed(3)
        n = 24
        self.coord = (torch.rand(1, n, 3, generator=gen, device=self.device) * 8.0).to(
            torch.float64
        )
        self.atype = torch.randint(0, 2, (1, n), generator=gen, device=self.device)
        self.box = (
            (torch.eye(3, device=self.device) * 8.0).reshape(1, 9).to(torch.float64)
        )

    def _graph(self, des):
        from deepmd.dpmodel.utils.neighbor_graph import (
            from_dense_quartet,
        )

        ec, ea, mp, nl = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            des.get_rcut(),
            des.get_sel(),
            mixed_types=des.mixed_types(),
            box=self.box,
        )
        graph = from_dense_quartet(ec, nl, mp, compact=True)
        return graph, self.atype.reshape(-1).to(self.device)

    def _assert_parity(self, des) -> None:
        from deepmd.dpmodel.descriptor.dpa1 import DescrptDPA1 as DescrptDPA1DP

        self.assertTrue(des._fused_eligible("triton"))
        tebd = des.type_embedding.call()
        graph, atype_local = self._graph(des)

        def run(fn):
            ev = graph.edge_vec.detach().clone().requires_grad_(True)
            g = dataclasses.replace(graph, edge_vec=ev)
            grrg, rot = fn(g)
            (gvec,) = torch.autograd.grad(grrg.sum(), ev)
            return grrg.detach(), rot.detach(), gvec.detach()

        d0, r0, g0 = run(
            lambda g: DescrptDPA1DP.call_graph(des, g, atype_local, type_embedding=tebd)
        )
        d1, r1, g1 = run(
            lambda g: des._call_graph_triton(g, atype_local, type_embedding=tebd)
        )
        torch.testing.assert_close(d1, d0, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(r1, r0, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(g1, g0, atol=1e-6, rtol=1e-6)

    def test_parity_concat_doubling(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [8, 16, 32]))

    def test_parity_concat_identity(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [8, 16, 16]))

    def test_parity_concat_one_side(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [8, 16, 32], one_side=True))

    def test_parity_concat_silu(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [8, 16, 32], act="silu"))

    def test_parity_concat_nonpow2(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [12, 25, 50]))

    def _strip(self, neuron, **kw):
        return _build_dpa1_expt(self.device, neuron, tebd_input_mode="strip", **kw)

    def test_parity_strip_doubling(self) -> None:
        # Strip: the type feature enters as the multiplicative type-pair gate
        # ``1 + tt[idx] * sw`` (smooth switch on by default); ``edge_conv`` runs
        # with ``gated == 1``.
        self._assert_parity(self._strip([8, 16, 32]))

    def test_parity_strip_identity(self) -> None:
        self._assert_parity(self._strip([8, 16, 16]))

    def test_parity_strip_one_side(self) -> None:
        # One-side folds the gate index by neighbor type alone (vs. the two-side
        # (center, neighbor) pair).
        self._assert_parity(self._strip([8, 16, 32], one_side=True))

    def test_parity_strip_silu(self) -> None:
        self._assert_parity(self._strip([8, 16, 32], act="silu"))

    def test_parity_strip_nonpow2(self) -> None:
        self._assert_parity(self._strip([12, 25, 50]))

    def test_parity_strip_nonsmooth(self) -> None:
        # Non-smooth strip drops the switch from the gate (``sw`` -> ones), the
        # other ``gated == 1`` branch of ``_call_graph_triton``.
        self._assert_parity(self._strip([8, 16, 32], smooth=False))

    def test_make_fx_bakes_edge_conv(self) -> None:
        des = _build_dpa1_expt(self.device, [8, 16, 32])
        tebd = des.type_embedding.call()
        graph, atype_local = self._graph(des)

        def fn(edge_vec):
            g = dataclasses.replace(graph, edge_vec=edge_vec)
            return des._call_graph_triton(g, atype_local, type_embedding=tebd)

        traced = make_fx(fn, tracing_mode="real")(graph.edge_vec)
        targets = [str(n.target) for n in traced.graph.nodes]
        self.assertGreaterEqual(sum("edge_conv" in t for t in targets), 1)

    def test_make_fx_bakes_edge_conv_strip(self) -> None:
        # Strip bakes both the gated ``edge_conv`` and the ``return_sw`` env-mat
        # (the gate's per-edge switch) as opaque nodes in the graph .pt2. The
        # env-mat routes to the opaque operator only at ``DP_TRITON_INFER >= 1``
        # (the export condition), so set the level while tracing.
        des = self._strip([8, 16, 32])
        tebd = des.type_embedding.call()
        graph, atype_local = self._graph(des)

        def fn(edge_vec):
            g = dataclasses.replace(graph, edge_vec=edge_vec)
            return des._call_graph_triton(g, atype_local, type_embedding=tebd)

        saved = os.environ.get("DP_TRITON_INFER")
        try:
            os.environ["DP_TRITON_INFER"] = "1"
            traced = make_fx(fn, tracing_mode="real")(graph.edge_vec)
        finally:
            if saved is None:
                os.environ.pop("DP_TRITON_INFER", None)
            else:
                os.environ["DP_TRITON_INFER"] = saved
        targets = [str(n.target) for n in traced.graph.nodes]
        self.assertGreaterEqual(sum("edge_conv" in t for t in targets), 1)
        self.assertGreaterEqual(sum("edge_env_mat" in t for t in targets), 1)

    def test_call_graph_routes_on_level(self) -> None:
        # call_graph bakes edge_conv only when DP_TRITON_INFER >= 1; level 0
        # keeps the dpmodel reference (no operator).
        des = _build_dpa1_expt(self.device, [8, 16, 32])
        tebd = des.type_embedding.call()
        graph, atype_local = self._graph(des)
        saved = os.environ.get("DP_TRITON_INFER")
        try:
            counts = {}
            for level in ("0", "1"):
                os.environ["DP_TRITON_INFER"] = level

                def fn(edge_vec):
                    g = dataclasses.replace(graph, edge_vec=edge_vec)
                    return des.call_graph(g, atype_local, type_embedding=tebd)

                traced = make_fx(fn, tracing_mode="real")(graph.edge_vec)
                counts[level] = sum(
                    "edge_conv" in str(n.target) for n in traced.graph.nodes
                )
        finally:
            if saved is None:
                os.environ.pop("DP_TRITON_INFER", None)
            else:
                os.environ["DP_TRITON_INFER"] = saved
        self.assertEqual(counts["0"], 0)
        self.assertGreaterEqual(counts["1"], 1)

    def test_call_graph_fp32_edge_vec_cast(self) -> None:
        # An fp32 model receives fp64 edge_vec at the graph .pt2 ABI. The public
        # call_graph must cast it to the model precision (no ``double != float``)
        # for both the reference and Triton routes, compute in fp32, and return
        # the force gradient in the fp64 leaf dtype -- with the routes matching.
        des = _build_dpa1_expt(self.device, [8, 16, 32], precision="float32")
        tebd = des.type_embedding.call()
        graph, atype_local = self._graph(des)
        self.assertEqual(graph.edge_vec.dtype, torch.float64)

        def run(level):
            os.environ["DP_TRITON_INFER"] = level
            ev = graph.edge_vec.detach().clone().requires_grad_(True)
            g = dataclasses.replace(graph, edge_vec=ev)
            grrg, _ = des.call_graph(g, atype_local, type_embedding=tebd)
            (gvec,) = torch.autograd.grad(grrg.sum(), ev)
            return grrg.detach(), gvec.detach()

        saved = os.environ.get("DP_TRITON_INFER")
        try:
            d0, g0 = run("0")  # dpmodel reference (edge_vec cast to fp32)
            d1, g1 = run("1")  # fused edge_conv (edge_vec cast to fp32)
        finally:
            if saved is None:
                os.environ.pop("DP_TRITON_INFER", None)
            else:
                os.environ["DP_TRITON_INFER"] = saved
        self.assertEqual(d1.dtype, torch.float32)  # fp32 compute
        self.assertEqual(g1.dtype, torch.float64)  # force in the fp64 leaf dtype
        self.assertTrue(torch.isfinite(g1).all())
        torch.testing.assert_close(d1, d0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(g1, g0, atol=1e-5, rtol=1e-5)


@_GPU
class TestDpa1DenseRouting(unittest.TestCase):
    """Dense-lower parity and make_fx bake of the fused env_mat + se_conv route.

    The dense ``call`` routes ``prod_env_mat`` through the fused
    :func:`~deepmd.kernels.triton.env_mat.env_mat` operator (and the embedding
    through :func:`~deepmd.kernels.triton.dpa1.se_conv.se_conv`) at
    ``DP_TRITON_INFER >= 1``; this checks the operator wiring (parameters, atype
    slicing) reproduces the dpmodel reference and bakes into the traced graph.
    """

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        gen = torch.Generator(device=self.device).manual_seed(3)
        n = 24
        self.coord = (torch.rand(1, n, 3, generator=gen, device=self.device) * 8.0).to(
            torch.float64
        )
        self.atype = torch.randint(0, 2, (1, n), generator=gen, device=self.device)
        self.box = (
            (torch.eye(3, device=self.device) * 8.0).reshape(1, 9).to(torch.float64)
        )

    def _dense_inputs(self, des):
        ec, ea, mp, nl = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            des.get_rcut(),
            des.get_sel(),
            mixed_types=des.mixed_types(),
            box=self.box,
        )
        return ec, ea, nl

    def _assert_parity(self, des) -> None:
        self.assertTrue(des._fused_eligible("triton"))
        ec, ea, nl = self._dense_inputs(des)
        saved = os.environ.get("DP_TRITON_INFER")

        def run(level):
            os.environ["DP_TRITON_INFER"] = level
            c = ec.detach().clone().requires_grad_(True)
            grrg = des.call(c, ea, nl)[0]
            (gc,) = torch.autograd.grad(grrg.sum(), c)
            return grrg.detach(), gc.detach()

        try:
            d0, g0 = run("0")  # dpmodel dense reference
            d1, g1 = run("1")  # fused env_mat + se_conv
        finally:
            if saved is None:
                os.environ.pop("DP_TRITON_INFER", None)
            else:
                os.environ["DP_TRITON_INFER"] = saved
        torch.testing.assert_close(d1, d0, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(g1, g0, atol=1e-6, rtol=1e-6)

    def test_parity_concat_doubling(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [8, 16, 32]))

    def test_parity_concat_identity(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [8, 16, 16]))

    def test_parity_concat_nonpow2(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [12, 25, 50]))

    def test_make_fx_bakes_env_mat(self) -> None:
        des = _build_dpa1_expt(self.device, [8, 16, 32])
        ec, ea, nl = self._dense_inputs(des)
        saved = os.environ.get("DP_TRITON_INFER")
        try:
            os.environ["DP_TRITON_INFER"] = "1"

            def fn(c):
                return des.call(c, ea, nl)[0]

            traced = make_fx(fn, tracing_mode="real")(ec)
        finally:
            if saved is None:
                os.environ.pop("DP_TRITON_INFER", None)
            else:
                os.environ["DP_TRITON_INFER"] = saved
        targets = [str(n.target) for n in traced.graph.nodes]
        self.assertGreaterEqual(sum("env_mat.default" in t for t in targets), 1)


if __name__ == "__main__":
    unittest.main()
