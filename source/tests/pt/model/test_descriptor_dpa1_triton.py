# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the opt-in Triton inference kernel of the DPA1 descriptor
(``se_atten`` with ``attn_layer == 0``, in either ``strip`` or ``concat``
tebd-input mode), enabled via the ``DP_TRITON_INFER`` level (see
:func:`deepmd.kernels.utils.triton_infer_level`).

Three properties are covered:

1. Numerical correctness of the fused environment convolution ``se_conv``
   (forward and the inference force backward) against the eager reference,
   for both gate modes (``gated`` 1 = strip type-pair gate, 0 = concat no gate),
   across every residual structure (doubling / identity / none), both inlined
   activations (``tanh`` / ``silu``), both power-of-two and non-power-of-two
   channel widths (the latter exercising the masked-padding path), and with or
   without a per-layer timestep.
2. ``make_fx`` composability: the operator and its backward each trace as a
   single opaque node, mirroring the ``pt_expt`` inference path that lowers a
   ``make_fx`` graph through ``torch.compile`` / export.
3. Descriptor-level parity: routing ``DescrptDPA1`` through the fused kernel
   reproduces the dense reference descriptor and its coordinate gradient for
   both tebd-input modes across the doubling, identity, ``resnet_dt`` and
   non-power-of-two embedding shapes, and the routed forward remains
   ``torch.jit.script``-able (the LAMMPS freeze path prunes the operator and
   keeps the reference).
"""

import os
import unittest

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.kernels.triton.dpa1.activation import (
    TRITON_AVAILABLE,
)
from deepmd.kernels.triton.dpa1.edge_conv import (
    _edge_conv_reference,
    _edge_conv_reference_backward,
    edge_conv,
)
from deepmd.kernels.triton.dpa1.gemm_fp16x3 import (
    embed_gemm_fp16x3,
)
from deepmd.kernels.triton.dpa1.se_conv import (
    _se_conv_reference,
    se_conv,
)
from deepmd.kernels.triton.dpa1.tile_configs import (
    DEFAULT_CONFIG,
    resolve_conv_config,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA1,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

_CUDA = torch.cuda.is_available()
_GPU = unittest.skipUnless(
    _CUDA and TRITON_AVAILABLE,
    "CUDA and Triton are required for the DPA1 fused kernel",
)


def _rand_conv_inputs(nfnl, nnei, ng, resnet_mult, has_idt, ntype_pair, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    h1_dim = ng // resnet_mult if resnet_mult > 0 else ng
    z2 = torch.randn(nfnl, nnei, ng, device=device, generator=gen)
    h1 = torch.randn(nfnl, nnei, h1_dim, device=device, generator=gen)
    idt = (
        0.1 + torch.rand(ng, device=device, generator=gen)
        if has_idt
        else torch.ones(ng, device=device)
    )
    tt = torch.randn(ntype_pair, ng, device=device, generator=gen) * 0.3
    idx = torch.randint(0, ntype_pair, (nfnl * nnei,), device=device, generator=gen)
    sw = torch.rand(nfnl, nnei, device=device, generator=gen)
    rr = torch.randn(nfnl, nnei, 4, device=device, generator=gen)
    return z2, h1, idt, tt, idx, sw, rr


class TestSeConvConfig(unittest.TestCase):
    """Launch-configuration resolution (CPU-safe, no kernel launch)."""

    def test_level1_returns_default(self) -> None:
        self.assertEqual(resolve_conv_config(128, 64, level=1), DEFAULT_CONFIG)

    def test_level0_returns_default(self) -> None:
        self.assertEqual(resolve_conv_config(128, 64, level=0), DEFAULT_CONFIG)

    def test_level2_unknown_shape_falls_back(self) -> None:
        # An unswept channel width can only fall back to the universal default.
        self.assertEqual(resolve_conv_config(777, 333, level=2), DEFAULT_CONFIG)


@_GPU
class TestSeConvOp(unittest.TestCase):
    """Forward / backward correctness and make_fx composability of ``se_conv``.

    Every residual structure (``resnet_mult`` 2/1/0) is checked both with and
    without a per-layer timestep.
    """

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        # ``ng = 128`` exercises the power-of-two fast path (no padding, and the
        # doubling fold via a reshape); ``ng = 100`` exercises the masked-padding
        # path (padded to 128) and the direct doubling fold at the true ``H1``.
        # ``act`` selects the inlined activation: 0 = tanh, 1 = silu.
        self.cases = [
            (ng, mult, has_idt, act, gated)
            for ng in (128, 100)
            for mult in (2, 1, 0)
            for has_idt in (False, True)
            for act in (0, 1)
            for gated in (1, 0)
        ]

    def test_forward_matches_reference(self) -> None:
        for ng, mult, has_idt, act, gated in self.cases:
            with self.subTest(ng=ng, mult=mult, has_idt=has_idt, act=act, gated=gated):
                z2, h1, idt, tt, idx, sw, rr = _rand_conv_inputs(
                    512, 47, ng, mult, has_idt, 169, self.device
                )
                ref = _se_conv_reference(z2, h1, idt, tt, idx, sw, rr, mult, act, gated)
                got = se_conv(z2, h1, idt, tt, idx, sw, rr, mult, act, gated)
                rel = (got - ref).abs().max() / ref.abs().max()
                self.assertLess(rel.item(), 1e-5)

    def test_backward_matches_reference(self) -> None:
        for ng, mult, has_idt, act, gated in self.cases:
            with self.subTest(ng=ng, mult=mult, has_idt=has_idt, act=act, gated=gated):
                z2, h1, idt, tt, idx, sw, rr = _rand_conv_inputs(
                    512, 47, ng, mult, has_idt, 169, self.device
                )
                gout = torch.randn_like(
                    _se_conv_reference(z2, h1, idt, tt, idx, sw, rr, mult, act, gated)
                )
                ref_in = [
                    t.detach().clone().requires_grad_(True) for t in (z2, h1, sw, rr)
                ]
                _se_conv_reference(
                    ref_in[0],
                    ref_in[1],
                    idt,
                    tt,
                    idx,
                    ref_in[2],
                    ref_in[3],
                    mult,
                    act,
                    gated,
                ).backward(gout)
                got_in = [
                    t.detach().clone().requires_grad_(True) for t in (z2, h1, sw, rr)
                ]
                se_conv(
                    got_in[0],
                    got_in[1],
                    idt,
                    tt,
                    idx,
                    got_in[2],
                    got_in[3],
                    mult,
                    act,
                    gated,
                ).backward(gout)
                for name, a, b in zip(
                    ("z2", "h1", "sw", "rr"),
                    (t.grad for t in ref_in),
                    (t.grad for t in got_in),
                    strict=True,
                ):
                    # h1 carries no gradient without a residual last layer;
                    # sw carries none in concat (no gate). Both are None on the
                    # reference side, and the kernel returns an exact zero.
                    if (name == "h1" and mult == 0) or (name == "sw" and not gated):
                        self.assertLess(b.abs().max().item(), 1e-12)
                        continue
                    rel = (a - b).abs().max() / a.abs().max().clamp_min(1e-20)
                    self.assertLess(rel.item(), 1e-5, msg=f"grad {name}")

    def test_make_fx_force_trace(self) -> None:
        z2, h1, idt, tt, idx, sw, rr = _rand_conv_inputs(
            512, 47, 128, 2, False, 169, self.device
        )
        gout = torch.randn_like(
            _se_conv_reference(z2, h1, idt, tt, idx, sw, rr, 2, 0, 1)
        )

        def force_fn(z2, h1, idt, tt, idx, sw, rr):
            z2r = z2.detach().requires_grad_(True)
            out = se_conv(z2r, h1, idt, tt, idx, sw, rr, 2, 0, 1)
            (grad_z2,) = torch.autograd.grad(out, z2r, gout)
            return out, grad_z2

        traced = make_fx(force_fn)(z2, h1, idt, tt, idx, sw, rr)
        n_fwd = sum(1 for n in traced.graph.nodes if "se_conv.default" in str(n.target))
        n_bwd = sum(1 for n in traced.graph.nodes if "se_conv_bwd" in str(n.target))
        self.assertEqual(n_fwd, 1)
        self.assertEqual(n_bwd, 1)
        out_ref, _ = force_fn(z2, h1, idt, tt, idx, sw, rr)
        out_traced, _ = traced(z2, h1, idt, tt, idx, sw, rr)
        torch.testing.assert_close(out_traced, out_ref)


def _rand_edge_inputs(
    n_edge, n_node, ng, resnet_mult, has_idt, ntype_pair, device, seed=0
):
    gen = torch.Generator(device=device).manual_seed(seed)
    h1_dim = ng // resnet_mult if resnet_mult > 0 else ng
    z2 = torch.randn(n_edge, ng, device=device, generator=gen)
    h1 = torch.randn(n_edge, h1_dim, device=device, generator=gen)
    idt = (
        0.1 + torch.rand(ng, device=device, generator=gen)
        if has_idt
        else torch.ones(ng, device=device)
    )
    tt = torch.randn(ntype_pair, ng, device=device, generator=gen) * 0.3
    idx = torch.randint(0, ntype_pair, (n_edge,), device=device, generator=gen)
    sw = torch.rand(n_edge, device=device, generator=gen)
    rr = torch.randn(n_edge, 4, device=device, generator=gen)
    dst = torch.randint(0, n_node, (n_edge,), device=device, generator=gen)
    edge_mask = (torch.rand(n_edge, device=device, generator=gen) > 0.1).to(
        torch.float32
    )
    return z2, h1, idt, tt, idx, sw, rr, dst, edge_mask


@_GPU
class TestEdgeConvOp(unittest.TestCase):
    """Forward / backward correctness and make_fx composability of the graph
    ``edge_conv`` (node-parallel CSR segment reduction), for both gate modes
    (``gated`` 1 = strip type-pair gate, 0 = concat no gate).
    """

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        self.n_edge, self.n_node = 4096, 256
        # ng = 128 exercises the power-of-two fast path; ng = 100 the masked
        # padding path; act 0 = tanh, 1 = silu; gated 1 = strip, 0 = concat.
        self.cases = [
            (ng, mult, has_idt, act, gated)
            for ng in (128, 100)
            for mult in (2, 1, 0)
            for has_idt in (False, True)
            for act in (0, 1)
            for gated in (1, 0)
        ]

    def test_forward_matches_reference(self) -> None:
        for ng, mult, has_idt, act, gated in self.cases:
            with self.subTest(ng=ng, mult=mult, has_idt=has_idt, act=act, gated=gated):
                z2, h1, idt, tt, idx, sw, rr, dst, em = _rand_edge_inputs(
                    self.n_edge, self.n_node, ng, mult, has_idt, 169, self.device
                )
                args = (
                    z2,
                    h1,
                    idt,
                    tt,
                    idx,
                    sw,
                    rr,
                    dst,
                    em,
                    self.n_node,
                    mult,
                    act,
                    gated,
                )
                ref = _edge_conv_reference(*args)
                got = edge_conv(*args)
                rel = (got - ref).abs().max() / ref.abs().max()
                self.assertLess(rel.item(), 1e-5)

    def test_backward_matches_reference(self) -> None:
        for ng, mult, has_idt, act, gated in self.cases:
            with self.subTest(ng=ng, mult=mult, has_idt=has_idt, act=act, gated=gated):
                z2, h1, idt, tt, idx, sw, rr, dst, em = _rand_edge_inputs(
                    self.n_edge, self.n_node, ng, mult, has_idt, 169, self.device
                )
                tail = (dst, em, self.n_node, mult, act, gated)
                gout = torch.randn_like(
                    _edge_conv_reference(z2, h1, idt, tt, idx, sw, rr, *tail)
                )
                ref_in = [
                    t.detach().clone().requires_grad_(True) for t in (z2, h1, sw, rr)
                ]
                _edge_conv_reference(
                    ref_in[0], ref_in[1], idt, tt, idx, ref_in[2], ref_in[3], *tail
                ).backward(gout)
                got_in = [
                    t.detach().clone().requires_grad_(True) for t in (z2, h1, sw, rr)
                ]
                edge_conv(
                    got_in[0], got_in[1], idt, tt, idx, got_in[2], got_in[3], *tail
                ).backward(gout)
                for name, a, b in zip(
                    ("z2", "h1", "sw", "rr"),
                    (t.grad for t in ref_in),
                    (t.grad for t in got_in),
                    strict=True,
                ):
                    # h1 carries no gradient without a residual last layer; sw
                    # carries none in concat (no gate). Both are None on the
                    # reference side; the kernel returns an exact zero.
                    if (name == "h1" and mult == 0) or (name == "sw" and not gated):
                        self.assertLess(b.abs().max().item(), 1e-12)
                        continue
                    rel = (a - b).abs().max() / a.abs().max().clamp_min(1e-20)
                    self.assertLess(rel.item(), 1e-5, msg=f"grad {name}")

    def test_make_fx_force_trace(self) -> None:
        # The strip gate (gated == 1) exercises the type-pair table gather inside
        # the opaque operator; it must still trace as one fwd + one bwd node.
        z2, h1, idt, tt, idx, sw, rr, dst, em = _rand_edge_inputs(
            2048, 128, 128, 2, False, 169, self.device
        )

        def force_fn(rr):
            rrr = rr.detach().requires_grad_(True)
            out = edge_conv(z2, h1, idt, tt, idx, sw, rrr, dst, em, 128, 2, 0, 1)
            (grad_rr,) = torch.autograd.grad(out.sum(), rrr)
            return out, grad_rr

        traced = make_fx(force_fn)(rr)
        targets = [str(n.target) for n in traced.graph.nodes]
        self.assertEqual(sum("edge_conv.default" in t for t in targets), 1)
        self.assertEqual(sum("edge_conv_bwd" in t for t in targets), 1)


class TestEdgeConvReferenceBackward(unittest.TestCase):
    """Closed-form CPU backward vs autograd through the eager reference.

    ``_edge_conv_reference_backward`` is the operator's non-CUDA gradient (used
    when the pt_expt graph is traced or run off the GPU); it must be a closed
    form so the fallback composes under ``make_fx``. This validates it against
    ``torch.autograd`` through the eager forward across every residual structure
    and both gate modes. CPU-only, so it runs without a GPU.
    """

    def _check(self, ng: int, mult: int, gated: int, act: int) -> None:
        dev = torch.device("cpu")
        n_node = 64
        z2, h1, idt, tt, idx, sw, rr, dst, em = _rand_edge_inputs(
            512, n_node, ng, mult, True, 37, dev
        )
        tail = (dst, em, n_node, mult, act, gated)
        gout = torch.randn_like(
            _edge_conv_reference(z2, h1, idt, tt, idx, sw, rr, *tail)
        )
        leaves = [t.detach().clone().requires_grad_(True) for t in (z2, h1, sw, rr)]
        _edge_conv_reference(
            leaves[0], leaves[1], idt, tt, idx, leaves[2], leaves[3], *tail
        ).backward(gout)
        closed = dict(
            zip(
                ("z2", "h1", "sw", "rr"),
                _edge_conv_reference_backward(
                    gout, z2, h1, idt, tt, idx, sw, rr, *tail
                ),
                strict=True,
            )
        )
        for name, leaf in zip(("z2", "h1", "sw", "rr"), leaves, strict=True):
            c = closed[name]
            if (name == "h1" and mult == 0) or (name == "sw" and not gated):
                self.assertLess(c.abs().max().item(), 1e-12)
                continue
            rel = (leaf.grad - c).abs().max() / leaf.grad.abs().max().clamp_min(1e-20)
            self.assertLess(rel.item(), 1e-6, msg=f"grad {name}")

    def test_closed_form_matches_autograd(self) -> None:
        for mult in (2, 1, 0):
            for gated in (1, 0):
                for act in (0, 1):
                    with self.subTest(mult=mult, gated=gated, act=act):
                        self._check(128, mult, gated, act)


@_GPU
class TestEmbedGemmFp16x3(unittest.TestCase):
    """fp16x3 embedding GEMM: fp32-reference accuracy (forward + input gradient)
    and ``make_fx`` composability. Only the large-M embedding shapes are the
    intended target; the operator stays numerically correct everywhere.
    """

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")

    def test_matches_fp32_reference(self) -> None:
        for m, k, n in [
            (65536, 64, 128),
            (65536, 128, 64),
            (65536, 32, 64),
            (4096, 256, 256),
        ]:
            with self.subTest(m=m, k=k, n=n):
                gen = torch.Generator(device=self.device).manual_seed(0)
                a = (
                    torch.randn(m, k, device=self.device, generator=gen) * 0.3
                ).requires_grad_(True)
                b = torch.randn(k, n, device=self.device, generator=gen) * 0.3
                ref = a.detach().double() @ b.double()
                c = embed_gemm_fp16x3(a, b)
                fwd = (c.double() - ref).abs().max() / ref.abs().max()
                self.assertLess(fwd.item(), 1e-5)
                go = torch.randn_like(c)
                (ga,) = torch.autograd.grad(c, a, go)
                ga_ref = go.double() @ b.double().t()
                bwd = (ga.double() - ga_ref).abs().max() / ga_ref.abs().max()
                self.assertLess(bwd.item(), 1e-5, msg="grad_a")

    def test_make_fx_force_trace(self) -> None:
        a = torch.randn(4096, 64, device=self.device)
        b = torch.randn(64, 128, device=self.device)
        go = torch.randn(4096, 128, device=self.device)

        def force_fn(a):
            ar = a.detach().requires_grad_(True)
            c = embed_gemm_fp16x3(ar, b)
            (grad_a,) = torch.autograd.grad(c.sum(), ar, go.new_ones(()))
            return c, grad_a

        traced = make_fx(force_fn)(a)
        targets = [str(n.target) for n in traced.graph.nodes]
        # forward + backward each trace as an opaque fp16x3 GEMM node.
        self.assertGreaterEqual(sum("embed_gemm_fp16x3" in t for t in targets), 2)


def _build_dpa1(
    device,
    neuron,
    resnet_dt=False,
    activation_function="tanh",
    tebd_input_mode="strip",
):
    des = DescrptDPA1(
        rcut=6.0,
        rcut_smth=2.0,
        sel=20,
        ntypes=2,
        neuron=neuron,
        axis_neuron=4,
        tebd_dim=8,
        tebd_input_mode=tebd_input_mode,
        attn=8,
        attn_layer=0,
        resnet_dt=resnet_dt,
        activation_function=activation_function,
        precision="float32",
        seed=1,
    ).to(device)
    des.eval()
    return des


@_GPU
class TestSeAttenRouting(unittest.TestCase):
    """Descriptor-level parity and TorchScript safety of the fused route."""

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        gen = torch.Generator(device=self.device).manual_seed(3)
        n = 24
        self.coord = (torch.rand(1, n, 3, generator=gen, device=self.device) * 8.0).to(
            env.GLOBAL_PT_FLOAT_PRECISION
        )
        self.atype = torch.randint(0, 2, (1, n), generator=gen, device=self.device)
        self.box = (
            (torch.eye(3, device=self.device) * 8.0)
            .reshape(1, 9)
            .to(env.GLOBAL_PT_FLOAT_PRECISION)
        )

    def _run(self, des, coord):
        ec, ea, mapping, nlist = extend_input_and_build_neighbor_list(
            coord,
            self.atype,
            des.get_rcut(),
            des.get_sel(),
            mixed_types=des.mixed_types(),
            box=self.box,
        )
        return des(ec, ea, nlist, mapping=mapping)

    def _assert_parity(self, des) -> None:
        self.assertTrue(des.se_atten.se_conv_eligible)
        block = des.se_atten
        block.use_triton_infer = False
        c0 = self.coord.detach().clone().requires_grad_(True)
        d0 = self._run(des, c0)[0]
        (g0,) = torch.autograd.grad(d0.sum(), c0)
        block.use_triton_infer = True
        c1 = self.coord.detach().clone().requires_grad_(True)
        d1 = self._run(des, c1)[0]
        (g1,) = torch.autograd.grad(d1.sum(), c1)
        torch.testing.assert_close(d1, d0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(g1, g0, atol=1e-5, rtol=1e-5)

    def test_parity_doubling(self) -> None:
        self._assert_parity(_build_dpa1(self.device, [8, 16, 32]))

    def test_parity_identity(self) -> None:
        self._assert_parity(_build_dpa1(self.device, [8, 16, 16]))

    def test_parity_resnet_dt(self) -> None:
        self._assert_parity(_build_dpa1(self.device, [8, 16, 32], resnet_dt=True))

    def test_parity_nonpow2_doubling(self) -> None:
        # Non-power-of-two widths (ng = 50, H1 = 25) route through the kernel's
        # masked-padding path and the direct (non-reshape) doubling fold.
        self._assert_parity(_build_dpa1(self.device, [12, 25, 50]))

    def test_parity_nonpow2_identity(self) -> None:
        self._assert_parity(_build_dpa1(self.device, [12, 25, 25]))

    def test_parity_silu(self) -> None:
        self._assert_parity(
            _build_dpa1(self.device, [8, 16, 32], activation_function="silu")
        )

    def test_parity_concat_doubling(self) -> None:
        # Concat mode: the type feature enters the embedding input (no gate);
        # ``se_conv`` runs with ``gated == 0``.
        self._assert_parity(
            _build_dpa1(self.device, [8, 16, 32], tebd_input_mode="concat")
        )

    def test_parity_concat_identity(self) -> None:
        self._assert_parity(
            _build_dpa1(self.device, [8, 16, 16], tebd_input_mode="concat")
        )

    def test_parity_concat_nonpow2(self) -> None:
        self._assert_parity(
            _build_dpa1(self.device, [12, 25, 50], tebd_input_mode="concat")
        )

    def test_parity_level3_fp16x3(self) -> None:
        # DP_TRITON_INFER=3 folds the z2 embedding GEMM into fp16x3; the routed
        # descriptor and its coordinate gradient must match the fp32 (level-2)
        # output to ~fp32 rounding (~2^-22 step).
        des = _build_dpa1(self.device, [8, 16, 32], tebd_input_mode="concat")
        des.se_atten.use_triton_infer = True

        def run(level):
            os.environ["DP_TRITON_INFER"] = level
            c = self.coord.detach().clone().requires_grad_(True)
            d = self._run(des, c)[0]
            (g,) = torch.autograd.grad(d.sum(), c)
            return d.detach(), g.detach()

        saved = os.environ.get("DP_TRITON_INFER")
        try:
            d2, g2 = run("2")
            d3, g3 = run("3")
        finally:
            if saved is None:
                os.environ.pop("DP_TRITON_INFER", None)
            else:
                os.environ["DP_TRITON_INFER"] = saved
        torch.testing.assert_close(d3, d2, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(g3, g2, atol=1e-4, rtol=1e-4)

    def test_ineligible_unsupported_activation(self) -> None:
        # The kernel inlines only ``tanh`` / ``silu``; any other last-layer
        # activation must keep the dense reference path.
        des = _build_dpa1(self.device, [8, 16, 32], activation_function="gelu")
        self.assertFalse(des.se_atten.se_conv_eligible)

    def test_torchscript_prunes_fused_branch(self) -> None:
        # ``torch.jit.script`` (the LAMMPS freeze path) must build even with the
        # fused route enabled; the operator lives behind ``is_scripting``.
        des = _build_dpa1(self.device, [8, 16, 32])
        des.se_atten.use_triton_infer = True
        self.assertIsNotNone(torch.jit.script(des))


if __name__ == "__main__":
    unittest.main()
