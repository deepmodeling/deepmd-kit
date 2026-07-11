# SPDX-License-Identifier: LGPL-3.0-or-later
"""Graph-lower CUDA mega-kernel path of the pt_expt DPA1 model.

At ``DP_CUDA_INFER >= 1`` the concat, attention-free graph lower routes
through three fused CUDA operator suites (see ``source/op/pt``):

* ``deepmd::dpa1_graph_descriptor`` -- the descriptor mega kernels
  (:mod:`deepmd.kernels.cuda.dpa1.graph_descriptor`);
* ``deepmd::graph_fitting`` -- the fused energy fitting network
  (:mod:`deepmd.kernels.cuda.graph_fitting`);
* ``deepmd::edge_force_virial`` -- the fused force / virial assembly
  (:mod:`deepmd.kernels.cuda.edge_force_virial`).

Covered properties:

1. Descriptor parity (concat): the fused route reproduces the dpmodel
   reference ``call_graph`` (grrg, rot_mat and the ``edge_vec`` gradient)
   across the supported widths, activations, one/two-side and ``resnet_dt``,
   with an fp64 ``edge_vec`` leaf against an fp32 model (the graph ``.pt2``
   ABI).
2. Descriptor parity (strip): the dpmodel graph reference does not implement
   strip, so values are checked against the dense reference ``call`` on the
   same neighbor list and the ``edge_vec`` gradient against the operator's
   CPU implementation (an independent autograd formulation).
3. Eligibility gating: unsupported configurations fall back, and the traced
   graph contains the operator only at ``DP_CUDA_INFER >= 1``.
4. Fitting parity: values and the descriptor gradient of the fused fitting
   match the dpmodel reference.
5. Scatter parity: force / atom-virial / per-frame virial of the fused
   assembly match the array-API reference on a multi-frame graph, and the
   CPU trace-time implementation matches the CUDA one.
"""

import copy
import dataclasses
import os
import unittest

import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)

_CUDA = torch.cuda.is_available()


def _cuda_ops_loaded() -> bool:
    if not _CUDA:
        return False
    from deepmd.kernels.cuda.dpa1.graph_descriptor import (
        op_available,
    )

    return op_available()


_GPU = unittest.skipUnless(
    _CUDA and _cuda_ops_loaded(),
    "CUDA and the compiled deepmd op library are required",
)


class _CudaLevel:
    """Context manager pinning ``DP_CUDA_INFER`` and restoring it on exit."""

    def __init__(self, level: str) -> None:
        self.level = level
        self.saved: str | None = None

    def __enter__(self) -> None:
        self.saved = os.environ.get("DP_CUDA_INFER")
        os.environ["DP_CUDA_INFER"] = self.level

    def __exit__(self, *exc: object) -> None:
        if self.saved is None:
            os.environ.pop("DP_CUDA_INFER", None)
        else:
            os.environ["DP_CUDA_INFER"] = self.saved


def _build_dpa1_expt(
    device,
    neuron,
    one_side=False,
    act="tanh",
    resnet_dt=False,
    tebd_input_mode="concat",
    smooth=True,
    ntypes=2,
    axis_neuron=4,
    tebd_dim=8,
):
    from deepmd.pt_expt.descriptor.dpa1 import (
        DescrptDPA1,
    )

    des = DescrptDPA1(
        rcut=6.0,
        rcut_smth=2.0,
        sel=40,
        ntypes=ntypes,
        neuron=neuron,
        axis_neuron=axis_neuron,
        tebd_dim=tebd_dim,
        tebd_input_mode=tebd_input_mode,
        attn_layer=0,
        type_one_side=one_side,
        activation_function=act,
        resnet_dt=resnet_dt,
        smooth_type_embedding=smooth,
        precision="float32",
        seed=1,
    ).to(device)
    des.eval()
    return des


@_GPU
class TestDpa1GraphCudaDescriptor(unittest.TestCase):
    """Parity and routing of the fused descriptor mega kernels."""

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        gen = torch.Generator(device=self.device).manual_seed(3)
        n = 48
        self.coord = (torch.rand(1, n, 3, generator=gen, device=self.device) * 9.0).to(
            torch.float64
        )
        self.atype = torch.randint(0, 2, (1, n), generator=gen, device=self.device)
        self.box = (
            (torch.eye(3, device=self.device) * 9.0).reshape(1, 9).to(torch.float64)
        )

    def _graph(self, des):
        graph, atype, _dense = self._graph_and_dense(des)
        return graph, atype

    def _graph_and_dense(self, des):
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
        return graph, self.atype.reshape(-1).to(self.device), (ec, ea, nl)

    def _run(self, des, graph, atype, level):
        """(grrg, rot_mat, d_edge_vec) through the public call_graph route."""
        with _CudaLevel(level):
            ev = graph.edge_vec.detach().clone().requires_grad_(True)
            g = dataclasses.replace(graph, edge_vec=ev)
            tebd = des.type_embedding.call()
            grrg, rot = des.call_graph(g, atype, type_embedding=tebd)
            # A non-uniform cotangent exercises every output channel.
            cot = torch.linspace(0.5, 1.5, grrg.numel(), device=grrg.device).reshape(
                grrg.shape
            )
            (gvec,) = torch.autograd.grad((grrg * cot).sum(), ev)
        return grrg.detach(), rot.detach(), gvec.detach()

    def _assert_parity(self, des) -> None:
        self.assertTrue(des._fused_eligible("cuda"))
        graph, atype = self._graph(des)
        d0, r0, g0 = self._run(des, graph, atype, "0")
        d1, r1, g1 = self._run(des, graph, atype, "1")
        # fp32 compute against the fp32 reference; the gradient returns in the
        # fp64 leaf dtype on both routes.
        self.assertEqual(g1.dtype, torch.float64)
        torch.testing.assert_close(d1, d0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(r1, r0, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(g1, g0.to(g1.dtype), atol=1e-5, rtol=1e-4)

    def test_parity_two_side_tanh(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [32, 64, 128]))

    def test_parity_one_side_silu_resnet_dt(self) -> None:
        self._assert_parity(
            _build_dpa1_expt(
                self.device,
                [32, 64, 128],
                one_side=True,
                act="silu",
                resnet_dt=True,
            )
        )

    def test_parity_narrow_silu(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [16, 32, 64], act="silu"))

    def test_parity_narrow_one_side_resnet_dt(self) -> None:
        self._assert_parity(
            _build_dpa1_expt(self.device, [16, 32, 64], one_side=True, resnet_dt=True)
        )

    def test_parity_width8(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [8, 16, 32]))

    def test_parity_uniform_width(self) -> None:
        # Identity residual shape on both upper layers.
        self._assert_parity(_build_dpa1_expt(self.device, [64, 64, 64]))

    def test_parity_mixed_identity_doubling(self) -> None:
        self._assert_parity(_build_dpa1_expt(self.device, [16, 32, 32], act="silu"))
        self._assert_parity(_build_dpa1_expt(self.device, [32, 32, 64]))

    def test_rotation_output_can_be_suppressed(self) -> None:
        from deepmd.kernels.cuda.dpa1.graph_descriptor import (
            dpa1_graph_descriptor,
        )

        des = _build_dpa1_expt(self.device, [16, 32, 64])
        graph, atype = self._graph(des)
        type_embedding = des.type_embedding.call()
        descriptor, rotation = dpa1_graph_descriptor(
            des,
            graph,
            atype,
            type_embedding,
            write_rotation=False,
        )
        reference, _ = dpa1_graph_descriptor(
            des,
            graph,
            atype,
            type_embedding,
        )
        self.assertEqual(rotation.shape, (0, des.se_atten.neuron[-1], 3))
        torch.testing.assert_close(descriptor, reference)

    def _assert_strip_parity(self, des) -> None:
        """Strip mode: values vs the dense reference ``call`` on the same
        neighbor list; the ``edge_vec`` gradient vs the operator's CPU
        implementation (the dpmodel graph reference does not implement
        strip).
        """
        from deepmd.kernels.cuda.dpa1.graph_descriptor import (
            dpa1_graph_descriptor,
        )

        self.assertTrue(des._fused_eligible("cuda"))
        graph, atype, (ec, ea, nl) = self._graph_and_dense(des)
        nloc = self.atype.shape[1]

        # === Step 1. Values against the dense reference ===
        d_ref = des.call(ec, ea, nl)[0]  # (1, nloc, out), fp64 output
        tebd = des.type_embedding.call()
        grrg, _rot = des._call_graph_cuda(graph, atype, tebd)
        torch.testing.assert_close(
            grrg[:nloc].to(torch.float64), d_ref[0], atol=1e-5, rtol=1e-5
        )

        # === Step 2. edge_vec gradient against the CPU implementation ===
        def run(d, g, at, te):
            ev = g.edge_vec.detach().clone().requires_grad_(True)
            gg = dataclasses.replace(g, edge_vec=ev)
            out, _ = dpa1_graph_descriptor(d, gg, at, te)
            cot = torch.linspace(0.5, 1.5, out.numel(), device=out.device).reshape(
                out.shape
            )
            (gvec,) = torch.autograd.grad((out * cot).sum(), ev)
            return gvec.detach()

        g_cuda = run(des, graph, atype, tebd)
        des_cpu = copy.deepcopy(des).to("cpu")
        graph_cpu = dataclasses.replace(
            graph,
            edge_vec=graph.edge_vec.cpu(),
            edge_index=graph.edge_index.cpu(),
            edge_mask=graph.edge_mask.cpu(),
            destination_order=(
                graph.destination_order.cpu()
                if graph.destination_order is not None
                else None
            ),
            destination_row_ptr=(
                graph.destination_row_ptr.cpu()
                if graph.destination_row_ptr is not None
                else None
            ),
            source_row_ptr=(
                graph.source_row_ptr.cpu() if graph.source_row_ptr is not None else None
            ),
            source_order=(
                graph.source_order.cpu() if graph.source_order is not None else None
            ),
            n_node=graph.n_node.cpu(),
        )
        g_cpu = run(des_cpu, graph_cpu, atype.cpu(), tebd.cpu())
        torch.testing.assert_close(g_cuda.cpu(), g_cpu, atol=1e-6, rtol=1e-4)

    def test_parity_strip_smooth_two_side(self) -> None:
        self._assert_strip_parity(
            _build_dpa1_expt(self.device, [32, 64, 128], tebd_input_mode="strip")
        )

    def test_parity_strip_smooth_one_side_silu(self) -> None:
        self._assert_strip_parity(
            _build_dpa1_expt(
                self.device,
                [16, 32, 64],
                one_side=True,
                act="silu",
                tebd_input_mode="strip",
            )
        )

    def test_parity_strip_nosmooth(self) -> None:
        self._assert_strip_parity(
            _build_dpa1_expt(
                self.device, [16, 32, 64], tebd_input_mode="strip", smooth=False
            )
        )

    def test_eligibility_gate(self) -> None:
        # Non-power-of-two widths, shrinking layers and over-limit widths stay
        # on the reference. Layers two and three support equal-or-doubling
        # stacks; a first-layer residual remains on the reference path.
        self.assertFalse(
            _build_dpa1_expt(self.device, [24, 48, 96])._fused_eligible("cuda")
        )
        self.assertFalse(
            _build_dpa1_expt(self.device, [32, 64, 32])._fused_eligible("cuda")
        )
        self.assertFalse(
            _build_dpa1_expt(self.device, [64, 128, 128])._fused_eligible("cuda")
        )
        self.assertTrue(
            _build_dpa1_expt(self.device, [32, 64, 64])._fused_eligible("cuda")
        )
        self.assertFalse(
            _build_dpa1_expt(
                self.device,
                [8, 16, 32],
                one_side=True,
                tebd_dim=3,
            )._fused_eligible("cuda")
        )
        self.assertFalse(
            _build_dpa1_expt(self.device, [32, 64, 64])
            .to(torch.float64)
            ._fused_eligible("cuda")
        )

    def test_call_graph_routes_on_level(self) -> None:
        des = _build_dpa1_expt(self.device, [32, 64, 128])
        tebd = des.type_embedding.call()
        graph, atype = self._graph(des)
        counts = {}
        for level in ("0", "1"):
            with _CudaLevel(level):

                def fn(edge_vec):
                    g = dataclasses.replace(graph, edge_vec=edge_vec)
                    return des.call_graph(g, atype, type_embedding=tebd)

                traced = make_fx(fn, tracing_mode="real")(graph.edge_vec)
            counts[level] = sum(
                "dpa1_graph_descriptor" in str(n.target) for n in traced.graph.nodes
            )
        self.assertEqual(counts["0"], 0)
        self.assertGreaterEqual(counts["1"], 1)


def _build_compressed_dpa1(
    device,
    neuron,
    one_side=False,
    act="tanh",
    smooth=True,
    min_nbor_dist=0.9,
    ntypes=2,
    axis_neuron=4,
    tebd_dim=8,
):
    """Strip DPA1 with the geometric embedding tabulated (``geo_compress``)."""
    des = _build_dpa1_expt(
        device,
        neuron,
        one_side=one_side,
        act=act,
        resnet_dt=False,
        tebd_input_mode="strip",
        smooth=smooth,
        ntypes=ntypes,
        axis_neuron=axis_neuron,
        tebd_dim=tebd_dim,
    )
    des.enable_compression(min_nbor_dist)
    des.to(device)
    des.eval()
    return des


@_GPU
class TestDpa1GraphCudaCompress(unittest.TestCase):
    """Parity and routing of the fused compressed (tabulated) descriptor kernel."""

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        gen = torch.Generator(device=self.device).manual_seed(3)
        n = 48
        self.coord = (torch.rand(1, n, 3, generator=gen, device=self.device) * 9.0).to(
            torch.float64
        )
        self.atype = torch.randint(0, 2, (1, n), generator=gen, device=self.device)
        self.box = (
            (torch.eye(3, device=self.device) * 9.0).reshape(1, 9).to(torch.float64)
        )

    def _graph_and_dense(self, des):
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
        graph = from_dense_quartet(ec, nl, mp, compact=True, with_csr=True)
        return graph, self.atype.reshape(-1).to(self.device), (ec, ea, nl)

    def _assert_parity(self, des) -> None:
        """Values against the dense compressed reference ``call`` (which runs
        ``tabulate_fusion_se_atten`` on the same table); the ``edge_vec``
        gradient against the operator's CPU implementation (an independent
        autograd formulation through the same quintic table).
        """
        from deepmd.kernels.cuda.dpa1.graph_compress import (
            dpa1_graph_compress,
        )

        self.assertTrue(des._fused_eligible("cuda"))
        graph, atype, (ec, ea, nl) = self._graph_and_dense(des)
        nloc = self.atype.shape[1]

        # === Step 1. Values against the dense tabulated reference ===
        d_ref = des.call(ec, ea, nl)[0]  # (1, nloc, out), fp64 output
        tebd = des.type_embedding.call()
        with _CudaLevel("1"):
            grrg, _rot = des._call_graph_cuda_compress(graph, atype, tebd)
        torch.testing.assert_close(
            grrg[:nloc].to(torch.float64), d_ref[0], atol=1e-5, rtol=1e-5
        )

        # === Step 2. edge_vec gradient against the CPU implementation ===
        def run(d, g, at, te):
            ev = g.edge_vec.detach().clone().requires_grad_(True)
            gg = dataclasses.replace(g, edge_vec=ev)
            out, _ = dpa1_graph_compress(d, gg, at, te)
            cot = torch.linspace(0.5, 1.5, out.numel(), device=out.device).reshape(
                out.shape
            )
            (gvec,) = torch.autograd.grad((out * cot).sum(), ev)
            return gvec.detach()

        g_cuda = run(des, graph, atype, tebd)
        des_cpu = copy.deepcopy(des).to("cpu")
        graph_cpu = dataclasses.replace(
            graph,
            edge_vec=graph.edge_vec.cpu(),
            edge_index=graph.edge_index.cpu(),
            edge_mask=graph.edge_mask.cpu(),
            destination_order=graph.destination_order.cpu(),
            destination_row_ptr=graph.destination_row_ptr.cpu(),
            source_row_ptr=graph.source_row_ptr.cpu(),
            source_order=graph.source_order.cpu(),
            n_node=graph.n_node.cpu(),
        )
        g_cpu = run(des_cpu, graph_cpu, atype.cpu(), tebd.cpu())
        torch.testing.assert_close(g_cuda.cpu(), g_cpu, atol=1e-6, rtol=1e-4)

    def test_parity_two_side_smooth(self) -> None:
        # NG = 64: eight warps active in the moment backward.
        self._assert_parity(_build_compressed_dpa1(self.device, [16, 32, 64]))

    def test_parity_wide_two_side(self) -> None:
        # NG = 128: the moment backward covers the table in two channel blocks.
        self._assert_parity(_build_compressed_dpa1(self.device, [32, 64, 128]))

    def test_parity_narrow_ng8(self) -> None:
        # NG = 8: a single warp is active -- the shuffle fold must stay whole.
        self._assert_parity(_build_compressed_dpa1(self.device, [16, 16, 8]))

    def test_parity_non_power_of_two_ng(self) -> None:
        # NG = 100 (a non-power-of-two width): the kernel runs at the padded
        # width 128 and the padding channels are sliced off.
        self._assert_parity(_build_compressed_dpa1(self.device, [25, 50, 100]))

    def test_parity_one_side_silu(self) -> None:
        self._assert_parity(
            _build_compressed_dpa1(self.device, [16, 32, 64], one_side=True, act="silu")
        )

    def test_parity_nosmooth(self) -> None:
        self._assert_parity(
            _build_compressed_dpa1(self.device, [16, 32, 64], smooth=False)
        )

    def test_parity_axis16(self) -> None:
        """Axis width 16 exercises the full symmetric Gram gradient."""
        self._assert_parity(
            _build_compressed_dpa1(self.device, [16, 32, 64], axis_neuron=16)
        )

    def test_concat_type_embedding_wider_than_warp(self) -> None:
        """The output tail covers every type-embedding channel."""
        self._assert_parity(
            _build_compressed_dpa1(self.device, [16, 32, 64], tebd_dim=64)
        )

    def test_parity_four_types(self) -> None:
        """Two-sided pair indexing covers a representative multi-element model."""
        self.atype = (
            torch.arange(self.atype.numel(), device=self.device).reshape(
                self.atype.shape
            )
            % 4
        )
        self._assert_parity(_build_compressed_dpa1(self.device, [16, 32, 64], ntypes=4))

    def test_int32_edge_index_parity(self) -> None:
        """Int32 edge addressing matches the default int64 graph ABI."""
        from deepmd.kernels.cuda.dpa1.graph_compress import (
            dpa1_graph_compress,
        )

        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        graph64, atype, _ = self._graph_and_dense(des)
        graph32 = dataclasses.replace(
            graph64,
            edge_index=graph64.edge_index.to(torch.int32),
            destination_order=graph64.destination_order.to(torch.int32),
            source_order=graph64.source_order.to(torch.int32),
        )
        type_embedding = des.type_embedding.call()

        def run(graph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
            graph = dataclasses.replace(graph, edge_vec=edge_vec)
            descriptor, rotation = dpa1_graph_compress(
                des, graph, atype, type_embedding
            )
            cotangent = torch.linspace(
                0.5, 1.5, descriptor.numel(), device=descriptor.device
            ).reshape(descriptor.shape)
            (gradient,) = torch.autograd.grad((descriptor * cotangent).sum(), edge_vec)
            return descriptor.detach(), rotation.detach(), gradient.detach()

        outputs64 = run(graph64)
        outputs32 = run(graph32)
        for output32, output64 in zip(outputs32, outputs64, strict=True):
            torch.testing.assert_close(output32, output64, atol=1e-6, rtol=1e-6)

    def test_compact_canonical_descriptor_parity(self) -> None:
        """Source-only topology matches the generic canonical operator."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            canonicalize_neighbor_graph,
        )
        from deepmd.kernels.cuda.dpa1.canonical import (
            ensure_registered,
        )
        from deepmd.kernels.cuda.dpa1.graph_compress import (
            dpa1_graph_compress,
        )
        from deepmd.pt_expt.utils.canonical_graph import (
            canonical_graph_from_neighbor_graph,
        )

        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        graph, atype, _ = self._graph_and_dense(des)
        graph = canonicalize_neighbor_graph(
            dataclasses.replace(graph, n_local=graph.n_node),
            atype.shape[0],
        )
        compact = canonical_graph_from_neighbor_graph(graph)
        type_embedding = des.type_embedding.call()
        se = des.se_atten
        inverse_stddev = torch.reciprocal(se.stddev[:, 0, :]).contiguous()
        lower, upper, table_max, stride0, stride1 = (
            float(value) for value in des.compress_info[0].tolist()[:5]
        )

        generic_descriptor, _ = dpa1_graph_compress(
            des,
            graph,
            atype,
            type_embedding,
        )
        ensure_registered()
        descriptor, _rotation, moment = torch.ops.deepmd.dpa1_canonical_compress(
            compact.edge_vec,
            compact.source,
            compact.destination_row_ptr,
            atype,
            type_embedding,
            se.mean[:, 0, :].contiguous(),
            inverse_stddev,
            des.compress_data[0].contiguous(),
            des.type_embd_data.contiguous(),
            int(se.type_one_side),
            int(des.concat_output_tebd),
            0,
            int(se.smooth),
            int(se.axis_neuron),
            lower,
            upper,
            table_max,
            stride0,
            stride1,
            float(se.rcut),
            float(se.rcut_smth),
            float(se.env_protection),
            float(se.nnei),
        )
        torch.testing.assert_close(descriptor, generic_descriptor)

        cotangent = torch.linspace(
            0.5,
            1.5,
            descriptor.numel(),
            dtype=descriptor.dtype,
            device=descriptor.device,
        ).reshape(descriptor.shape)
        generic_edge_vec = graph.edge_vec.detach().requires_grad_(True)
        generic_graph = dataclasses.replace(graph, edge_vec=generic_edge_vec)
        generic_value, _ = dpa1_graph_compress(
            des,
            generic_graph,
            atype,
            type_embedding,
        )
        (generic_gradient,) = torch.autograd.grad(
            (generic_value * cotangent).sum(),
            generic_edge_vec,
        )
        compact_gradient = torch.ops.deepmd.dpa1_canonical_compress_backward(
            cotangent,
            None,
            moment,
            compact.edge_vec,
            compact.source,
            compact.destination_row_ptr,
            atype,
            se.mean[:, 0, :].contiguous(),
            inverse_stddev,
            des.compress_data[0].contiguous(),
            des.type_embd_data.contiguous(),
            int(se.type_one_side),
            int(se.smooth),
            int(se.axis_neuron),
            lower,
            upper,
            table_max,
            stride0,
            stride1,
            float(se.rcut),
            float(se.rcut_smth),
            float(se.env_protection),
            float(se.nnei),
        )
        physical_edge_count = int(compact.destination_row_ptr[-1].item())
        torch.testing.assert_close(
            compact_gradient[:physical_edge_count],
            generic_gradient[:physical_edge_count].to(compact_gradient.dtype),
            atol=1e-6,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            compact_gradient[physical_edge_count:],
            torch.zeros_like(compact_gradient[physical_edge_count:]),
        )

    def test_adaptive_resource_selection_large_graph(self) -> None:
        """First-use tuning preserves the reference on a non-trivial graph."""
        generator = torch.Generator(device=self.device).manual_seed(37)
        atom_count = 192
        box_length = 15.0
        self.coord = (
            torch.rand(
                1,
                atom_count,
                3,
                generator=generator,
                device=self.device,
            )
            * box_length
        ).to(torch.float64)
        self.atype = torch.randint(
            0,
            2,
            (1, atom_count),
            generator=generator,
            device=self.device,
        )
        self.box = (
            (torch.eye(3, device=self.device) * box_length)
            .reshape(1, 9)
            .to(torch.float64)
        )
        self._assert_parity(_build_compressed_dpa1(self.device, [16, 32, 64]))

    def test_cached_topology_cutoff_mask_parity(self) -> None:
        """Masked skin edges inside cached CSR rows contribute exactly zero."""
        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        graph, atype, _ = self._graph_and_dense(des)
        edge_mask = graph.edge_mask.clone()
        self.assertTrue(bool(edge_mask[0]))
        edge_mask[0] = False
        graph = dataclasses.replace(graph, edge_mask=edge_mask)
        type_embedding = des.type_embedding.call()

        def run(level: str) -> tuple[torch.Tensor, torch.Tensor]:
            edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
            masked_graph = dataclasses.replace(graph, edge_vec=edge_vec)
            with _CudaLevel(level):
                descriptor, _ = des.call_graph(
                    masked_graph, atype, type_embedding=type_embedding
                )
            cotangent = torch.linspace(
                0.5, 1.5, descriptor.numel(), device=descriptor.device
            ).reshape(descriptor.shape)
            (gradient,) = torch.autograd.grad((descriptor * cotangent).sum(), edge_vec)
            return descriptor.detach(), gradient.detach()

        descriptor_ref, gradient_ref = run("0")
        descriptor_fused, gradient_fused = run("1")
        torch.testing.assert_close(
            descriptor_fused, descriptor_ref, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(gradient_fused, gradient_ref, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(
            gradient_fused[0], torch.zeros_like(gradient_fused[0])
        )

    def test_unsorted_edge_stream_uses_permutation_csr(self) -> None:
        """The generic level-1 path preserves arbitrary edge payload order."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_edge_csr,
        )

        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        graph, atype, _ = self._graph_and_dense(des)
        generator = torch.Generator(device=self.device).manual_seed(19)
        permutation = torch.randperm(
            graph.edge_index.shape[1], generator=generator, device=self.device
        )
        (
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_row_ptr,
            source_order,
        ) = build_edge_csr(
            graph.edge_index[:, permutation],
            graph.edge_vec[permutation],
            graph.edge_mask[permutation],
            atype.shape[0],
        )
        graph = dataclasses.replace(
            graph,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            destination_order=destination_order,
            destination_row_ptr=destination_row_ptr,
            source_row_ptr=source_row_ptr,
            source_order=source_order,
            destination_sorted=False,
        )
        type_embedding = des.type_embedding.call()

        def run(level: str) -> tuple[torch.Tensor, torch.Tensor]:
            current_edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
            current_graph = dataclasses.replace(graph, edge_vec=current_edge_vec)
            with _CudaLevel(level):
                descriptor, _ = des.call_graph(
                    current_graph, atype, type_embedding=type_embedding
                )
            cotangent = torch.linspace(
                0.5, 1.5, descriptor.numel(), device=descriptor.device
            ).reshape(descriptor.shape)
            (gradient,) = torch.autograd.grad(
                (descriptor * cotangent).sum(), current_edge_vec
            )
            return descriptor.detach(), gradient.detach()

        descriptor_ref, gradient_ref = run("0")
        descriptor_fused, gradient_fused = run("1")
        torch.testing.assert_close(
            descriptor_fused, descriptor_ref, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(gradient_fused, gradient_ref, atol=1e-5, rtol=1e-4)

    def test_level0_reference_parity(self) -> None:
        """Level-0 graph path (original fused table op on the edge stream)
        matches the dense compressed reference ``call``.
        """
        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        graph, atype, (ec, ea, nl) = self._graph_and_dense(des)
        nloc = self.atype.shape[1]
        d_ref = des.call(ec, ea, nl)[0]
        tebd = des.type_embedding.call()
        with _CudaLevel("0"):
            grrg, _rot = des.call_graph(graph, atype, type_embedding=tebd)
        torch.testing.assert_close(
            grrg[:nloc].to(torch.float64), d_ref[0], atol=1e-5, rtol=1e-5
        )

    def test_call_graph_routes_compress(self) -> None:
        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        tebd = des.type_embedding.call()
        graph, atype, _ = self._graph_and_dense(des)
        counts = {}
        for level in ("0", "1"):
            with _CudaLevel(level):

                def fn(edge_vec):
                    g = dataclasses.replace(graph, edge_vec=edge_vec)
                    return des.call_graph(g, atype, type_embedding=tebd)

                traced = make_fx(fn, tracing_mode="real")(graph.edge_vec)
            counts[level] = sum(
                "dpa1_graph_compress" in str(n.target) for n in traced.graph.nodes
            )
        self.assertEqual(counts["0"], 0)
        self.assertGreaterEqual(counts["1"], 1)

    def test_missing_csr_falls_back(self) -> None:
        """A graph without optional CSR views remains a valid input."""
        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        type_embedding = des.type_embedding.call()
        graph, atype, _ = self._graph_and_dense(des)
        graph = dataclasses.replace(
            graph,
            destination_order=None,
            destination_row_ptr=None,
            source_row_ptr=None,
            source_order=None,
        )
        with _CudaLevel("0"):
            descriptor_ref, rotation_ref = des.call_graph(
                graph, atype, type_embedding=type_embedding
            )
        with _CudaLevel("1"):
            descriptor, rotation = des.call_graph(
                graph, atype, type_embedding=type_embedding
            )
        torch.testing.assert_close(descriptor, descriptor_ref)
        torch.testing.assert_close(rotation, rotation_ref)

    def test_empty_node_graph_has_zero_edge_gradient(self) -> None:
        """An empty graph does not launch a zero-sized CUDA grid."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            NeighborGraph,
        )

        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        edge_vec = torch.zeros(
            2,
            3,
            dtype=torch.float64,
            device=self.device,
            requires_grad=True,
        )
        order = torch.arange(2, dtype=torch.int64, device=self.device)
        graph = NeighborGraph(
            n_node=torch.zeros(1, dtype=torch.int64, device=self.device),
            edge_index=torch.zeros(2, 2, dtype=torch.int64, device=self.device),
            edge_vec=edge_vec,
            edge_mask=torch.zeros(2, dtype=torch.bool, device=self.device),
            destination_order=order,
            destination_row_ptr=torch.zeros(1, dtype=torch.int64, device=self.device),
            source_row_ptr=torch.zeros(1, dtype=torch.int64, device=self.device),
            source_order=order,
            destination_sorted=True,
        )
        atype = torch.empty(0, dtype=torch.int64, device=self.device)

        with _CudaLevel("1"):
            descriptor, rotation = des.call_graph(
                graph,
                atype,
                type_embedding=des.type_embedding.call(),
            )
        self.assertEqual(descriptor.shape, (0, des.get_dim_out()))
        self.assertEqual(rotation.shape, (0, des.se_atten.neuron[-1], 3))
        (gradient,) = torch.autograd.grad(descriptor.sum(), edge_vec)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_empty_node_cpu_operator_has_zero_edge_gradient(self) -> None:
        """The registered CPU implementation preserves the empty-graph contract."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            NeighborGraph,
        )

        device = torch.device("cpu")
        des = _build_compressed_dpa1(device, [16, 32, 64])
        edge_vec = torch.zeros(2, 3, dtype=torch.float64, requires_grad=True)
        order = torch.arange(2, dtype=torch.int64)
        graph = NeighborGraph(
            n_node=torch.zeros(1, dtype=torch.int64),
            edge_index=torch.zeros(2, 2, dtype=torch.int64),
            edge_vec=edge_vec,
            edge_mask=torch.zeros(2, dtype=torch.bool),
            destination_order=order,
            destination_row_ptr=torch.zeros(1, dtype=torch.int64),
            source_row_ptr=torch.zeros(1, dtype=torch.int64),
            source_order=order,
            destination_sorted=True,
        )
        atype = torch.empty(0, dtype=torch.int64)

        with _CudaLevel("1"):
            descriptor, rotation = des.call_graph(
                graph,
                atype,
                type_embedding=des.type_embedding.call(),
            )
        self.assertEqual(descriptor.shape, (0, des.get_dim_out()))
        self.assertEqual(rotation.shape, (0, des.se_atten.neuron[-1], 3))
        (gradient,) = torch.autograd.grad(descriptor.sum(), edge_vec)
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))

    def test_torch_compile_first_backward(self) -> None:
        """Inductor preserves the custom forward and its registered backward."""
        des = _build_compressed_dpa1(self.device, [16, 32, 64])
        type_embedding = des.type_embedding.call()
        graph, atype, _ = self._graph_and_dense(des)

        def objective(edge_vec: torch.Tensor) -> torch.Tensor:
            current_graph = dataclasses.replace(graph, edge_vec=edge_vec)
            descriptor, _ = des.call_graph(
                current_graph, atype, type_embedding=type_embedding
            )
            return descriptor.square().sum()

        with _CudaLevel("1"):
            traced = make_fx(objective, tracing_mode="real")(graph.edge_vec)
        compiled = torch.compile(traced, fullgraph=True, dynamic=True)

        reference_input = graph.edge_vec.detach().clone().requires_grad_(True)
        (reference_gradient,) = torch.autograd.grad(
            traced(reference_input), reference_input
        )
        compiled_input = graph.edge_vec.detach().clone().requires_grad_(True)
        (compiled_gradient,) = torch.autograd.grad(
            compiled(compiled_input), compiled_input
        )
        torch.testing.assert_close(
            compiled_gradient, reference_gradient, atol=1e-6, rtol=1e-5
        )


@_GPU
class TestDpa1GraphCudaFitting(unittest.TestCase):
    """Parity of the fused energy fitting network."""

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")

    def _build(
        self,
        resnet_dt,
        act="tanh",
        dim_descrpt=64,
        neuron=None,
        precision="float32",
    ):
        from deepmd.pt_expt.fitting.ener_fitting import (
            EnergyFittingNet,
        )

        if neuron is None:
            neuron = [48, 48, 48]
        fit = EnergyFittingNet(
            ntypes=2,
            dim_descrpt=dim_descrpt,
            neuron=neuron,
            resnet_dt=resnet_dt,
            activation_function=act,
            precision=precision,
            mixed_types=True,
            seed=2,
        ).to(self.device)
        fit.eval()
        # A non-trivial per-type energy bias exercises the head epilogue.
        fit.bias_atom_e = torch.tensor(
            [[1.5], [-2.5]], dtype=torch.float64, device=self.device
        )
        return fit

    def _assert_parity(self, fit) -> None:
        from deepmd.kernels.cuda.graph_fitting import (
            fitting_eligible,
        )

        self.assertTrue(fitting_eligible(fit))
        gen = torch.Generator(device=self.device).manual_seed(5)
        n = 37
        desc = torch.randn(
            n, 64, generator=gen, device=self.device, dtype=torch.float32
        )
        atype = torch.randint(0, 2, (n,), generator=gen, device=self.device)

        def run(level):
            with _CudaLevel(level):
                x = desc.detach().clone().requires_grad_(True)
                e = fit.call_graph(x, atype)["energy"]
                (dx,) = torch.autograd.grad(e.sum(), x)
            return e.detach(), dx.detach()

        e0, dx0 = run("0")
        e1, dx1 = run("1")
        self.assertEqual(e1.dtype, torch.float64)
        torch.testing.assert_close(e1, e0, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(dx1, dx0, atol=1e-5, rtol=1e-5)

    def test_parity_plain(self) -> None:
        self._assert_parity(self._build(resnet_dt=False))

    def test_parity_resnet_dt_silu(self) -> None:
        self._assert_parity(self._build(resnet_dt=True, act="silu"))

    def test_fparam_falls_back(self) -> None:
        from deepmd.kernels.cuda.graph_fitting import (
            fitting_eligible,
        )
        from deepmd.pt_expt.fitting.ener_fitting import (
            EnergyFittingNet,
        )

        fit = EnergyFittingNet(
            ntypes=2,
            dim_descrpt=64,
            neuron=[48, 48],
            numb_fparam=2,
            precision="float32",
            mixed_types=True,
            seed=2,
        ).to(self.device)
        self.assertFalse(fitting_eligible(fit))

    def test_width_doubling_residual_falls_back(self) -> None:
        from deepmd.kernels.cuda.graph_fitting import (
            fitting_eligible,
        )

        fit = self._build(
            resnet_dt=False,
            dim_descrpt=32,
            neuron=[64, 64],
        )
        self.assertTrue(fit.nets[0].layers[0].resnet)
        self.assertFalse(fitting_eligible(fit))

    def test_float64_parameters_fall_back(self) -> None:
        from deepmd.kernels.cuda.graph_fitting import (
            fitting_eligible,
        )

        self.assertFalse(
            fitting_eligible(self._build(resnet_dt=False, precision="float64"))
        )


@_GPU
class TestDpa1GraphEnergyForce(unittest.TestCase):
    """Parity of the fused end-to-end energy-force operator (DP_CUDA_INFER=2).

    The inference-only descriptor, fitting, descriptor-backward, and CSR scatter
    sequence must reproduce the level-1 result with force and virial obtained
    through ``autograd.grad``.
    """

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        gen = torch.Generator(device=self.device).manual_seed(3)
        n = 48
        self.coord = (torch.rand(1, n, 3, generator=gen, device=self.device) * 9.0).to(
            torch.float64
        )
        self.atype = torch.randint(0, 2, (1, n), generator=gen, device=self.device)
        self.box = (
            (torch.eye(3, device=self.device) * 9.0).reshape(1, 9).to(torch.float64)
        )

    def _build_fitting(self, dim_descrpt):
        from deepmd.pt_expt.fitting.ener_fitting import (
            EnergyFittingNet,
        )

        fit = EnergyFittingNet(
            ntypes=2,
            dim_descrpt=dim_descrpt,
            neuron=[48, 48],
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=2,
        ).to(self.device)
        fit.eval()
        fit.bias_atom_e = torch.tensor(
            [[1.5], [-2.5]], dtype=torch.float64, device=self.device
        )
        return fit

    def _graph(self, des):
        from deepmd.dpmodel.utils.neighbor_graph import (
            from_dense_quartet,
        )

        ec, _ea, mp, nl = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            des.get_rcut(),
            des.get_sel(),
            mixed_types=des.mixed_types(),
            box=self.box,
        )
        graph = from_dense_quartet(ec, nl, mp, compact=True, canonicalize=True)
        return graph, self.atype.reshape(-1).to(self.device)

    def test_parity_vs_separate_ops(self) -> None:
        from deepmd.kernels.cuda.dpa1.graph_energy_force import (
            dpa1_graph_energy_force,
        )
        from deepmd.kernels.cuda.edge_force_virial import (
            edge_force_virial,
        )

        # A doubling stack exercises the retiled backward inside the fusion.
        des = _build_dpa1_expt(self.device, [8, 16, 32], act="silu")
        fit = self._build_fitting(des.get_dim_out())
        graph, atype = self._graph(des)
        tebd = des.type_embedding.call()
        n = atype.shape[0]

        with _CudaLevel("2"):
            energy, atom_e, force, virial, atom_vir = dpa1_graph_energy_force(
                des,
                fit,
                graph,
                atype,
                tebd,
                torch.ones(n, dtype=torch.bool, device=self.device),
                fit.bias_atom_e[:, 0].contiguous(),
                node_capacity=n,
                do_atomic_virial=True,
            )

        # Reference: the level-1 separate operators with the force from autograd.
        with _CudaLevel("1"):
            ev = graph.edge_vec.detach().clone().requires_grad_(True)
            g2 = dataclasses.replace(graph, edge_vec=ev)
            grrg, _rot = des.call_graph(g2, atype, tebd)
            e_atom = fit.call_graph(grrg, atype)[fit.var_name]
            (g_e,) = torch.autograd.grad(e_atom.sum(), ev)
            # The fused operator assembles force / virial in the model compute
            # precision (fp32), so mirror that dtype in the reference scatter.
            r_force, r_atom_vir, r_virial = edge_force_virial(
                g_e.to(force.dtype),
                ev.detach().to(force.dtype),
                graph.edge_index,
                graph.edge_mask,
                graph.destination_order,
                graph.destination_row_ptr,
                graph.source_row_ptr,
                graph.source_order,
                graph.n_node,
                n,
                True,
            )

        self.assertEqual(energy.dtype, torch.float64)
        torch.testing.assert_close(
            atom_e, e_atom.to(torch.float64), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            energy[0, 0], e_atom.sum().to(torch.float64), atol=1e-5, rtol=1e-6
        )
        torch.testing.assert_close(force, r_force, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(virial, r_virial, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(atom_vir, r_atom_vir, atol=1e-4, rtol=1e-4)

    def test_missing_csr_declines_energy_force_fusion(self) -> None:
        """The caller can fall back when optional CSR views are absent."""
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                des = (
                    _build_compressed_dpa1(self.device, [16, 32, 64])
                    if compressed
                    else _build_dpa1_expt(self.device, [8, 16, 32], act="silu")
                )
                fit = self._build_fitting(des.get_dim_out())
                graph, atype = self._graph(des)
                graph = dataclasses.replace(
                    graph,
                    destination_order=None,
                    destination_row_ptr=None,
                    source_row_ptr=None,
                    source_order=None,
                )
                result = des.fused_energy_force_graph(
                    fit,
                    graph,
                    atype,
                    torch.ones(atype.shape[0], dtype=torch.bool, device=self.device),
                    fit.bias_atom_e[:, 0].contiguous(),
                    do_atomic_virial=True,
                )
                self.assertIsNone(result)

    def test_level2_reuses_virtual_and_pair_exclusion_masks(self) -> None:
        from deepmd.pt_expt.model import (
            EnergyModel,
        )

        descriptor = _build_dpa1_expt(self.device, [8, 16, 32], act="silu")
        fitting = self._build_fitting(descriptor.get_dim_out())
        model = EnergyModel(
            descriptor,
            fitting,
            type_map=["A", "B"],
            atom_exclude_types=[1],
            pair_exclude_types=[(0, 1)],
        ).to(self.device)
        model.eval()
        graph, atype = self._graph(descriptor)
        atype = atype.clone()
        atype[0] = -1
        args = (
            atype,
            graph.n_node,
            graph.n_node,
            graph.edge_index,
            graph.edge_vec,
            graph.edge_mask,
            graph.destination_order,
            graph.destination_row_ptr,
            graph.source_row_ptr,
            graph.source_order,
        )

        with _CudaLevel("1"):
            reference = model.forward_common_lower_graph(
                *args,
                do_atomic_virial=True,
                destination_sorted=graph.destination_sorted,
            )
        with _CudaLevel("2"):
            actual = model.forward_common_lower_graph(
                *args,
                do_atomic_virial=True,
                destination_sorted=graph.destination_sorted,
            )

        self.assertEqual(set(actual), set(reference))
        for key in actual:
            torch.testing.assert_close(
                actual[key],
                reference[key],
                atol=1e-4,
                rtol=1e-4,
            )

    def test_fused_energy_uses_owned_nodes_only(self) -> None:
        from deepmd.kernels.cuda.edge_force_virial import (
            edge_force_virial,
        )

        des = _build_dpa1_expt(self.device, [8, 16, 32], act="silu")
        fit = self._build_fitting(des.get_dim_out())
        graph, atype = self._graph(des)
        n_node = atype.shape[0]
        ownership = torch.arange(n_node, device=self.device) < n_node // 2
        output_bias = torch.tensor([2.0, -3.0], dtype=torch.float64, device=self.device)

        with _CudaLevel("2"):
            fused = des.fused_energy_force_graph(
                fit,
                graph,
                atype,
                ownership,
                fit.bias_atom_e[:, 0] + output_bias,
                do_atomic_virial=True,
            )
        assert fused is not None
        energy, atom_energy, force, virial, atom_virial = fused

        with _CudaLevel("1"):
            edge_vec = graph.edge_vec.detach().clone().requires_grad_(True)
            current_graph = dataclasses.replace(graph, edge_vec=edge_vec)
            descriptor, _ = des.call_graph(
                current_graph,
                atype,
                type_embedding=des.type_embedding.call(),
            )
            atom_energy_raw = fit.call_graph(descriptor, atype)[fit.var_name]
            atom_energy_ref = (atom_energy_raw + output_bias[atype, None]) * ownership[
                :, None
            ]
            (edge_gradient,) = torch.autograd.grad(atom_energy_ref.sum(), edge_vec)
            force_ref, atom_virial_ref, virial_ref = edge_force_virial(
                edge_gradient.to(force.dtype),
                edge_vec.detach().to(force.dtype),
                graph.edge_index,
                graph.edge_mask,
                graph.destination_order,
                graph.destination_row_ptr,
                graph.source_row_ptr,
                graph.source_order,
                graph.n_node,
                n_node,
                True,
            )

        torch.testing.assert_close(atom_energy, atom_energy_ref)
        torch.testing.assert_close(energy[0], atom_energy_ref.sum(dim=0))
        torch.testing.assert_close(force, force_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(virial, virial_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            atom_virial,
            atom_virial_ref,
            atol=1e-4,
            rtol=1e-4,
        )


@_GPU
class TestDpa1GraphCompressEnergyForce(unittest.TestCase):
    """Parity of the fused compressed end-to-end energy-force operator.

    The tabulated descriptor, fitting, descriptor-backward, and CSR scatter
    sequence must reproduce the level-1 result with force and virial obtained
    through ``autograd.grad``.
    """

    def setUp(self) -> None:
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda")
        gen = torch.Generator(device=self.device).manual_seed(3)
        n = 48
        self.coord = (torch.rand(1, n, 3, generator=gen, device=self.device) * 9.0).to(
            torch.float64
        )
        self.atype = torch.randint(0, 2, (1, n), generator=gen, device=self.device)
        self.box = (
            (torch.eye(3, device=self.device) * 9.0).reshape(1, 9).to(torch.float64)
        )

    def _build_fitting(self, dim_descrpt, ntypes=2):
        from deepmd.pt_expt.fitting.ener_fitting import (
            EnergyFittingNet,
        )

        fit = EnergyFittingNet(
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            neuron=[64, 64, 64],
            resnet_dt=True,
            activation_function="silu",
            precision="float32",
            mixed_types=True,
            seed=2,
        ).to(self.device)
        fit.eval()
        fit.bias_atom_e = torch.linspace(
            -2.5, 1.5, ntypes, dtype=torch.float64, device=self.device
        ).reshape(ntypes, 1)
        return fit

    def _graph(self, des):
        from deepmd.dpmodel.utils.neighbor_graph import (
            from_dense_quartet,
        )

        ec, _ea, mp, nl = extend_input_and_build_neighbor_list(
            self.coord,
            self.atype,
            des.get_rcut(),
            des.get_sel(),
            mixed_types=des.mixed_types(),
            box=self.box,
        )
        graph = from_dense_quartet(ec, nl, mp, compact=True, canonicalize=True)
        return graph, self.atype.reshape(-1).to(self.device)

    def test_parity_vs_separate_ops(self) -> None:
        from deepmd.kernels.cuda.dpa1.graph_compress import (
            dpa1_graph_compress,
            dpa1_graph_compress_energy_force,
            mega_eligible,
        )
        from deepmd.kernels.cuda.edge_force_virial import (
            edge_force_virial,
        )

        self.atype = (
            torch.arange(self.atype.numel(), device=self.device).reshape(
                self.atype.shape
            )
            % 4
        )
        des = _build_compressed_dpa1(
            self.device,
            [16, 32, 64],
            act="silu",
            ntypes=4,
            axis_neuron=16,
        )
        self.assertTrue(mega_eligible(des))
        fit = self._build_fitting(des.get_dim_out(), ntypes=4)
        graph, atype = self._graph(des)
        edge_mask = graph.edge_mask.clone()
        edge_mask[0] = False
        graph = dataclasses.replace(graph, edge_mask=edge_mask)
        tebd = des.type_embedding.call()
        n = atype.shape[0]

        with _CudaLevel("2"):
            energy, atom_e, force, virial, atom_vir = dpa1_graph_compress_energy_force(
                des,
                fit,
                graph,
                atype,
                tebd,
                torch.ones(n, dtype=torch.bool, device=self.device),
                fit.bias_atom_e[:, 0].contiguous(),
                node_capacity=n,
                do_atomic_virial=True,
            )

        # Reference: the level-1 tabulated operator with the force from autograd.
        with _CudaLevel("1"):
            ev = graph.edge_vec.detach().clone().requires_grad_(True)
            g2 = dataclasses.replace(graph, edge_vec=ev)
            grrg, _rot = dpa1_graph_compress(des, g2, atype, tebd)
            e_atom = fit.call_graph(grrg, atype)[fit.var_name]
            (g_e,) = torch.autograd.grad(e_atom.sum(), ev)
            r_force, r_atom_vir, r_virial = edge_force_virial(
                g_e.to(force.dtype),
                ev.detach().to(force.dtype),
                graph.edge_index,
                graph.edge_mask,
                graph.destination_order,
                graph.destination_row_ptr,
                graph.source_row_ptr,
                graph.source_order,
                graph.n_node,
                n,
                True,
            )

        self.assertEqual(energy.dtype, torch.float64)
        torch.testing.assert_close(
            atom_e, e_atom.to(torch.float64), atol=1e-6, rtol=1e-6
        )
        torch.testing.assert_close(
            energy[0, 0], e_atom.sum().to(torch.float64), atol=1e-5, rtol=1e-6
        )
        torch.testing.assert_close(force, r_force, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(virial, r_virial, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(atom_vir, r_atom_vir, atol=1e-4, rtol=1e-4)

    def test_compact_canonical_model_trace(self) -> None:
        """The eight-tensor deployment forward composes under symbolic make_fx."""
        from deepmd.pt_expt.model import (
            EnergyModel,
        )
        from deepmd.pt_expt.utils.canonical_graph import (
            canonical_graph_from_neighbor_graph,
        )

        descriptor = _build_compressed_dpa1(
            self.device,
            [16, 32, 64],
            act="silu",
            axis_neuron=16,
        )
        fitting = self._build_fitting(descriptor.get_dim_out())
        model = EnergyModel(
            descriptor,
            fitting,
            type_map=["A", "B"],
        ).to(self.device)
        model.eval()
        graph, atype = self._graph(descriptor)
        graph = dataclasses.replace(graph, n_local=graph.n_node)
        compact = canonical_graph_from_neighbor_graph(graph)
        inputs = (
            atype,
            compact.n_node,
            compact.n_local,
            compact.source,
            compact.edge_vec,
            compact.destination_row_ptr,
            compact.source_row_ptr,
            compact.source_order,
        )

        reference = model.forward_lower_canonical_graph(
            *inputs,
            do_atomic_virial=True,
        )
        traced = model.forward_lower_canonical_graph_exportable(
            *inputs,
            do_atomic_virial=True,
            tracing_mode="real",
            _allow_non_fake_inputs=True,
        )
        actual = traced(*inputs)
        self.assertEqual(set(actual), set(reference))
        for key in actual:
            torch.testing.assert_close(actual[key], reference[key])

    def test_compact_canonical_torch_export_contract(self) -> None:
        """torch.export records the fixed eight-tensor deployment ABI."""
        from deepmd.pt_expt.model import (
            EnergyModel,
        )
        from deepmd.pt_expt.utils.serialization import (
            _trace_and_export,
        )

        descriptor = _build_compressed_dpa1(
            self.device,
            [16, 32, 64],
            act="silu",
            axis_neuron=16,
        )
        fitting = self._build_fitting(descriptor.get_dim_out())
        model = EnergyModel(
            descriptor,
            fitting,
            type_map=["A", "B"],
        ).to(self.device)
        model.eval()
        exported, metadata, _model_json, output_keys = _trace_and_export(
            {"model": model.serialize()},
            do_atomic_virial=True,
            lower_kind="dpa1_canonical",
        )

        self.assertEqual(metadata["lower_input_kind"], "dpa1_canonical")
        self.assertEqual(metadata["graph_edge_dtype"], "float32")
        self.assertNotIn("graph_index_dtype", metadata)
        self.assertEqual(
            output_keys,
            ["atom_energy", "energy", "force", "virial", "mask", "atom_virial"],
        )
        user_inputs = [
            spec
            for spec in exported.graph_signature.input_specs
            if spec.kind.name == "USER_INPUT"
        ]
        self.assertEqual(len(user_inputs), 8)

    def test_level2_permutation_csr_parity(self) -> None:
        """Level 2 preserves force and virial for an arbitrary edge stream."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_edge_csr,
        )
        from deepmd.kernels.cuda.dpa1.graph_compress import (
            dpa1_graph_compress_energy_force,
        )

        des = _build_compressed_dpa1(self.device, [16, 32, 64], act="silu")
        fit = self._build_fitting(des.get_dim_out())
        canonical_graph, atype = self._graph(des)
        edge_mask = canonical_graph.edge_mask.clone()
        edge_mask[0] = False
        canonical_graph = dataclasses.replace(
            canonical_graph,
            edge_mask=edge_mask,
        )
        generator = torch.Generator(device=self.device).manual_seed(29)
        permutation = torch.randperm(
            canonical_graph.edge_index.shape[1],
            generator=generator,
            device=self.device,
        )
        (
            edge_index,
            edge_vec,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_row_ptr,
            source_order,
        ) = build_edge_csr(
            canonical_graph.edge_index[:, permutation],
            canonical_graph.edge_vec[permutation],
            canonical_graph.edge_mask[permutation],
            atype.shape[0],
        )
        permutation_graph = dataclasses.replace(
            canonical_graph,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            destination_order=destination_order,
            destination_row_ptr=destination_row_ptr,
            source_row_ptr=source_row_ptr,
            source_order=source_order,
            destination_sorted=False,
        )
        type_embedding = des.type_embedding.call()

        with _CudaLevel("2"):
            canonical_output = dpa1_graph_compress_energy_force(
                des,
                fit,
                canonical_graph,
                atype,
                type_embedding,
                torch.ones(atype.shape[0], dtype=torch.bool, device=self.device),
                fit.bias_atom_e[:, 0].contiguous(),
                node_capacity=atype.shape[0],
                do_atomic_virial=True,
            )
            permutation_output = dpa1_graph_compress_energy_force(
                des,
                fit,
                permutation_graph,
                atype,
                type_embedding,
                torch.ones(atype.shape[0], dtype=torch.bool, device=self.device),
                fit.bias_atom_e[:, 0].contiguous(),
                node_capacity=atype.shape[0],
                do_atomic_virial=True,
            )
        for actual, expected in zip(
            permutation_output,
            canonical_output,
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@_GPU
class TestEdgeForceVirialCuda(unittest.TestCase):
    """Parity of the fused force / virial scatter (CUDA and CPU impls)."""

    def test_linear_csr_builder(self) -> None:
        source = torch.tensor([2, 0, 1, 2, 0, 0, 0], dtype=torch.int64, device="cuda")
        destination = torch.tensor(
            [0, 0, 1, 2, 2, 0, 0], dtype=torch.int64, device="cuda"
        )
        edge_index = torch.stack([source, destination])
        destination_order, destination_row_ptr, source_row_ptr, source_order = (
            torch.ops.deepmd.build_graph_csr(edge_index, 3, 5)
        )

        torch.testing.assert_close(
            destination_order, torch.arange(7, dtype=torch.int64, device="cuda")
        )
        torch.testing.assert_close(
            destination_row_ptr,
            torch.tensor([0, 2, 3, 5], dtype=torch.int64, device="cuda"),
        )
        torch.testing.assert_close(
            source_row_ptr,
            torch.tensor([0, 2, 3, 5], dtype=torch.int64, device="cuda"),
        )
        ordered_source = source[source_order[:5]]
        torch.testing.assert_close(
            ordered_source,
            torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64, device="cuda"),
        )
        torch.testing.assert_close(
            source_order[5:],
            torch.tensor([5, 6], dtype=torch.int64, device="cuda"),
        )

    def _random_graph(self, device, n_frame=2, n_per_frame=9, n_edge=200):
        gen = torch.Generator(device="cpu").manual_seed(7)
        n_node = torch.full((n_frame,), n_per_frame, dtype=torch.int64)
        total = int(n_node.sum())
        # Edges within each frame's node range, with a masked padding tail.
        src, dst = [], []
        for f in range(n_frame):
            lo = f * n_per_frame
            src.append(
                torch.randint(lo, lo + n_per_frame, (n_edge // n_frame,), generator=gen)
            )
            dst.append(
                torch.randint(lo, lo + n_per_frame, (n_edge // n_frame,), generator=gen)
            )
        edge_index = torch.stack([torch.cat(src), torch.cat(dst)])
        E = edge_index.shape[1]
        mask = torch.rand(E, generator=gen) > 0.15
        g_e = torch.randn(E, 3, generator=gen, dtype=torch.float64)
        edge_vec = torch.randn(E, 3, generator=gen, dtype=torch.float64)
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_edge_csr,
        )

        (
            edge_index,
            payload,
            _topology_mask,
            dst_order,
            dst_row_ptr,
            src_row_ptr,
            src_order,
        ) = build_edge_csr(
            edge_index,
            torch.cat([g_e, edge_vec], dim=1),
            torch.ones_like(mask),
            total,
        )
        g_e, edge_vec = payload[:, :3], payload[:, 3:]
        move = lambda t: t.to(device)  # noqa: E731
        return (
            move(g_e),
            move(edge_vec),
            move(edge_index),
            move(mask),
            move(dst_order),
            move(dst_row_ptr),
            move(src_row_ptr),
            move(src_order),
            move(n_node),
            total,
        )

    def _reference(
        self,
        g_e,
        edge_vec,
        edge_index,
        mask,
        dst_order,
        dst_row_ptr,
        src_row_ptr,
        src_order,
        n_node,
        total,
    ):
        from deepmd.dpmodel.utils.neighbor_graph import (
            edge_force_virial,
        )

        return edge_force_virial(
            g_e, edge_vec, edge_index, mask, n_node, node_capacity=total
        )

    def _fused(
        self,
        g_e,
        edge_vec,
        edge_index,
        mask,
        dst_order,
        dst_row_ptr,
        src_row_ptr,
        src_order,
        n_node,
        total,
    ):
        from deepmd.kernels.cuda.edge_force_virial import (
            edge_force_virial,
        )

        return edge_force_virial(
            g_e,
            edge_vec,
            edge_index,
            mask,
            dst_order,
            dst_row_ptr,
            src_row_ptr,
            src_order,
            n_node,
            total,
            True,
        )

    def _assert_device_parity(self, device) -> None:
        args = self._random_graph(torch.device(device))
        f0, av0, v0 = self._reference(*args)
        f1, av1, v1 = self._fused(*args)
        torch.testing.assert_close(f1, f0, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(av1, av0, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(v1, v0, atol=1e-12, rtol=1e-12)

    def test_parity_cuda(self) -> None:
        self._assert_device_parity("cuda")

    def test_compact_canonical_parity(self) -> None:
        """Source-only force and virial match the generic dual-CSR operator."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            NeighborGraph,
            build_edge_csr,
        )
        from deepmd.kernels.cuda.edge_force_virial import (
            canonical_edge_force_virial,
            edge_force_virial,
        )
        from deepmd.pt_expt.utils.canonical_graph import (
            canonical_graph_from_neighbor_graph,
        )

        (
            g_e,
            edge_vec,
            edge_index,
            _mask,
            _dst_order,
            _dst_row_ptr,
            _src_row_ptr,
            _src_order,
            n_node,
            total,
        ) = self._random_graph(torch.device("cuda"))
        (
            edge_index,
            payload,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_row_ptr,
            source_order,
        ) = build_edge_csr(
            edge_index,
            torch.cat((g_e, edge_vec), dim=1),
            torch.ones(edge_index.shape[1], dtype=torch.bool, device="cuda"),
            total,
            canonicalize=True,
        )
        g_e = payload[:, :3].to(torch.float32)
        edge_vec = payload[:, 3:].to(torch.float32)
        graph = NeighborGraph(
            n_node=n_node,
            edge_index=edge_index,
            edge_vec=edge_vec,
            edge_mask=edge_mask,
            n_local=n_node,
            destination_order=destination_order,
            destination_row_ptr=destination_row_ptr,
            source_row_ptr=source_row_ptr,
            source_order=source_order,
            destination_sorted=True,
        )
        compact = canonical_graph_from_neighbor_graph(graph)

        generic = edge_force_virial(
            g_e,
            edge_vec,
            edge_index,
            edge_mask,
            destination_order,
            destination_row_ptr,
            source_row_ptr,
            source_order,
            n_node,
            total,
            True,
        )
        canonical = canonical_edge_force_virial(
            g_e,
            compact.edge_vec,
            compact.destination_row_ptr,
            compact.source_row_ptr,
            compact.source_order,
            compact.n_node,
            total,
            True,
        )
        for actual, expected in zip(canonical, generic, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_many_small_frames(self) -> None:
        """Frame reduction is valid beyond the CUDA grid-y limit."""
        frame_count = 8192
        node = torch.arange(frame_count, device="cuda", dtype=torch.int64)
        edge_index = torch.stack([node, node])
        edge_mask = torch.ones(frame_count, device="cuda", dtype=torch.bool)
        generator = torch.Generator(device="cuda").manual_seed(23)
        edge_gradient = torch.randn(
            frame_count,
            3,
            generator=generator,
            device="cuda",
            dtype=torch.float64,
        )
        edge_vec = torch.randn(
            frame_count,
            3,
            generator=generator,
            device="cuda",
            dtype=torch.float64,
        )
        order = node.clone()
        row_ptr = torch.arange(frame_count + 1, device="cuda", dtype=torch.int64)
        n_node_per_frame = torch.ones(frame_count, device="cuda", dtype=torch.int64)

        force, atom_virial, virial = self._fused(
            edge_gradient,
            edge_vec,
            edge_index,
            edge_mask,
            order,
            row_ptr,
            row_ptr,
            order,
            n_node_per_frame,
            frame_count,
        )

        expected_atom_virial = -torch.einsum("ei,ej->eij", edge_gradient, edge_vec)
        torch.testing.assert_close(force, torch.zeros_like(force))
        torch.testing.assert_close(atom_virial, expected_atom_virial)
        torch.testing.assert_close(virial, expected_atom_virial)

    def test_ragged_many_frame_parity(self) -> None:
        """CSR scatter preserves force and virial across heterogeneous frames."""
        from deepmd.dpmodel.utils.neighbor_graph import (
            build_edge_csr,
        )

        n_node = torch.tensor([1, 4, 2, 7] * 1024, dtype=torch.int64)
        offsets = torch.cat(
            [torch.zeros(1, dtype=torch.int64), torch.cumsum(n_node, dim=0)]
        )
        source_parts = []
        destination_parts = []
        for frame in range(n_node.numel()):
            nodes = torch.arange(int(offsets[frame]), int(offsets[frame + 1]))
            source_parts.extend((nodes, torch.roll(nodes, shifts=1)))
            destination_parts.extend((nodes, nodes))
        edge_index = torch.stack(
            [torch.cat(source_parts), torch.cat(destination_parts)]
        )
        generator = torch.Generator(device="cpu").manual_seed(31)
        edge_mask = (
            torch.rand(
                edge_index.shape[1],
                generator=generator,
            )
            > 0.2
        )
        edge_gradient = torch.randn(
            edge_index.shape[1],
            3,
            generator=generator,
            dtype=torch.float64,
        )
        edge_vec = torch.randn(
            edge_index.shape[1],
            3,
            generator=generator,
            dtype=torch.float64,
        )
        total = int(n_node.sum())
        (
            edge_index,
            payload,
            _topology_mask,
            destination_order,
            destination_row_ptr,
            source_row_ptr,
            source_order,
        ) = build_edge_csr(
            edge_index,
            torch.cat([edge_gradient, edge_vec], dim=1),
            torch.ones_like(edge_mask),
            total,
        )
        edge_gradient, edge_vec = payload[:, :3], payload[:, 3:]
        args = (
            edge_gradient.cuda(),
            edge_vec.cuda(),
            edge_index.cuda(),
            edge_mask.cuda(),
            destination_order.cuda(),
            destination_row_ptr.cuda(),
            source_row_ptr.cuda(),
            source_order.cuda(),
            n_node.cuda(),
            total,
        )

        reference = self._reference(*args)
        fused = self._fused(*args)
        for actual, expected in zip(fused, reference, strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

    def test_parity_cpu_impl(self) -> None:
        # The CPU registration serves trace-time sample evaluation.
        self._assert_device_parity("cpu")


if __name__ == "__main__":
    unittest.main()
