# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end parity of the pt_expt DPA4 accelerated inference paths."""

from types import (
    SimpleNamespace,
)
from typing import (
    Any,
)

import numpy as np
import pytest
import torch

try:
    import deepmd.pt.cxx_op  # noqa: F401
except ImportError:
    pass

from deepmd.dpmodel.utils.neighbor_graph import (
    NeighborGraph,
)
from deepmd.pt_expt.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.pt_expt.descriptor.dpa4_nn.embedding import (
    GeometricInitialEmbedding,
)
from deepmd.pt_expt.descriptor.dpa4_nn.grid_net import (
    S2GridNet,
    SO3GridNet,
)
from deepmd.pt_expt.descriptor.dpa4_nn.so2 import (
    SO2Convolution,
)
from deepmd.pt_expt.descriptor.dpa4_nn.wignerd import (
    WignerDCalculator,
)
from deepmd.pt_expt.kernels.cuda.dpa4 import (
    edge_radial,
    grid_pair,
    so2_conv,
    zonal_scatter,
)
from deepmd.pt_expt.kernels.cute.sezm import (
    runtime_policy,
)
from deepmd.pt_expt.kernels.cute.sezm.output_grid import (
    readout_l0,
)
from deepmd.pt_expt.kernels.cute.sezm.so2 import operation as so2
from deepmd.pt_expt.kernels.cutile import (
    CUTILE_AVAILABLE,
)
from deepmd.pt_expt.kernels.triton.sezm.force_assembly import (
    FORCE_ASSEMBLY_TRITON_AVAILABLE,
)
from deepmd.pt_expt.kernels.triton.sezm.so2_value_path import (
    SO2_VALUE_PATH_TRITON_AVAILABLE,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...common.test_mixins import (
    TestCaseSingleFrameWithNlist,
)


def _make_descriptor(
    ntypes: int,
    sel: list[int],
    rcut: float,
    precision: str = "float32",
    *,
    source_gated: bool = False,
) -> DescrptDPA4:
    return DescrptDPA4(
        ntypes=ntypes,
        sel=sel,
        rcut=rcut,
        channels=32,
        n_radial=8,
        lmax=2,
        mmax=1,
        n_blocks=2,
        mixing_layers=3,
        radial_so2_mode="degree_channel",
        radial_so2_rank=1,
        n_atten_head=1,
        grid_branch=[1, 1, 1],
        s2_activation=[False, True],
        random_gamma=False,
        precision=precision,
        seed=7,
        inner_clamp_r_inner=0.8 if source_gated else None,
        inner_clamp_r_outer=1.2 if source_gated else None,
    )


def make_neo_descriptor(
    sel: int | list[int] = 4,
    *,
    edge_norm: bool = True,
) -> DescrptDPA4:
    """Build the exact Neo layout served by the current CuTe kernels."""
    return DescrptDPA4(
        ntypes=2,
        sel=sel,
        channels=32,
        lmax=3,
        mmax=1,
        n_blocks=2,
        so2_layers=3,
        n_focus=2,
        message_node_so3=True,
        ffn_neurons=0,
        ffn_so3_grid=True,
        grid_branch=[0, 0, 1],
        ffn_blocks=1,
        so3_readout="mlp",
        use_amp=False,
        random_gamma=False,
        edge_norm=edge_norm,
        precision="float32",
        trainable=False,
        seed=42,
    ).eval()


def test_cute_contract_recognizes_pt_expt_neo_modules(monkeypatch) -> None:
    """Keep CuTe eligibility independent of the concrete PT module classes."""
    for name in (
        "DP_TRITON_INFER",
        "DP_CUDA_INFER",
        "DP_CUTILE_INFER",
        "DP_CUTE_INFER",
    ):
        monkeypatch.setenv(name, "0")

    descriptor = make_neo_descriptor()

    assert descriptor.blocks
    assert all(so2.is_supported_neo_so2_block(block) for block in descriptor.blocks)
    assert readout_l0.has_neo_readout_contract(descriptor.output_ffn)


def test_packed_wigner_graph_compacts_and_reuses_csr(monkeypatch) -> None:
    """Reuse canonical graph metadata without sorting the edge stream again."""
    monkeypatch.setenv("DP_CUTE_INFER", "1")
    descriptor = _make_descriptor(2, [4], 4.0).eval()
    monkeypatch.setattr(
        descriptor,
        "is_cute_infer_packed_wigner_candidate",
        lambda *args: True,
    )
    graph = NeighborGraph(
        n_node=torch.tensor([4], dtype=torch.int64),
        edge_index=torch.tensor(
            [[3, 0, 2, 0, 0], [0, 1, 2, 0, 0]],
            dtype=torch.int64,
        ),
        edge_vec=torch.arange(15, dtype=torch.float32).reshape(5, 3),
        edge_mask=torch.tensor([True, True, True, False, False]),
        destination_order=torch.arange(5, dtype=torch.int64),
        destination_row_ptr=torch.tensor([0, 1, 2, 3, 3], dtype=torch.int64),
        source_order=torch.tensor([1, 2, 0, 3, 4], dtype=torch.int64),
        source_row_ptr=torch.tensor([0, 1, 1, 2, 3], dtype=torch.int64),
        destination_sorted=True,
    )
    monkeypatch.setattr(
        torch,
        "argsort",
        lambda *args, **kwargs: pytest.fail("canonical CSR must not be rebuilt"),
    )

    packed_graph = descriptor.prepare_packed_wigner_graph(graph, n_nodes=4)

    assert packed_graph is not None
    torch.testing.assert_close(
        packed_graph.edge_index,
        torch.tensor([[3, 0, 2], [0, 1, 2]], dtype=torch.int64),
    )
    torch.testing.assert_close(
        packed_graph.edge_vec,
        torch.arange(9, dtype=torch.float32).reshape(3, 3),
    )
    assert torch.all(packed_graph.edge_mask)
    assert packed_graph.destination_sorted
    torch.testing.assert_close(
        packed_graph.destination_row_ptr,
        torch.tensor([0, 1, 2, 3, 3], dtype=torch.int64),
    )
    torch.testing.assert_close(
        packed_graph.source_order,
        torch.tensor([1, 2, 0], dtype=torch.int64),
    )
    torch.testing.assert_close(
        packed_graph.source_row_ptr,
        torch.tensor([0, 1, 1, 2, 3], dtype=torch.int64),
    )

    edge_cache = SimpleNamespace(
        src=packed_graph.edge_index[0],
        dst=packed_graph.edge_index[1],
        csr_cache={
            "dst": (
                packed_graph.destination_order,
                packed_graph.destination_row_ptr,
            ),
            "src": (packed_graph.source_order, packed_graph.source_row_ptr),
        },
        destinations_sorted=True,
        D_packed=torch.empty(3, 46),
        edge_src_gate=None,
    )
    metadata = descriptor.prepare_cute_infer_so2_metadata(
        edge_cache,
        n_nodes=4,
    )
    assert metadata is not None
    destination_row_ptr, source_order, source_row_ptr = metadata
    torch.testing.assert_close(
        destination_row_ptr,
        packed_graph.destination_row_ptr.to(torch.int32),
    )
    torch.testing.assert_close(
        source_order,
        packed_graph.source_order.to(torch.int32),
    )
    torch.testing.assert_close(
        source_row_ptr,
        packed_graph.source_row_ptr.to(torch.int32),
    )


def test_packed_wigner_graph_without_csr_builds_metadata_once(monkeypatch) -> None:
    """Build each CSR ordering once when the graph does not provide one."""
    monkeypatch.setenv("DP_CUTE_INFER", "1")
    descriptor = _make_descriptor(2, [4], 4.0).eval()
    monkeypatch.setattr(
        descriptor,
        "is_cute_infer_packed_wigner_candidate",
        lambda *args: True,
    )
    graph = NeighborGraph(
        n_node=torch.tensor([2], dtype=torch.int64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
        edge_vec=torch.ones(2, 3),
        edge_mask=torch.ones(2, dtype=torch.bool),
    )
    argsort = torch.argsort
    sort_count = 0

    def count_argsort(*args: Any, **kwargs: Any) -> torch.Tensor:
        nonlocal sort_count
        sort_count += 1
        return argsort(*args, **kwargs)

    monkeypatch.setattr(torch, "argsort", count_argsort)

    packed_graph = descriptor.prepare_packed_wigner_graph(graph, n_nodes=2)

    assert packed_graph is not None
    edge_cache = SimpleNamespace(
        src=packed_graph.edge_index[0],
        dst=packed_graph.edge_index[1],
        csr_cache={
            "dst": (
                packed_graph.destination_order,
                packed_graph.destination_row_ptr,
            ),
            "src": (packed_graph.source_order, packed_graph.source_row_ptr),
        },
        destinations_sorted=True,
        D_packed=torch.empty(2, 46),
        edge_src_gate=None,
    )

    assert descriptor.prepare_cute_infer_so2_metadata(edge_cache, n_nodes=2) is not None
    assert sort_count == 2


def test_packed_wigner_cache_routes_pt_expt_block_to_cute(monkeypatch) -> None:
    """Use packed Wigner storage as the prevalidated per-block dispatch token."""
    block = make_neo_descriptor().blocks[0]
    destination_row_ptr = torch.tensor([0, 1], dtype=torch.int32)
    source_order = torch.tensor([0], dtype=torch.int32)
    source_row_ptr = torch.tensor([0, 1], dtype=torch.int32)
    edge_cache = SimpleNamespace(
        D_packed=torch.empty(1, 46),
        cute_infer_so2_metadata=(
            destination_row_ptr,
            source_order,
            source_row_ptr,
        ),
    )
    expected = torch.empty(1)

    def run_cute(
        candidate_block: Any,
        x: torch.Tensor,
        candidate_edge_cache: Any,
        radial_feat: torch.Tensor,
        *,
        dst_ptr: torch.Tensor,
        source_order: torch.Tensor,
        source_ptr: torch.Tensor,
    ) -> torch.Tensor:
        del x, radial_feat
        assert candidate_block is block
        assert candidate_edge_cache is edge_cache
        assert dst_ptr is destination_row_ptr
        assert source_order is edge_cache.cute_infer_so2_metadata[1]
        assert source_ptr is source_row_ptr
        return expected

    monkeypatch.setattr(so2, "maybe_run_cute_so2", run_cute)

    actual = block._run_so2_unit_impl(
        torch.empty(1),
        edge_cache,
        torch.empty(1),
    )

    assert actual is expected


@pytest.mark.parametrize(
    ("precision", "expected_bound"),
    [("float32", True), ("float64", False)],
)
def test_fp32_only_cuda_bindings(
    monkeypatch,
    precision: str,
    expected_bound: bool,
) -> None:
    """Bind the handwritten CUDA path only for its supported precision."""
    for name in (
        "DP_TRITON_INFER",
        "DP_CUDA_INFER",
        "DP_CUTILE_INFER",
        "DP_CUTE_INFER",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    monkeypatch.setattr(edge_radial, "op_available", lambda: True)
    monkeypatch.setattr(grid_pair, "op_available", lambda: True)
    monkeypatch.setattr(zonal_scatter, "op_available", lambda: True)

    descriptor = _make_descriptor(2, [20], 4.0, precision=precision).eval()
    initial_embeddings = [
        module
        for module in descriptor.modules()
        if isinstance(module, GeometricInitialEmbedding)
    ]
    grid_nets = [
        module
        for module in descriptor.modules()
        if isinstance(module, (S2GridNet, SO3GridNet))
    ]

    assert len(initial_embeddings) == 1
    assert grid_nets
    assert (descriptor.cuda_infer_l_1_radial is not None) is expected_bound
    assert all(
        module.cuda_infer_l_1_scatter is expected_bound for module in initial_embeddings
    )
    assert all(
        (module.cuda_infer_l_1_grid_pair is not None) is expected_bound
        for module in grid_nets
    )
    cpu_zonal = torch.empty(1)
    assert all(
        not module.can_run_cuda_infer_l_1_scatter(cpu_zonal)
        for module in initial_embeddings
    )
    for module in initial_embeddings:
        module.force_cuda_infer_l_1_scatter = True
    assert all(
        module.can_run_cuda_infer_l_1_scatter(cpu_zonal) is expected_bound
        for module in initial_embeddings
    )


def test_cuda_edge_csr_preserves_symbolic_node_count() -> None:
    """Keep the CSR row-pointer length tied to the dynamic node axis."""

    class EdgeCSR(torch.nn.Module):
        def forward(
            self,
            key: torch.Tensor,
            nodes: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return so2_conv.edge_csr(key, nodes.shape[0])

    key = torch.tensor([0, 2, 1, 6, 2, 0, 5, 3, 6, 4, 1])
    nodes = torch.empty(7)
    exported = torch.export.export(
        EdgeCSR(),
        (key, nodes),
        dynamic_shapes=(
            {0: torch.export.Dim("n_edge", min=2)},
            {0: torch.export.Dim("n_node", min=2)},
        ),
        strict=False,
    )

    assert all(
        "bincount" not in str(node.target) for node in exported.graph_module.graph.nodes
    )
    replay_key = torch.tensor([4, 0, 1, 4, 3, 1, 0, 2])
    replay_nodes = torch.empty(5)
    order, row_ptr = exported.module()(replay_key, replay_nodes)
    torch.testing.assert_close(order, torch.tensor([1, 6, 2, 5, 7, 4, 0, 3]))
    torch.testing.assert_close(row_ptr, torch.tensor([0, 2, 4, 5, 6, 8]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
class TestDPA4AcceleratedParity(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def _inputs(self):
        coord = torch.tensor(
            self.coord_ext, dtype=torch.float32, device=self.device, requires_grad=True
        )
        atype = torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=torch.int64, device=self.device)
        return coord, atype, nlist

    def _step(self, descriptor: DescrptDPA4) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one descriptor output and its coordinate gradient."""
        coord, atype, nlist = self._inputs()
        output = descriptor(coord, atype, nlist)[0]
        gradient = torch.autograd.grad(output.sum(), coord)[0]
        return output.detach().cpu().numpy(), gradient.detach().cpu().numpy()

    def test_source_gated_flash_retains_dense_rotations(self, monkeypatch) -> None:
        """Source-gated flash inference retains the rotations its fallback uses."""
        if not so2_conv.op_available():
            pytest.skip("the DPA4 CUDA inference operators are unavailable")
        if not SO2_VALUE_PATH_TRITON_AVAILABLE:
            pytest.skip("Triton is unavailable")

        for name in (
            "DP_TRITON_INFER",
            "DP_CUDA_INFER",
            "DP_CUTILE_INFER",
            "DP_CUTE_INFER",
            "DP_TRITON_TRAIN",
            "DP_CUDA_TRAIN",
        ):
            monkeypatch.setenv(name, "0")
        data = _make_descriptor(
            self.nt,
            self.sel_mix,
            self.rcut,
            source_gated=True,
        ).serialize()
        dense = DescrptDPA4.deserialize(data).to(self.device).eval()
        dense_output, dense_gradient = self._step(dense)

        monkeypatch.setenv("DP_TRITON_INFER", "1")
        monkeypatch.setenv("DP_CUDA_INFER", "2")
        accelerated = DescrptDPA4.deserialize(data).to(self.device).eval()
        conv = next(
            module
            for module in accelerated.modules()
            if isinstance(module, SO2Convolution)
        )
        if conv.cuda_infer_l_2_conv is None:
            pytest.skip("the descriptor layout has no fused CUDA convolution")
        assert conv.flash_attention is not None
        assert conv.cuda_train_value is None
        assert not accelerated.cuda_infer_l_2_covers_all_blocks
        assert accelerated._build_full_wigner()

        output, gradient = self._step(accelerated)
        np.testing.assert_allclose(output, dense_output, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(gradient, dense_gradient, rtol=2e-4, atol=2e-5)

    @pytest.mark.parametrize("edge_norm", [True, False])
    def test_cute_neo_forward_and_coordinate_gradient(
        self,
        monkeypatch,
        edge_norm: bool,
    ) -> None:
        """Exercise packed Wigner and SO2 through the PT-expt descriptor."""
        try:
            import cuda.bindings.driver  # noqa: F401
            import cutlass.cute  # noqa: F401
            import tvm_ffi  # noqa: F401

            from deepmd.pt_expt.kernels.cute.sezm import (
                wignerd,
            )
        except ImportError:
            pytest.skip("the CuTe DSL runtime is unavailable")
        capability = tuple(torch.cuda.get_device_capability(self.device))
        if not runtime_policy.is_supported_so2_capability(capability):
            pytest.skip(f"the CuTe SO2 path does not support {capability}")

        for name in (
            "DP_TRITON_INFER",
            "DP_CUDA_INFER",
            "DP_CUTILE_INFER",
            "DP_CUTE_INFER",
        ):
            monkeypatch.setenv(name, "0")
        data = make_neo_descriptor(self.sel_mix, edge_norm=edge_norm).serialize()
        reference = DescrptDPA4.deserialize(data).to(self.device).eval()
        reference_output, reference_gradient = self._step(reference)

        monkeypatch.setenv("DP_CUTE_INFER", "1")
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        accelerated = DescrptDPA4.deserialize(data).to(self.device).eval()
        dispatch_count = {"wigner": 0, "so2": 0}
        run_wignerd = wignerd.run_cute_wignerd
        run_so2 = so2.maybe_run_cute_so2

        def count_wignerd(*args: Any, **kwargs: Any) -> Any:
            result = run_wignerd(*args, **kwargs)
            dispatch_count["wigner"] += result is not None
            return result

        def count_so2(*args: Any, **kwargs: Any) -> Any:
            result = run_so2(*args, **kwargs)
            dispatch_count["so2"] += result is not None
            return result

        monkeypatch.setattr(wignerd, "run_cute_wignerd", count_wignerd)
        monkeypatch.setattr(so2, "maybe_run_cute_so2", count_so2)
        output, gradient = self._step(accelerated)

        assert dispatch_count == {"wigner": 1, "so2": len(accelerated.blocks)}
        np.testing.assert_allclose(output, reference_output, rtol=5e-5, atol=5e-5)
        np.testing.assert_allclose(
            gradient,
            reference_gradient,
            rtol=5e-5,
            atol=5e-5,
        )

    @pytest.mark.parametrize("backend", ["triton", "cuda", "cutile"])
    def test_forward_and_coordinate_gradient(self, monkeypatch, backend) -> None:
        if backend == "triton" and not FORCE_ASSEMBLY_TRITON_AVAILABLE:
            pytest.skip("Triton is unavailable")
        if backend == "cutile" and not CUTILE_AVAILABLE:
            pytest.skip("cuda.tile is unavailable")

        for name in (
            "DP_TRITON_INFER",
            "DP_CUDA_INFER",
            "DP_CUTILE_INFER",
            "DP_CUTE_INFER",
        ):
            monkeypatch.setenv(name, "0")
        data = _make_descriptor(self.nt, self.sel_mix, self.rcut).serialize()
        reference = DescrptDPA4.deserialize(data).to(self.device).eval()

        levels = {
            "triton": ("2", "0", "0"),
            "cuda": ("0", "2", "0"),
            "cutile": ("0", "0", "1"),
        }[backend]
        monkeypatch.setenv("DP_TRITON_INFER", levels[0])
        monkeypatch.setenv("DP_CUDA_INFER", levels[1])
        monkeypatch.setenv("DP_CUTILE_INFER", levels[2])
        accelerated = DescrptDPA4.deserialize(data).to(self.device).eval()

        so2 = next(
            module
            for module in accelerated.modules()
            if isinstance(module, SO2Convolution)
        )
        wigner = next(
            module
            for module in accelerated.modules()
            if isinstance(module, WignerDCalculator)
        )
        if backend == "triton":
            assert so2.flash_attention is not None
            assert so2.triton_infer_l_2_value is not None
            assert wigner.triton_infer_l_1_monomials
        elif backend == "cuda":
            if so2.cuda_infer_l_2_conv is None:
                pytest.skip("DPA4 CUDA operators are unavailable")
            assert accelerated.cuda_infer_l_1_radial is not None
            assert accelerated.cuda_infer_l_1_wigner is not None
        else:
            assert so2.flash_attention is not None
            assert so2.cutile_infer_value is not None
            assert wigner.cutile_infer_monomials

        coord_ref, atype, nlist = self._inputs()
        output_ref = reference(coord_ref, atype, nlist)[0]
        grad_ref = torch.autograd.grad(output_ref.sum(), coord_ref)[0]

        coord, atype, nlist = self._inputs()
        output = accelerated(coord, atype, nlist)[0]
        grad = torch.autograd.grad(output.sum(), coord)[0]

        np.testing.assert_allclose(
            output.detach().cpu().numpy(),
            output_ref.detach().cpu().numpy(),
            rtol=2e-4,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            grad.detach().cpu().numpy(),
            grad_ref.detach().cpu().numpy(),
            rtol=2e-4,
            atol=2e-5,
        )


@pytest.mark.parametrize("lmax", [2, 3, 4, 5, 6])
def test_wigner_kernels_are_device_buffers_outside_the_state_dict(lmax: int) -> None:
    """Hold the low-order Wigner kernels as buffers, but not as saved state.

    The dpmodel calculator keeps them as NumPy arrays inside a container, which
    the generic conversion cannot reach: every evaluation would convert them
    again, and a NumPy to CUDA conversion is a synchronizing host-to-device
    copy on the training hot path. Registering them as buffers moves them with
    the module instead.

    They are a pure function of ``lmax``, so they must stay out of the state
    dict: a stored copy would both break checkpoints written before they
    existed and let a checkpoint override a value the configuration decides.
    """
    calculator = WignerDCalculator(lmax)

    held = [name for name, _ in calculator.named_buffers() if "_small_order_" in name]
    assert held, "no low-order kernel was adopted as a buffer"
    saved = [
        key
        for key in calculator.state_dict()
        if "_small_order_" in key or "_l2_monomial_coeff" in key
    ]
    assert not saved, (
        f"configuration-derived arrays leaked into the state dict: {saved}"
    )

    # The container must alias the buffers, since that is what the dpmodel
    # evaluation reads; an array left behind there would keep converting.
    for name in held:
        kernel = getattr(
            calculator.small_order_kernels, name.removeprefix("_small_order_")
        )
        assert isinstance(kernel, torch.Tensor), name
