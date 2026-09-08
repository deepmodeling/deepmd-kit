# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end parity of the pt_expt DPA4 accelerated inference paths."""

import numpy as np
import pytest
import torch

try:
    import deepmd.pt.cxx_op  # noqa: F401
except ImportError:
    pass

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
    assert (descriptor._cuda_radial_fn is not None) is expected_bound
    assert all(module._cuda_scatter is expected_bound for module in initial_embeddings)
    assert all(
        (module._grid_pair_fn is not None) is expected_bound for module in grid_nets
    )
    cpu_zonal = torch.empty(1)
    assert all(not module._can_fuse_scatter(cpu_zonal) for module in initial_embeddings)
    for module in initial_embeddings:
        module._force_fused_scatter = True
    assert all(
        module._can_fuse_scatter(cpu_zonal) is expected_bound
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
        if conv._cuda_conv_fn is None:
            pytest.skip("the descriptor layout has no fused CUDA convolution")
        assert conv._flash_atten_fn is not None
        assert conv._cuda_value_train is None
        assert not accelerated._wigner_free_conv
        assert accelerated._build_full_wigner()

        output, gradient = self._step(accelerated)
        np.testing.assert_allclose(output, dense_output, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(gradient, dense_gradient, rtol=2e-4, atol=2e-5)

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
            assert so2._flash_atten_fn is not None
            assert so2._triton_value_path is not None
            assert wigner._use_triton_monomials
        elif backend == "cuda":
            if so2._cuda_conv_fn is None:
                pytest.skip("DPA4 CUDA operators are unavailable")
            assert accelerated._cuda_radial_fn is not None
            assert accelerated._cuda_wigner_fn is not None
        else:
            assert so2._flash_atten_fn is not None
            assert so2._cutile_value_path is not None
            assert wigner._use_cutile_monomials

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
