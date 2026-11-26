# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end parity of the pt_expt DPA4 accelerated inference paths."""

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
from ...dpa4_fixtures import (
    four_atom_spin_inputs,
    jitter_zero_arrays,
)


def _make_descriptor(
    ntypes: int,
    sel: list[int],
    rcut: float,
    precision: str = "float32",
    *,
    bridged: bool = False,
    env_exp: int | list[int] | None = None,
    **overrides: Any,
) -> DescrptDPA4:
    options = {
        "ntypes": ntypes,
        "sel": sel,
        "rcut": rcut,
        "env_exp": env_exp,
        "channels": 32,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "mixing_layers": 3,
        "radial_so2_mode": "degree_channel",
        "radial_so2_rank": 1,
        "n_atten_head": 1,
        "grid_branch": [1, 1, 1],
        "s2_activation": [False, True],
        "random_gamma": False,
        "precision": precision,
        "seed": 7,
        "type_map": ["O", "H"][:ntypes] if bridged else None,
        "contact_clamp": 0.75 if bridged else None,
    }
    return DescrptDPA4(**{**options, **overrides})


@pytest.mark.parametrize(
    ("precision", "expected_bound"),  # descriptor dtype and CUDA binding eligibility
    [("float32", True), ("float64", False)],
)
@pytest.mark.parametrize("env_exp", [5, [7, 5]])  # single vs double envelope
def test_fp32_only_cuda_bindings(
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    expected_bound: bool,
    env_exp: int | list[int],
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
    # Binding eligibility is independent of native operator registration.
    monkeypatch.setattr(edge_radial, "ensure_registered", lambda: None)
    monkeypatch.setattr(grid_pair, "op_available", lambda: True)
    monkeypatch.setattr(zonal_scatter, "op_available", lambda: True)

    descriptor = _make_descriptor(
        2, [20], 4.0, precision=precision, env_exp=env_exp
    ).eval()
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


@pytest.mark.usefixtures("calibrated_so2_device")
def test_profitability_counts_the_shared_parity_block() -> None:
    """The fused convolution is judged on the arithmetic it executes.

    A parity map multiplies both signs of an order by one shared ``W_u``
    block, half the order arithmetic of the plain layer of the same shape, so
    a shape whose plain layer exceeds the budget can stay within it.
    """

    def conv(parity: bool) -> object:
        # lmax 1, focus width 64, two focus streams: 73728 multiply-adds per
        # plain layer and 57344 with the shared parity block.
        return type(
            "Conv", (), {"so2_focus_dim": 64, "lmax": 1, "n_focus": 2, "parity": parity}
        )()

    assert not so2_conv._profitable(conv(False))
    assert so2_conv._profitable(conv(True))


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
@pytest.mark.usefixtures("calibrated_so2_device")
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

    def test_serialized_radial_basis_reaches_fused_inference(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fused inference of a deserialized descriptor reads its learned basis."""
        if not edge_radial.op_available():
            pytest.skip("DPA4 CUDA radial operators are unavailable")
        for name in (
            "DP_TRITON_INFER",
            "DP_CUDA_INFER",
            "DP_CUTILE_INFER",
            "DP_CUTE_INFER",
        ):
            monkeypatch.setenv(name, "0")
        data = _make_descriptor(self.nt, self.sel_mix, self.rcut).serialize()
        data = jitter_zero_arrays(data, np.random.default_rng(73))
        variables = data["@variables"]
        variables["radial_basis.adam_freqs"] = (
            variables["radial_basis.adam_freqs"] + np.pi / self.rcut
        )
        reference = DescrptDPA4.deserialize(data).to(self.device).eval()
        output_ref, gradient_ref = self._step(reference)
        assert np.max(np.abs(gradient_ref)) > 1e-10

        monkeypatch.setenv("DP_CUDA_INFER", "1")
        restored = DescrptDPA4.deserialize(data).to(self.device).eval()
        assert restored._cuda_radial_fn is not None
        calls: list[tuple[int, ...]] = []
        original = edge_radial.EdgeRadialCuda.__call__

        def spy(operator, edge_len, keep):
            calls.append(tuple(edge_len.shape))
            return original(operator, edge_len, keep)

        monkeypatch.setattr(edge_radial.EdgeRadialCuda, "__call__", spy)
        output, gradient = self._step(restored)
        assert calls, "the fused radial operator was not used"
        np.testing.assert_allclose(output, output_ref, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(gradient, gradient_ref, rtol=2e-4, atol=2e-5)

    @pytest.mark.parametrize("l_schedule", [[1, 2], [2, 1]])  # rising and falling
    def test_fused_convolution_of_a_lower_degree_block_with_two_focus_streams(
        self, monkeypatch: pytest.MonkeyPatch, l_schedule: list[int]
    ) -> None:
        """A block below the largest degree competes its focus streams exactly.

        The cross-focus competition reads the edge Wigner blocks of the largest
        degree of the descriptor against the node features of the block, which
        span only the block's own degree.
        """
        if not so2_conv.op_available():
            pytest.skip("DPA4 CUDA convolution operators are unavailable")
        for name in (
            "DP_TRITON_INFER",
            "DP_CUDA_INFER",
            "DP_CUTILE_INFER",
            "DP_CUTE_INFER",
        ):
            monkeypatch.setenv(name, "0")
        data = _make_descriptor(
            self.nt,
            self.sel_mix,
            self.rcut,
            l_schedule=l_schedule,
            n_focus=2,
            channels=64,
            focus_dim=32,
        ).serialize()
        data = jitter_zero_arrays(data, np.random.default_rng(79))
        reference = DescrptDPA4.deserialize(data).to(self.device).eval()
        output_ref, gradient_ref = self._step(reference)

        monkeypatch.setenv("DP_CUDA_INFER", "2")
        fused = DescrptDPA4.deserialize(data).to(self.device).eval()
        assert all(block.so2_conv._cuda_conv_fn is not None for block in fused.blocks)
        assert all(block.so2_conv._cuda_conv_fn._compete for block in fused.blocks)
        output, gradient = self._step(fused)
        np.testing.assert_allclose(output, output_ref, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(gradient, gradient_ref, rtol=2e-4, atol=2e-5)

    def test_bridged_inference_takes_the_fused_convolution(self, monkeypatch) -> None:
        """A bridged descriptor runs the fused convolution and matches the dense path.

        The contact clamp acts on the edge lengths before the envelope and the
        radial features are built, so the fused convolution and the flash
        attention see it through the inputs they already read, and the dense
        per-edge rotations are not built. The data edges are set so that the
        pairs of the fixture lie in the frozen zone and in the transition zone
        of their clamp, where the clamped descriptor differs from the
        unbridged one.
        """
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
            bridged=True,
        ).serialize()
        # Pair distances of the fixture in Å: O-O 1.0 and 2.0, O-H 1.0 and
        # sqrt(2). With the edges below, the O-O pair at 1.0 and the O-H
        # pair at 1.0 are frozen, the O-H pair at sqrt(2) lies in the
        # transition zone, and the O-O pair at 2.0 is read unchanged.
        data["@variables"]["contact_clamp.edges"] = np.array(
            [[1.5, 1.8, 1.0], [1.8, 1.4, 1.0], [1.0, 1.0, 1.0]]
        )
        data = jitter_zero_arrays(data, np.random.default_rng(3))
        unbridged = {
            **data,
            "config": {**data["config"], "contact_clamp": None},
            "@variables": {
                key: value
                for key, value in data["@variables"].items()
                if key != "contact_clamp.edges"
            },
        }
        dense = DescrptDPA4.deserialize(data).to(self.device).eval()
        dense_output, dense_gradient = self._step(dense)
        control = DescrptDPA4.deserialize(unbridged).to(self.device).eval()
        control_output, _ = self._step(control)
        assert np.abs(dense_output - control_output).max() > 1e-3

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
        assert accelerated._wigner_free_conv
        assert not accelerated._build_full_wigner()

        output, gradient = self._step(accelerated)
        np.testing.assert_allclose(output, dense_output, rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(gradient, dense_gradient, rtol=2e-4, atol=2e-5)

    @pytest.mark.parametrize(
        "backend", ["triton", "cuda", "cutile"]
    )  # inference kernels
    @pytest.mark.parametrize("env_exp", [None, 5])  # default double vs single envelope
    def test_forward_and_coordinate_gradient(
        self, monkeypatch: pytest.MonkeyPatch, backend: str, env_exp: int | None
    ) -> None:
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
        data = _make_descriptor(
            self.nt, self.sel_mix, self.rcut, env_exp=env_exp
        ).serialize()
        data = jitter_zero_arrays(data, np.random.default_rng(73))
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
        assert grad_ref.abs().max().item() > 1e-10

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.usefixtures("calibrated_so2_device")
@pytest.mark.parametrize("spin_mode", ["orbit", "exchange"])
def test_native_spin_cuda_matches_reference_and_symmetry(
    monkeypatch: pytest.MonkeyPatch, spin_mode: str
) -> None:
    """The fused inference path preserves scalar spin conditioning and axial parity."""
    if not so2_conv.op_available():
        pytest.skip("the DPA4 CUDA inference operators are unavailable")
    for name in (
        "DP_TRITON_INFER",
        "DP_CUDA_INFER",
        "DP_CUTILE_INFER",
        "DP_CUTE_INFER",
        "DP_TRITON_TRAIN",
        "DP_CUDA_TRAIN",
    ):
        monkeypatch.setenv(name, "0")
    data = _make_descriptor(2, [16], 4.0).serialize()
    data["config"].update(use_spin=[True, False], spin_mode=spin_mode)
    native = DescrptDPA4(**data["config"])
    data = jitter_zero_arrays(native.serialize(), np.random.default_rng(319))
    reference = DescrptDPA4.deserialize(data).cuda().eval()
    monkeypatch.setenv("DP_CUDA_INFER", "2")
    fused = DescrptDPA4.deserialize(data).cuda().eval()
    conv = fused.blocks[0].so2_conv
    assert conv._cuda_conv_fn is not None
    assert fused._cuda_radial_fn is not None
    coord_array, atype_array, spin_array, nlist_array = four_atom_spin_inputs()
    coord = torch.as_tensor(coord_array, dtype=torch.float32, device="cuda")
    atype = torch.as_tensor(atype_array, device="cuda")
    nlist = torch.as_tensor(nlist_array, device="cuda")
    spin = torch.as_tensor(spin_array, dtype=torch.float32, device="cuda")

    def evaluate(
        model: DescrptDPA4, positions: torch.Tensor, moments: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = positions.detach().requires_grad_(True)
        moments = moments.detach().requires_grad_(True)
        output = model(positions, atype, nlist, spin=moments)[0]
        dr, ds = torch.autograd.grad(output.sum(), (positions, moments))
        return output.detach(), dr.detach(), ds.detach()

    expected = evaluate(reference, coord, spin)
    actual = evaluate(fused, coord, spin)
    assert actual[2].abs().max() > 1e-6
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-4)
    reversed_spin = evaluate(fused, coord, -spin)
    for got, want in zip(
        reversed_spin, (actual[0], actual[1], -actual[2]), strict=True
    ):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    inverted = evaluate(fused, -coord, spin)
    for got, want in zip(inverted, (actual[0], -actual[1], actual[2]), strict=True):
        torch.testing.assert_close(got, want, atol=3e-5, rtol=3e-4)
    zero = evaluate(fused, coord, torch.zeros_like(spin))
    torch.testing.assert_close(zero[2], torch.zeros_like(spin), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("backend", ["pt", "pt_expt"])
@pytest.mark.parametrize("autocast", [False, True])
def test_grid_pair_eval_checks_actual_operand_dtype(
    monkeypatch: pytest.MonkeyPatch, backend: str, autocast: bool
) -> None:
    """AMP grid products use the composition when operands exceed CUDA capability."""
    if not grid_pair.op_available():
        pytest.skip("the DPA4 CUDA grid operator is unavailable")
    for name in (
        "DP_TRITON_INFER",
        "DP_CUDA_INFER",
        "DP_CUTILE_INFER",
        "DP_CUTE_INFER",
        "DP_TRITON_TRAIN",
        "DP_CUDA_TRAIN",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    kwargs = {
        "lmax": 2,
        "mmax": 2,
        "kmax": 1,
        "channels": 32,
        "n_focus": 1,
        "mode": "self",
        "op_type": "branch",
        "layout": "ndfc",
        "parity": True,
        "trainable": True,
        "seed": 17,
    }
    if backend == "pt":
        from deepmd.pt.model.descriptor.sezm_nn.grid_net import SO3GridNet as PTGridNet

        model = PTGridNet(**kwargs, dtype=torch.float32).cuda().eval()
    else:
        model = SO3GridNet(**kwargs, precision="float32").cuda().eval()
    original = model._grid_pair_fn
    assert original is not None
    calls = []

    def fused(
        left: torch.Tensor,
        right: torch.Tensor,
        to_grid: torch.Tensor,
        from_grid: torch.Tensor,
        parity_order: torch.Tensor | None = None,
        even_count: int = 0,
    ) -> torch.Tensor:
        calls.append(left.dtype)
        return original(left, right, to_grid, from_grid, parity_order, even_count)

    model._grid_pair_fn = fused
    generator = torch.Generator(device="cuda").manual_seed(731)
    shape = (2, model.projector.coeff_dim // model.n_frames, 1, model.query_channels)
    query = torch.randn(shape, device="cuda", generator=generator)
    cotangent = torch.randn(
        2, shape[1], 1, model.output_channels, device="cuda", generator=generator
    )

    def evaluate() -> tuple[torch.Tensor, torch.Tensor]:
        value = query.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            output = model(value)
            loss = (output * cotangent).sum()
        gradient = torch.autograd.grad(loss, value)[0]
        return output.detach(), gradient.detach()

    actual = evaluate()
    assert bool(calls) is not autocast
    assert all(dtype == torch.float32 for dtype in calls)
    model._grid_pair_fn = None
    expected = evaluate()
    for value, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(
            value,
            reference,
            atol=0.0 if autocast else 3e-5,
            rtol=0.0 if autocast else 3e-4,
        )
