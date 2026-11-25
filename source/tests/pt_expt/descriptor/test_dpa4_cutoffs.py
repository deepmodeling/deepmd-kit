# SPDX-License-Identifier: LGPL-3.0-or-later
"""Per-block cutoff semantics of the DPA4 descriptor in pt_expt and dpmodel."""

from __future__ import (
    annotations,
)

import copy
from typing import (
    TYPE_CHECKING,
)

import numpy as np
import pytest
import torch

from deepmd.dpmodel.descriptor.dpa4 import DescrptDPA4 as DPDescriptor
from deepmd.pt_expt.descriptor.dpa4 import DescrptDPA4 as ExportDescriptor
from deepmd.pt_expt.utils import (
    env,
)

from ...dpa4_fixtures import (
    four_atom_spin_inputs,
    jitter_zero_arrays,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


def descriptor_config(rcut: float | list[float]) -> dict:
    """Return a small geometry-sensitive descriptor configuration."""
    return {
        "ntypes": 2,
        "sel": 8,
        "rcut": rcut,
        "channels": 8,
        "n_radial": 4,
        "lmax": 2,
        "l_schedule": [1, 2],
        "n_blocks": 2,
        "mixing_layers": 2,
        "random_gamma": False,
        "precision": "float64",
        "edge_norm": False,
        "density_floor": 8.0,
        "seed": 7,
    }


def system(scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four atoms whose edge lengths straddle the smaller cutoff."""
    coord = scale * torch.tensor(
        [[[0.0, 0.0, 0.0], [1.2, 0.3, 0.2], [3.2, 0.2, 0.4], [5.1, 0.3, 0.5]]],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    atype = torch.tensor([[0, 1, 0, 1]], device=env.DEVICE)
    nlist = torch.tensor(
        [[[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]], device=env.DEVICE
    )
    return coord, atype, nlist


def sensitive_data(config: dict, jitter_seed: int = 31) -> dict:
    """Return serialized weights whose zero residual projections are jittered.

    The residual output projections of a fresh descriptor start at zero, which
    hides every interaction path from the output.
    """
    descriptor = ExportDescriptor(**config)
    return jitter_zero_arrays(
        descriptor.serialize(), np.random.default_rng(jitter_seed)
    )


def sensitive_descriptor(rcut: float | list[float]) -> ExportDescriptor:
    """Return an evaluation-mode pt_expt descriptor with jittered weights."""
    data = sensitive_data(descriptor_config(rcut))
    return ExportDescriptor.deserialize(data).to(env.DEVICE).eval()


def capture_block_caches(model: ExportDescriptor, *inputs: torch.Tensor) -> list:
    """Run ``model`` and return the edge cache handed to each interaction block."""
    seen = []

    def capture(module: torch.nn.Module, args: tuple) -> None:
        seen.append(args[1])

    handles = [block.register_forward_pre_hook(capture) for block in model.blocks]
    try:
        model(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return seen


def finite_difference_gradient(
    function: Callable[[np.ndarray], float], point: np.ndarray, step: float = 1e-5
) -> np.ndarray:
    """Return the central-difference gradient of a scalar ``function`` at ``point``."""
    gradient = np.zeros_like(point)
    for index in np.ndindex(point.shape):
        shift = np.zeros_like(point)
        shift[index] = step
        upper = function(point + shift)
        lower = function(point - shift)
        gradient[index] = (upper - lower) / (2.0 * step)
    return gradient


def test_constant_cutoffs_preserve_scalar() -> None:
    """A constant list preserves the scalar state and output bit for bit."""
    data = sensitive_data(descriptor_config(4.0))
    constant_data = copy.deepcopy(data)
    constant_data["config"]["rcut"] = [4.0, 4.0]
    model = ExportDescriptor.deserialize(data).to(env.DEVICE).eval()
    constant = ExportDescriptor.deserialize(constant_data).to(env.DEVICE).eval()
    assert model.state_dict().keys() == constant.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(value, constant.state_dict()[name])
    coord, atype, nlist = system()
    coord.requires_grad_(True)
    expected = model(coord, atype, nlist)[0]
    actual = constant(coord, atype, nlist)[0]
    assert torch.equal(expected, actual)
    expected_force = torch.autograd.grad(expected.sum(), coord)[0]
    actual_force = torch.autograd.grad(actual.sum(), coord)[0]
    assert torch.equal(expected_force, actual_force)
    assert constant.serialize()["config"]["rcut"] == 4.0


@pytest.mark.parametrize("rcut", [[2.5, 4.0], [4.0, 2.5], [3.0, 4.0]])
def test_cutoff_backend_parity(rcut: list[float]) -> None:
    """Compact cutoff schedules agree in pt_expt and dpmodel, values and forces."""
    data = sensitive_data(descriptor_config(rcut))
    model = ExportDescriptor.deserialize(data).to(env.DEVICE).eval()
    assert model.serialize()["config"]["rcut"] == rcut
    assert model.get_rcut() == max(rcut)
    portable = DPDescriptor.deserialize(data)
    assert portable.get_rcut() == max(rcut)
    coord, atype, nlist = system()
    coord.requires_grad_(True)
    actual = model(coord, atype, nlist)[0]
    actual_grad = torch.autograd.grad(actual.square().sum(), coord)[0]
    coord_np, atype_np, nlist_np = (
        value.detach().cpu().numpy() for value in (coord, atype, nlist)
    )
    expected = portable(coord_np, atype_np, nlist_np)[0]
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=1e-10, atol=1e-11
    )
    expected_grad = finite_difference_gradient(
        lambda position: np.sum(portable(position, atype_np, nlist_np)[0] ** 2),
        coord_np,
    )
    np.testing.assert_allclose(
        actual_grad.cpu().numpy(), expected_grad, rtol=1e-6, atol=2e-9
    )
    assert actual_grad.abs().max() > 1e-7


@pytest.mark.parametrize("rcut", [[2.5, 4.0], [4.0, 2.5]])
def test_cutoff_jax_jit_parity(rcut: list[float]) -> None:
    """A cutoff schedule compiles under ``jax.jit`` with pt_expt values and forces."""
    pytest.importorskip("flax")
    from deepmd.jax.descriptor.dpa4 import DescrptDPA4 as JaxDescriptor
    from deepmd.jax.env import (
        jax,
        jnp,
    )

    data = sensitive_data(descriptor_config(rcut))
    model = ExportDescriptor.deserialize(data).to(env.DEVICE).eval()
    inputs = system()
    coord = inputs[0].detach().requires_grad_(True)
    expected = model(coord, *inputs[1:])[0]
    expected_grad = torch.autograd.grad(expected.square().sum(), coord)[0]
    portable = JaxDescriptor.deserialize(data)
    atype, nlist = (jnp.asarray(value.cpu().numpy()) for value in inputs[1:])

    def descriptor(position: jnp.ndarray) -> jnp.ndarray:
        return portable(position.reshape(1, -1), atype, nlist)[0]

    position = jnp.asarray(coord.detach().cpu().numpy())
    actual = jax.jit(descriptor)(position)
    actual_grad = jax.jit(jax.grad(lambda x: jnp.sum(descriptor(x) ** 2)))(position)
    np.testing.assert_allclose(
        actual, expected.detach().cpu().numpy(), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_allclose(
        actual_grad, expected_grad.cpu().numpy(), rtol=1e-9, atol=1e-10
    )


@pytest.mark.parametrize("spin_mode", ["exchange", "orbit"])
@pytest.mark.parametrize("rcut", [[1.6, 4.0], [4.0, 1.6]])
def test_spin_cutoff_backend_parity(spin_mode: str, rcut: list[float]) -> None:
    """Cutoff-local spin conditioning preserves coordinate and moment gradients."""
    config = descriptor_config(rcut)
    config.update(type_map=["O", "H"], use_spin=[True, False], spin_mode=spin_mode)
    data = sensitive_data(config, jitter_seed=51)
    coord_np, atype_np, spin_np, nlist_np = four_atom_spin_inputs()
    coord = torch.tensor(coord_np, device=env.DEVICE, requires_grad=True)
    spin = torch.tensor(spin_np, device=env.DEVICE, requires_grad=True)
    atype = torch.tensor(atype_np, device=env.DEVICE)
    nlist = torch.tensor(nlist_np, device=env.DEVICE)
    exported = ExportDescriptor.deserialize(data).to(env.DEVICE).eval()
    actual = exported(coord, atype, nlist, spin=spin)[0]
    actual_grad = torch.autograd.grad(actual.square().sum(), (coord, spin))
    portable = DPDescriptor.deserialize(data)
    expected = portable(coord_np, atype_np, nlist_np, spin=spin_np)[0]
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), expected, rtol=1e-10, atol=1e-11
    )
    expected_grad = (
        finite_difference_gradient(
            lambda position: np.sum(
                portable(position, atype_np, nlist_np, spin=spin_np)[0] ** 2
            ),
            coord_np,
        ),
        finite_difference_gradient(
            lambda moment: np.sum(
                portable(coord_np, atype_np, nlist_np, spin=moment)[0] ** 2
            ),
            spin_np,
        ),
    )
    for observed, target in zip(actual_grad, expected_grad, strict=True):
        np.testing.assert_allclose(observed.cpu().numpy(), target, rtol=1e-6, atol=2e-9)
    assert actual_grad[1].abs().max() > 1e-8


def test_blocks_receive_compact_edges() -> None:
    """The smaller block receives fewer edges and the floor of its own mass."""
    model = sensitive_descriptor([2.5, 4.0])
    with torch.no_grad():
        model.envelope_mass.copy_(model.envelope_mass.new_tensor([0.7, 3.0]))
    small, large = capture_block_caches(model, *system())
    assert small.src.shape[0] < large.src.shape[0]
    assert small.csr_cache is not large.csr_cache
    torch.testing.assert_close(
        small.inv_sqrt_deg[:, 0, 0], (small.deg + 8.0 * 0.7).rsqrt()
    )
    torch.testing.assert_close(
        large.inv_sqrt_deg[:, 0, 0], (large.deg + 8.0 * 3.0).rsqrt()
    )
    valid = small.edge_env[:, 0] != 0
    assert (small.edge_vec[valid].norm(dim=-1) < 2.5).all()
    assert not small.edge_env[-2:].any()
    assert not small.edge_rbf[-2:].any()


@pytest.mark.parametrize("rcut", [[2.5, 4.0], [4.0, 2.5]])
def test_empty_compact_neighborhood(rcut: list[float]) -> None:
    """An empty smaller edge set has finite first and parameter derivatives."""
    model = sensitive_descriptor(rcut).train()
    coord, atype, nlist = system(scale=4.0)
    coord.requires_grad_(True)
    output = model(coord, atype, nlist)[0]
    force = torch.autograd.grad(output.sum(), coord, create_graph=True)[0]
    (output.square().sum() + force.square().sum()).backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(force).all()
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.parametrize(
    "rcut",
    [[4.0], [], [0.0, 4.0], [-1.0, 4.0], [float("nan"), 4.0], [float("inf"), 4.0]],
)
def test_invalid_block_cutoffs(rcut: list[float]) -> None:
    """Cutoff sequences must match the block count and contain positive radii."""
    for backend in (DPDescriptor, ExportDescriptor):
        with pytest.raises(ValueError, match="rcut"):
            backend(**descriptor_config(rcut))


def test_repeated_cutoffs_share_cache() -> None:
    """Repeated radii share one compact edge cache across separated blocks."""
    config = descriptor_config([2.5, 4.0, 2.5])
    config.update(l_schedule=[1, 2, 1], n_blocks=3)
    model = ExportDescriptor(**config).to(env.DEVICE).eval()
    seen = capture_block_caches(model, *system())
    assert seen[0] is seen[2]
    assert seen[0] is not seen[1]


@pytest.mark.parametrize(
    "backend", [DPDescriptor, ExportDescriptor], ids=["dpmodel", "pt_expt"]
)
def test_zero_block_cutoff_contract(backend: type) -> None:
    """An interaction-free descriptor keeps a scalar radius for its seeds."""
    config = descriptor_config(4.0)
    config.update(n_blocks=0, l_schedule=[])
    assert backend(**config).get_rcut() == 4.0
    config["rcut"] = [4.0]
    with pytest.raises(ValueError, match="zero-block"):
        backend(**config)


def test_smaller_cutoff_force_continuity() -> None:
    """Analytic coordinate derivatives cross a compact-set boundary smoothly."""
    model = sensitive_descriptor([2.5, 4.0])
    atype = torch.tensor([[0, 1]], device=env.DEVICE)
    nlist = torch.tensor([[[1], [0]]], device=env.DEVICE)

    def energy(distance: float) -> tuple[torch.Tensor, torch.Tensor]:
        coord = torch.tensor(
            [[[0, 0, 0], [distance, 0.2, 0.1]]],
            dtype=torch.float64,
            device=env.DEVICE,
            requires_grad=True,
        )
        value = model(coord, atype, nlist)[0].square().sum()
        gradient = torch.autograd.grad(value, coord)[0][0, 1, 0]
        return value.detach(), gradient.detach()

    boundary = np.sqrt(2.5**2 - 0.2**2 - 0.1**2)
    center, derivative = energy(boundary)
    left, left_derivative = energy(boundary - 1e-5)
    right, right_derivative = energy(boundary + 1e-5)
    torch.testing.assert_close((right - left) / 2e-5, derivative, rtol=2e-6, atol=1e-8)
    torch.testing.assert_close(left_derivative, right_derivative, rtol=2e-4, atol=1e-7)
    assert torch.isfinite(center)


@pytest.mark.parametrize(
    "backend", [DPDescriptor, ExportDescriptor], ids=["dpmodel", "pt_expt"]
)
def test_contact_edge_inside_every_cutoff(backend: type) -> None:
    """A contact clamp cannot read separations beyond a smaller block cutoff."""
    config = descriptor_config([1.2, 4.0])
    config.update(type_map=["O", "H"], contact_clamp=0.75)
    with pytest.raises(ValueError, match="contact edge"):
        backend(**config)


def test_analytical_end_inside_every_cutoff() -> None:
    """The ZBL end radius must lie inside the smallest interaction cutoff."""
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )

    config = descriptor_config([1.3, 4.0])
    config.pop("ntypes")
    with pytest.raises(ValueError, match="analytical potential"):
        get_model(
            {
                "type": "dpa4",
                "type_map": ["O", "H"],
                "bridging_method": "zbl",
                "descriptor": {"type": "dpa4", **config},
                "fitting_net": {
                    "type": "dpa4_ener",
                    "neuron": [8],
                    "precision": "float64",
                },
            }
        )


def _assert_compiled_cutoff_model(
    rcut: list[float], *, fused: bool = False, large: bool = False
) -> None:
    """Compare compiled force training across batch and compact-edge counts."""
    from deepmd.pt_expt.model.get_model import (
        get_model,
    )
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
    )

    config = {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {"type": "dpa4", **descriptor_config(rcut)},
        "fitting_net": {
            "type": "dpa4_ener",
            "neuron": [8],
            "precision": "float64",
            "seed": 11,
        },
    }
    config["descriptor"].pop("ntypes")
    if fused:
        config["descriptor"].update(
            precision="float32",
            channels=32,
            focus_dim=32,
            use_amp=False,
        )
        config["fitting_net"]["precision"] = "float32"
    reference = get_model(copy.deepcopy(config)).to(env.DEVICE).train()
    generator = torch.Generator(device=env.DEVICE).manual_seed(15)
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(
                0.03
                * torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=env.DEVICE,
                    generator=generator,
                )
            )
    model = get_model(config).to(env.DEVICE).train()
    model.load_state_dict(reference.state_dict())
    if fused:
        for block in model.get_descriptor().blocks:
            assert block.so2_conv._cuda_value_train is not None
            assert block.so2_conv._segment_softmax_fn is not None
    compiled = _CompiledModel(model, structure_key=(1,)).train()
    cases = [(1, 1.0), (2, 1.3), (3, 4.0)]
    compact_edge_counts = []
    if large:
        cases.append((6000, 1.0))

        def capture_edges(module: torch.nn.Module, args: tuple) -> None:
            compact_edge_counts.append(args[1].src.shape[0] - 2)

        reference.get_descriptor().blocks[-1].register_forward_pre_hook(capture_edges)
    for frames, scale in cases:
        coord, atype, _ = system(scale)
        coord, atype = coord.repeat(frames, 1, 1), atype.repeat(frames, 1)
        outputs = []
        gradients = []
        for candidate in (reference, compiled):
            candidate.zero_grad(set_to_none=True)
            output = candidate(coord, atype)
            # The per-frame objective gives gradient tolerances consistent units.
            loss = (
                output["energy"].square().sum() + output["force"].square().sum()
            ) / frames
            loss.backward()
            outputs.append(output)
            gradients.append(
                [
                    None if p.grad is None else p.grad.detach().clone()
                    for p in candidate.parameters()
                ]
            )
        for key in ("energy", "force", "virial"):
            torch.testing.assert_close(
                outputs[0][key],
                outputs[1][key],
                rtol=2e-4 if fused else 1e-8,
                atol=2e-5 if fused else 1e-9,
            )
        for expected, actual in zip(*gradients, strict=True):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=2e-3 if fused else 1e-7,
                    atol=1e-4 if fused else 1e-8,
                )
    if large:
        assert compact_edge_counts[-1] > 32_766


@pytest.mark.parametrize("rcut", [[2.5, 4.0], [4.0, 2.5]])
def test_compiled_cutoff_model(rcut: list[float]) -> None:
    """Compiled force training preserves gradients across batch and edge counts."""
    _assert_compiled_cutoff_model(rcut)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rcut", [[2.5, 4.0], [4.0, 2.5]])
def test_compiled_fused_cutoff_model(
    monkeypatch: pytest.MonkeyPatch, rcut: list[float]
) -> None:
    """Compiled CUDA/Triton force training keeps the data-dependent compact axis."""
    monkeypatch.setenv("DP_CUDA_TRAIN", "1")
    monkeypatch.setenv("DP_TRITON_TRAIN", "1")
    _assert_compiled_cutoff_model(rcut, fused=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_large_cutoff_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """A compact training graph grows past the int16 sorting boundary."""
    monkeypatch.setenv("DP_CUDA_TRAIN", "1")
    monkeypatch.setenv("DP_TRITON_TRAIN", "1")
    _assert_compiled_cutoff_model([4.0, 2.5], fused=True, large=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("path", ["cuda", "triton", "cuda-triton"])
@pytest.mark.parametrize("rcut", [4.0, [4.0, 2.5], [2.5, 4.0]])
def test_compact_fused_paths(
    monkeypatch: pytest.MonkeyPatch,
    training: bool,
    path: str,
    rcut: float | list[float],
) -> None:
    """Actual accelerated dispatch preserves compact-edge force derivatives."""
    for gate in (
        "DP_CUDA_TRAIN",
        "DP_TRITON_TRAIN",
        "DP_CUDA_INFER",
        "DP_TRITON_INFER",
    ):
        monkeypatch.setenv(gate, "0")
    config = descriptor_config(rcut)
    config.update(precision="float32", channels=32, focus_dim=32, use_amp=False)
    data = sensitive_data(config, jitter_seed=41)
    data["config"]["use_amp"] = False
    reference = ExportDescriptor.deserialize(data).to(env.DEVICE).train(training)
    suffix = "TRAIN" if training else "INFER"
    if "cuda" in path:
        monkeypatch.setenv(f"DP_CUDA_{suffix}", "1" if training else "2")
    if "triton" in path:
        monkeypatch.setenv(f"DP_TRITON_{suffix}", "1")
    fused = ExportDescriptor.deserialize(data).to(env.DEVICE).train(training)
    for block in fused.blocks:
        convolution = block.so2_conv
        if "cuda" in path:
            binding = (
                convolution._cuda_value_train if training else convolution._cuda_conv_fn
            )
            assert binding is not None
        if "triton" in path:
            assert convolution._flash_atten_fn is not None
            assert convolution._segment_softmax_fn is not None
    outputs, forces, gradients = [], [], []
    for descriptor in (reference, fused):
        coord, atype, nlist = system()
        coord = coord.to(torch.float32).detach().requires_grad_(True)
        output = descriptor(coord, atype, nlist)[0]
        force = torch.autograd.grad(
            output.square().sum(), coord, create_graph=training
        )[0]
        outputs.append(output)
        forces.append(force)
        if training:
            (output.square().sum() + force.square().sum()).backward()
            gradients.append(
                [
                    None if p.grad is None else p.grad.detach().clone()
                    for p in descriptor.parameters()
                ]
            )
    torch.testing.assert_close(outputs[1], outputs[0], rtol=5e-4, atol=2e-5)
    torch.testing.assert_close(forces[1], forces[0], rtol=5e-4, atol=2e-5)
    if training:
        for expected, actual in zip(*gradients, strict=True):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected, rtol=2e-3, atol=1e-4)
