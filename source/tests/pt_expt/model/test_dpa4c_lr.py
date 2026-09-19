# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the DPA4C-LR non-periodic latent-charge model."""

import copy

import numpy as np
import pytest
import torch

from deepmd.infer import (
    DeepPot,
)
from deepmd.pt_expt.fitting.dpa4c_lr import (
    DPA4CLRFitting,
)
from deepmd.pt_expt.model.dpa4c_lr_model import (
    E2_PER_ANGSTROM_TO_EV,
    _les_kernel_from_squared_distance,
    _SOGKernel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils import (
    env,
)

from ..compile_utils import (
    REQUIRES_SUPPORTED_COMPILE,
)


def _config(lr_kernel: str = "les") -> dict:
    """Minimal float64 DPA4C-LR config for unit tests."""
    config = {
        "type": "dpa4c_lr",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "dpa4c",
            "rcut": 3.0,
            "channels": 8,
            "lmax": 2,
            "n_radial": 4,
            "precision": "float64",
            "seed": 17,
        },
        "fitting_net": {
            "type": "dpa4c_lr",
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 19,
            "dim_out_lr": 1,
            "neuron_lr": [8, 8],
            "use_charge_constraint": True,
            "lr_kernel": "les",
            "les_alpha": 1.0,
        },
    }
    if lr_kernel == "sog":
        config["fitting_net"].update(
            {
                "dim_out_lr": 3,
                "use_charge_constraint": False,
                "lr_kernel": "sog",
                "amp": [0.7, -0.2, 0.1],
                "bandwidth": [0.8, 1.7, 3.2],
            }
        )
    return config


def _build_model(train: bool = False, lr_kernel: str = "les") -> torch.nn.Module:
    """Build a small DPA4C-LR model on the default test device."""
    model = get_model(_config(lr_kernel)).to(env.DEVICE)
    if train:
        model.train()
    else:
        model.eval()
    return model


def _make_batch(
    nlocs: tuple[int, ...],
    seed: int = 0,
    box: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded rectangular batch."""
    rng = np.random.default_rng(seed)
    pad_nloc = max(nlocs)
    padded_coord = np.zeros((len(nlocs), pad_nloc, 3))
    padded_atype = np.full((len(nlocs), pad_nloc), -1, dtype=np.int64)
    for index, nloc in enumerate(nlocs):
        padded_coord[index, :nloc] = rng.uniform(0.0, 6.0, (nloc, 3))
        padded_atype[index, :nloc] = rng.integers(0, 2, nloc)
    return padded_coord, padded_atype


def test_sog_kernel_matches_closed_form() -> None:
    """The dense SOG kernel contains only distinct-atom pairs."""
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]],
        dtype=torch.float64,
        device=env.DEVICE,
    )
    charges = torch.tensor(
        [[[1.0, 2.0], [-0.5, 1.5]]], dtype=torch.float64, device=env.DEVICE
    )
    amp = torch.tensor([0.7, -0.2], dtype=torch.float64, device=env.DEVICE)
    bandwidth = torch.tensor([0.8, 1.7], dtype=torch.float64, device=env.DEVICE)

    kernel_at_r = torch.sum(amp * torch.exp(-0.5 * 4.0 / bandwidth.square()))
    pair = torch.dot(charges[0, 0], charges[0, 1]) * kernel_at_r
    energy = _SOGKernel()(positions, charges, amp, bandwidth)
    torch.testing.assert_close(energy[0], pair * E2_PER_ANGSTROM_TO_EV)


def test_les_masked_zero_edge_has_finite_gradient() -> None:
    """A zero-length guard edge must not inject NaN into the LES force."""
    edge_vec = torch.zeros(
        (1, 3), dtype=torch.float64, device=env.DEVICE, requires_grad=True
    )
    alpha = torch.tensor(1.0, dtype=torch.float64, device=env.DEVICE)
    edge_mask = torch.zeros(1, dtype=torch.float64, device=env.DEVICE)
    r_sq = torch.sum(edge_vec * edge_vec, dim=-1)
    energy = (_les_kernel_from_squared_distance(r_sq, alpha) * edge_mask).sum()
    (gradient,) = torch.autograd.grad(energy, edge_vec)
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) == 0


def test_energy_force_are_finite_and_phantom_atoms_are_force_free() -> None:
    """A padded batch produces finite outputs and zero force on phantom atoms."""
    torch.manual_seed(1234)
    model = _build_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    coord, atype = _make_batch((4, 7, 3))
    result = model(
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.long, device=env.DEVICE),
    )
    for key in ("energy", "force", "atom_energy"):
        assert torch.isfinite(result[key]).all(), key
    phantom = torch.tensor(atype, device=env.DEVICE) < 0
    assert bool(phantom.any()), "fixture must exercise padding"
    assert bool(torch.all(result["force"][phantom] == 0.0))


def test_frames_do_not_interact_across_batch() -> None:
    """The dense LR kernel must not couple atoms in different frames."""
    torch.manual_seed(1234)
    model = _build_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    nlocs = (4, 7, 3)
    coord, atype = _make_batch(nlocs)

    def run(
        batch_coord: np.ndarray, batch_atype: np.ndarray
    ) -> dict[str, torch.Tensor]:
        return model(
            torch.tensor(batch_coord, dtype=torch.float64, device=env.DEVICE),
            torch.tensor(batch_atype, dtype=torch.long, device=env.DEVICE),
        )

    batched = run(coord, atype)
    for index, nloc in enumerate(nlocs):
        alone = run(
            coord[index : index + 1, :nloc],
            atype[index : index + 1, :nloc],
        )
        torch.testing.assert_close(
            batched["energy"].reshape(-1)[index],
            alone["energy"].reshape(-1)[0],
            atol=1e-12,
            rtol=1e-12,
        )
        torch.testing.assert_close(
            batched["force"][index, :nloc],
            alone["force"][0],
            atol=1e-12,
            rtol=1e-12,
        )


@pytest.mark.parametrize("lr_kernel", ["les", "sog"])
def test_force_matches_finite_difference(lr_kernel: str) -> None:
    """The public model force agrees with a central finite-difference stencil."""
    torch.manual_seed(1234)
    model = _build_model(lr_kernel=lr_kernel)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    rng = np.random.default_rng(1)
    coord = rng.uniform(0.0, 6.0, (1, 5, 3))
    atype = rng.integers(0, 2, (1, 5)).astype(np.int64)
    atype_t = torch.tensor(atype, dtype=torch.long, device=env.DEVICE)

    def energy(x: torch.Tensor) -> float:
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            ret = model(x, atype_t)
        return ret["energy"].item()

    coord_t = torch.tensor(coord, dtype=torch.float64, device=env.DEVICE)
    force = model(coord_t, atype_t)["force"].detach().cpu().numpy().reshape(-1, 3)

    eps = 1e-5
    force_fd = np.zeros_like(force)
    for i in range(coord.shape[1]):
        for d in range(3):
            cp = coord.copy()
            cp[0, i, d] += eps
            cm = coord.copy()
            cm[0, i, d] -= eps
            Ep = energy(torch.tensor(cp, dtype=torch.float64, device=env.DEVICE))
            Em = energy(torch.tensor(cm, dtype=torch.float64, device=env.DEVICE))
            force_fd[i, d] = -(Ep - Em) / (2 * eps)

    np.testing.assert_allclose(force, force_fd, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("lr_kernel", ["les", "sog"])
def test_atom_virial_sums_to_total_virial(lr_kernel: str) -> None:
    """Per-atom virials include the explicit LR geometry contribution."""
    torch.manual_seed(1234)
    model = _build_model(lr_kernel=lr_kernel)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    coord, atype = _make_batch((4, 7, 3), seed=31)
    result = model(
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.long, device=env.DEVICE),
        do_atomic_virial=True,
    )
    torch.testing.assert_close(
        result["atom_virial"].sum(dim=1),
        result["virial"],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("lr_kernel", ["les", "sog"])
def test_ragged_matches_padded(lr_kernel: str) -> None:
    """Ragged and padded layouts must describe the same batch."""
    torch.manual_seed(1234)
    model = _build_model(lr_kernel=lr_kernel)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    nlocs = (4, 7, 3)
    coord, atype = _make_batch(nlocs)

    padded = model(
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.long, device=env.DEVICE),
    )
    flat_coord = torch.tensor(
        np.concatenate([coord[i, :nloc] for i, nloc in enumerate(nlocs)]),
        dtype=torch.float64,
        device=env.DEVICE,
    )
    flat_atype = torch.tensor(
        np.concatenate([atype[i, :nloc] for i, nloc in enumerate(nlocs)]),
        dtype=torch.long,
        device=env.DEVICE,
    )
    n_node = torch.tensor(nlocs, dtype=torch.long, device=env.DEVICE)
    ragged = model.forward_ragged(flat_coord, flat_atype, n_node)

    torch.testing.assert_close(padded["energy"], ragged["energy"])
    offset = 0
    for index, nloc in enumerate(nlocs):
        torch.testing.assert_close(
            padded["force"][index, :nloc],
            ragged["force"][offset : offset + nloc],
            atol=1e-12,
            rtol=1e-12,
        )
        offset += nloc
    assert offset == ragged["force"].shape[0]


def test_charge_constraint_zero_target_preserves_total_charge() -> None:
    """With no charge_spin or fparam, constraint drives total latent charge to 0."""
    torch.manual_seed(1234)
    model = _build_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    fitting = model.get_fitting_net()
    nf, nloc = 2, 5
    q = torch.randn(nf * nloc, 1, dtype=torch.float64, device=env.DEVICE)
    n_node = torch.tensor([nloc] * nf, dtype=torch.long, device=env.DEVICE)
    from deepmd.dpmodel.utils.neighbor_graph import (
        frame_id_from_n_node,
    )

    frame_id = frame_id_from_n_node(n_node, n_total=q.shape[0])
    q_out = model._constrain_charges_flat(
        q, frame_id, n_node, fparam=None, charge_spin=None
    )
    for f in range(nf):
        assert torch.isclose(
            q_out[frame_id == f, 0].sum(),
            torch.zeros((), dtype=q_out.dtype, device=q_out.device),
        )


def test_periodic_box_is_rejected() -> None:
    """DPA4C-LR must reject non-zero simulation cells."""
    model = _build_model()
    coord = torch.zeros((1, 3, 3), dtype=torch.float64, device=env.DEVICE)
    atype = torch.zeros((1, 3), dtype=torch.long, device=env.DEVICE)
    box = torch.eye(3, dtype=torch.float64, device=env.DEVICE).reshape(1, 9)
    with pytest.raises(NotImplementedError):
        model(coord, atype, box=box)


def test_periodic_training_data_is_rejected_before_stats() -> None:
    """Periodic sampled frames fail fast with a clear error before stats."""
    model = _build_model()

    def sampled() -> list[dict]:
        return [
            {
                "coord": np.zeros((1, 3, 3)),
                "atype": np.zeros((1, 3), dtype=np.int64),
                "box": np.eye(3).reshape(1, 9),
            }
        ]

    with pytest.raises(NotImplementedError, match="non-periodic"):
        model.compute_or_load_stat(sampled)


@pytest.mark.parametrize("lr_kernel", ["les", "sog"])
def test_pt_checkpoint_deep_eval_matches_direct_forward(
    tmp_path, lr_kernel: str
) -> None:
    """Checkpoint inference builds and passes the required long-range graph."""
    torch.manual_seed(1234)
    model = _build_model(lr_kernel=lr_kernel)
    checkpoint = tmp_path / f"dpa4c_lr_{lr_kernel}.pt"
    wrapper = ModelWrapper(model, model_params=copy.deepcopy(_config(lr_kernel)))
    torch.save({"model": wrapper.state_dict()}, checkpoint)

    coord, atype = _make_batch((4, 4), seed=29)
    coord_t = torch.tensor(coord, dtype=torch.float64, device=env.DEVICE)
    atype_t = torch.tensor(atype, dtype=torch.long, device=env.DEVICE)
    expected = model(coord_t, atype_t, do_atomic_virial=True)

    dp = DeepPot(str(checkpoint), auto_batch_size=False)
    output_names = ("energy", "force", "virial", "atom_energy", "atom_virial")
    actual = dict(
        zip(
            output_names,
            dp.eval(coord, None, atype, atomic=True, mixed_type=True),
            strict=True,
        )
    )
    for name in output_names:
        np.testing.assert_allclose(
            actual[name],
            expected[name].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=name,
        )


@pytest.mark.parametrize("lr_kernel", ["les", "sog"])
def test_graph_training_gradient_reaches_lr_parameters(lr_kernel: str) -> None:
    """Backprop through the graph lower reaches descriptor and LR fitting params."""
    torch.manual_seed(1234)
    model = _build_model(train=True, lr_kernel=lr_kernel)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)

    coord, atype = _make_batch((4, 5))
    result = model(
        torch.tensor(coord, dtype=torch.float64, device=env.DEVICE),
        torch.tensor(atype, dtype=torch.long, device=env.DEVICE),
    )
    loss = result["energy"].square().mean() + result["force"].square().mean()
    loss.backward()

    fitting = model.get_fitting_net()
    for name, parameter in fitting.named_parameters():
        assert parameter.grad is not None, f"fitting parameter {name} has no grad"
        assert torch.isfinite(parameter.grad).all(), f"{name} grad non-finite"
    descriptor = model.get_descriptor()
    grads = {name: parameter.grad for name, parameter in descriptor.named_parameters()}
    assert grads, "descriptor has parameters"
    for name, gradient in grads.items():
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_sog_parameters_serialize_and_remain_trainable() -> None:
    """SOG amplitudes and widths survive fitting serialization as Parameters."""
    fitting = _build_model(lr_kernel="sog").get_fitting_net()
    named_parameters = dict(fitting.named_parameters())
    assert "amp" in named_parameters
    assert "bandwidth" in named_parameters
    assert "les_alpha" not in named_parameters

    restored = DPA4CLRFitting.deserialize(fitting.serialize()).to(env.DEVICE)
    assert restored.lr_kernel == "sog"
    assert restored.dim_out_lr == 3
    restored_parameters = dict(restored.named_parameters())
    assert "amp" in restored_parameters
    assert "bandwidth" in restored_parameters
    torch.testing.assert_close(restored.amp, fitting.amp)
    torch.testing.assert_close(restored.bandwidth, fitting.bandwidth)


def test_les_checkpoint_does_not_require_sog_parameters() -> None:
    """LES state loading accepts checkpoints from before and during SOG rollout."""
    fitting = _build_model(lr_kernel="les").get_fitting_net()
    state_dict = fitting.state_dict()
    assert "amp" not in state_dict
    assert "bandwidth" not in state_dict

    # Simulate the short-lived format that stored unused SOG buffers in LES.
    transitional_state = state_dict.copy()
    transitional_state["amp"] = torch.ones(12, device=env.DEVICE)
    transitional_state["bandwidth"] = torch.ones(12, device=env.DEVICE)

    restored = _build_model(lr_kernel="les").get_fitting_net()
    restored.load_state_dict(state_dict, strict=True)
    restored.load_state_dict(transitional_state, strict=True)


def test_sog_graph_matches_dense_legacy_path() -> None:
    """SOG graph and rectangular dense paths produce the same observables."""
    torch.manual_seed(1234)
    model = _build_model(lr_kernel="sog")
    coord, atype = _make_batch((5, 5), seed=37)
    coord_t = torch.tensor(coord, dtype=torch.float64, device=env.DEVICE)
    atype_t = torch.tensor(atype, dtype=torch.long, device=env.DEVICE)

    graph = model(coord_t, atype_t)
    # The dense (nlist) fallback must agree with the graph lower.
    model.neighbor_graph_method = "legacy"
    dense = model(coord_t, atype_t)
    for name in ("energy", "force", "virial"):
        torch.testing.assert_close(graph[name], dense[name], rtol=1e-10, atol=1e-10)


@REQUIRES_SUPPORTED_COMPILE
@pytest.mark.parametrize("lr_kernel", ["les", "sog"])
def test_compiled_lower_matches_eager(lr_kernel: str) -> None:
    """The compiled graph lower must reproduce the eager energy and force."""
    from deepmd.pt_expt.train.training import (
        _CompiledModel,
        _get_model_structure_key,
    )

    torch.manual_seed(0)
    model = _build_model(lr_kernel=lr_kernel)
    compiled = _CompiledModel(model, _get_model_structure_key(model))
    compiled.eval()

    nlocs = (4, 7, 3)
    coord, atype = _make_batch(nlocs)
    coord_t = torch.tensor(coord, dtype=torch.float64, device=env.DEVICE)
    atype_t = torch.tensor(atype, dtype=torch.long, device=env.DEVICE)

    expected = model(coord_t, atype_t)
    got = compiled(coord_t, atype_t, None)

    torch.testing.assert_close(got["energy"], expected["energy"])
    torch.testing.assert_close(got["force"], expected["force"])
