# SPDX-License-Identifier: LGPL-3.0-or-later
"""Long-range latent-charge modules on the DPA4C graph lower (pt_expt)."""

import copy

import numpy as np
import pytest
import torch

from deepmd.dpmodel.long_range import (
    COULOMB_CONSTANT,
)
from deepmd.dpmodel.utils.neighbor_graph import (
    ALL_PAIRS,
    build_pair_graph,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.training import (
    _CompiledModel,
    _get_model_structure_key,
    model_change_out_bias,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)

from ..compile_utils import (
    REQUIRES_SUPPORTED_COMPILE,
)

KERNELS = {
    "les": {"type": "les", "alpha": 0.8},
    "sog": {"type": "sog", "amplitude": [0.7, -0.2, 0.1], "width": [0.8, 1.7, 3.2]},
}
CHANNELS = {"les": 1, "sog": 3}
NEUTRAL = {"source": "neutral"}


def _config(kernel: str = "les", constraint: dict | None = NEUTRAL) -> dict:
    """Small float64 DPA4C model with a latent-charge long-range module."""
    return {
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
            "type": "ener",
            "neuron": [16, 16],
            "precision": "float64",
            "seed": 19,
        },
        "long_range": {
            "type": "latent_charge",
            "channels": CHANNELS[kernel],
            "charge_net": {"neuron": [8, 8], "precision": "float64", "seed": 23},
            "kernel": copy.deepcopy(KERNELS[kernel]),
            "charge_constraint": copy.deepcopy(constraint),
        },
    }


def _model(kernel: str = "les", **kwargs: object) -> torch.nn.Module:
    return get_model(_config(kernel, **kwargs)).to(env.DEVICE).eval()


def _tensor(x: np.ndarray, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype, device=env.DEVICE)


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Three frames of 6, 3 and 5 real atoms, padded to a common width."""
    rng = np.random.default_rng(seed)
    coord = rng.uniform(0.0, 6.0, (3, 6, 3))
    atype = np.array(
        [[0, 1, 0, 1, 1, 0], [1, 0, 0, -1, -1, -1], [0, 0, 1, 1, 0, -1]],
        dtype=np.int64,
    )
    return _tensor(coord), _tensor(atype, torch.int64)


def _set_constant_charges(model: torch.nn.Module, charges: list[float]) -> None:
    """Zero every network weight and give the atoms of each type a fixed charge.

    The energy is then the long-range pair sum alone, of charges that do not
    depend on the geometry; the kernel keeps its configured parameters.
    """
    with torch.no_grad():
        for name, parameter in model.atomic_model.named_parameters():
            if ".kernel." not in f".{name}":
                parameter.zero_()
        bias = model.atomic_model.long_range.charge_net.bias_atom_e
        bias.zero_()
        bias[:, 0] = torch.as_tensor(charges, dtype=bias.dtype)


def _reference_energy(coord: np.ndarray, charge: np.ndarray, kernel: str) -> float:
    """Direct pair sum of the latent-charge energy of one frame, in eV."""
    energy = 0.0
    for i in range(len(coord)):
        for j in range(i + 1, len(coord)):
            r = np.linalg.norm(coord[i] - coord[j])
            if kernel == "les":
                from scipy.special import (
                    erf,
                )

                k = erf(0.8 * r) / r
            else:
                amplitude = np.array([0.7, -0.2, 0.1])
                width = np.array([0.8, 1.7, 3.2])
                k = np.sum(amplitude * np.exp(-(r**2) / (2 * width**2)))
            energy += COULOMB_CONSTANT * float(charge[i] @ charge[j]) * k
    return energy


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_energy_matches_direct_pair_sum(kernel: str) -> None:
    """With fixed charges the energy is the documented pair sum, attributed half per atom."""
    model = _model(kernel, constraint=None)
    _set_constant_charges(model, [1.0, -0.5])
    coord = np.array([[[0.0, 0.0, 0.0], [1.3, 0.0, 0.0], [0.0, 2.1, 0.4]]])
    atype = np.array([[0, 1, 1]])
    out = model(_tensor(coord), _tensor(atype, torch.int64))
    charge = np.zeros((3, CHANNELS[kernel]))
    charge[:, 0] = [1.0, -0.5, -0.5]
    expected = _reference_energy(coord[0], charge, kernel)
    np.testing.assert_allclose(out["energy"].item(), expected, rtol=1e-12)
    np.testing.assert_allclose(
        out["atom_energy"].sum().item(), expected, rtol=1e-12, atol=1e-14
    )


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_force_virial_hessian_match_finite_differences(kernel: str) -> None:
    """Force, affine-strain virial and Hessian are derivatives of the complete energy."""
    model = _model(kernel)
    # Atoms within the cutoff of each other make the charges depend on the
    # geometry, so the derivatives hold the response of the charges.
    rng = np.random.default_rng(3)
    coord = rng.uniform(0.0, 2.5, (1, 4, 3))
    atype = np.array([[0, 1, 0, 1]])
    coord_t, atype_t = _tensor(coord), _tensor(atype, torch.int64)
    out = model(coord_t, atype_t)

    def energy(c: np.ndarray) -> float:
        return model(_tensor(c), atype_t)["energy"].item()

    h = 1e-5
    force_fd = np.zeros((4, 3))
    for i in range(4):
        for d in range(3):
            plus, minus = coord.copy(), coord.copy()
            plus[0, i, d] += h
            minus[0, i, d] -= h
            force_fd[i, d] = -(energy(plus) - energy(minus)) / (2 * h)
    np.testing.assert_allclose(
        out["force"][0].detach().cpu().numpy(), force_fd, atol=1e-8
    )

    # The virial is -dE/d(strain) at zero strain, for x -> (1 + eps) x.
    virial_fd = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            strain = np.zeros((3, 3))
            strain[a, b] = h
            plus = coord @ (np.eye(3) + strain)
            minus = coord @ (np.eye(3) - strain)
            virial_fd[a, b] = -(energy(plus) - energy(minus)) / (2 * h)
    np.testing.assert_allclose(
        out["virial"][0].detach().cpu().numpy().reshape(3, 3), virial_fd, atol=1e-8
    )

    hessian_model = _model(kernel)
    hessian_model.enable_hessian()
    hessian = hessian_model(coord_t, atype_t)["hessian"].detach().cpu().numpy()
    hessian = hessian.reshape(12, 12)
    hessian_fd = np.zeros((12, 12))
    for k in range(12):
        plus, minus = coord.copy(), coord.copy()
        plus.reshape(-1)[k] += h
        minus.reshape(-1)[k] -= h
        f_plus = model(_tensor(plus), atype_t)["force"].detach().cpu().numpy()
        f_minus = model(_tensor(minus), atype_t)["force"].detach().cpu().numpy()
        hessian_fd[:, k] = -(f_plus - f_minus).reshape(-1) / (2 * h)
    np.testing.assert_allclose(hessian, hessian_fd, atol=1e-6)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-10)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_atomic_reductions_and_padding(kernel: str) -> None:
    """Atomic energy and virial sum to the frame values; padding atoms are inert."""
    model = _model(kernel)
    coord, atype = _batch()
    out = model(coord, atype, do_atomic_virial=True)
    torch.testing.assert_close(out["atom_energy"].sum(1), out["energy"])
    torch.testing.assert_close(out["atom_virial"].sum(1), out["virial"])
    assert out["force"][1, 3:].abs().max() == 0
    moved = coord.clone()
    moved[1, 3:] += 50.0
    moved_out = model(moved, atype)
    torch.testing.assert_close(moved_out["energy"], out["energy"], rtol=0, atol=0)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_frames_are_independent(kernel: str) -> None:
    """A frame keeps its energy and force whatever other frames are batched with it."""
    model = _model(kernel)
    coord, atype = _batch()
    batched = model(coord, atype)
    for frame, nloc in enumerate((6, 3, 5)):
        alone = model(coord[frame : frame + 1, :nloc], atype[frame : frame + 1, :nloc])
        torch.testing.assert_close(alone["energy"][0], batched["energy"][frame])
        torch.testing.assert_close(
            alone["force"][0], batched["force"][frame, :nloc], atol=1e-12, rtol=0
        )


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_ragged_matches_rectangular(kernel: str) -> None:
    """The flat ragged entry evaluates the same model as the padded one."""
    model = _model(kernel)
    coord, atype = _batch()
    padded = model(coord, atype)
    keep = atype.reshape(-1) >= 0
    ragged = model.forward_ragged(
        coord.reshape(-1, 3)[keep],
        atype.reshape(-1)[keep],
        torch.tensor([6, 3, 5], device=env.DEVICE),
    )
    torch.testing.assert_close(ragged["energy"], padded["energy"])
    torch.testing.assert_close(ragged["force"], padded["force"].reshape(-1, 3)[keep])


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_empty_and_single_atom_frames(kernel: str) -> None:
    """Frames without pairs contribute no long-range energy and stay finite."""
    model = _model(kernel)
    coord = _tensor(np.random.default_rng(5).uniform(0.0, 3.0, (3, 2, 3)))
    atype = _tensor(np.array([[0, -1], [1, 0], [-1, -1]]), torch.int64)
    out = model(coord, atype)
    assert torch.isfinite(out["energy"]).all()
    assert torch.isfinite(out["force"]).all()
    single = model(coord[:1, :1], atype[:1, :1])
    torch.testing.assert_close(single["energy"][0], out["energy"][0])


def test_permutation_and_rigid_motion_invariance() -> None:
    """The energy is invariant under atom permutation, translation and rotation."""
    model = _model("sog")
    coord, atype = _batch()
    coord, atype = coord[:1], atype[:1]
    out = model(coord, atype)
    perm = torch.tensor([3, 0, 5, 1, 4, 2], device=env.DEVICE)
    permuted = model(coord[:, perm], atype[:, perm])
    torch.testing.assert_close(permuted["energy"], out["energy"])
    torch.testing.assert_close(permuted["force"], out["force"][:, perm])
    rotation, _ = torch.linalg.qr(
        _tensor(np.random.default_rng(13).normal(size=(3, 3)))
    )
    moved = model(coord @ rotation.T + 3.7, atype)
    torch.testing.assert_close(moved["energy"], out["energy"])
    torch.testing.assert_close(moved["force"], out["force"] @ rotation.T)


@pytest.mark.parametrize("source", ["charge_spin", "fparam"])
def test_constraint_default_matches_explicit_target(source: str) -> None:
    """An omitted frame condition takes its default for the constraint as well."""
    config = _config("les", constraint={"source": source, "fparam_index": 0})
    if source == "charge_spin":
        config["descriptor"]["add_chg_spin_ebd"] = True
        config["descriptor"]["default_chg_spin"] = [2.0, 1.0]
        explicit = {"charge_spin": _tensor(np.array([[2.0, 1.0], [2.0, 1.0]]))}
        other = {"charge_spin": _tensor(np.array([[0.0, 1.0], [0.0, 1.0]]))}
    else:
        config["fitting_net"]["numb_fparam"] = 1
        config["fitting_net"]["default_fparam"] = [2.0]
        explicit = {"fparam": _tensor(np.array([[2.0], [2.0]]))}
        other = {"fparam": _tensor(np.array([[0.0], [0.0]]))}
    model = get_model(config).to(env.DEVICE).eval()
    coord, atype = _batch()
    coord, atype = coord[:2], atype[:2]
    implicit_out = model(coord, atype)
    explicit_out = model(coord, atype, **explicit)
    torch.testing.assert_close(implicit_out["energy"], explicit_out["energy"])
    torch.testing.assert_close(implicit_out["force"], explicit_out["force"])
    other_out = model(coord, atype, **other)
    assert (other_out["energy"] - implicit_out["energy"]).abs().max() > 1e-6


@pytest.mark.parametrize("constraint", [None, NEUTRAL])
def test_pair_exclusion_removes_the_interaction(constraint: dict | None) -> None:
    """Pairs of excluded types do not interact; their atoms keep their charges."""
    config = _config("les", constraint=constraint)
    config["pair_exclude_types"] = [[0, 1]]
    model = get_model(config).to(env.DEVICE).eval()
    _set_constant_charges(model, [1.0, -1.0])
    coord = _tensor(np.array([[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [8.0, 0.0, 0.0]]]))
    atype = _tensor(np.array([[0, 0, 1]]), torch.int64)
    out = model(coord, atype)
    # Only the two type-0 atoms interact. The neutral constraint shifts the
    # charges (1, 1, -1) of the three atoms to (2/3, 2/3, -4/3).
    charge = 1.0 if constraint is None else 2.0 / 3.0
    erf = torch.erf(torch.tensor(3.2, dtype=torch.float64)).item()
    expected = COULOMB_CONSTANT * charge**2 * erf / 4.0
    np.testing.assert_allclose(out["energy"].item(), expected, rtol=1e-12)
    assert out["force"][0, 2].abs().max() == 0


@pytest.mark.parametrize("compiled", [False, True])
def test_vacuum_reference_with_long_range(compiled: bool) -> None:
    """An isolated atom keeps its reference energy; reference rows join no pair."""
    config = _config("les", constraint={"source": "charge_spin"})
    config["type_map"] = ["O", "H"]
    config["preset_out_bias"] = {"energy": {"O": -3.0, "H": 0.5}}
    config["descriptor"]["add_chg_spin_ebd"] = True
    config["descriptor"]["default_chg_spin"] = [0.0, 1.0]
    config["fitting_net"]["vacuum_ref"] = True
    model = get_model(config).to(env.DEVICE).eval()
    assert model.get_fitting_net().needs_vacuum_descriptor()
    rng = np.random.default_rng(11)
    coord = _tensor(rng.uniform(0.0, 2.5, (2, 4, 3)))
    atype = _tensor(np.array([[0, 1, 1, 0], [1, 0, -1, -1]]), torch.int64)
    if compiled:
        model.train()
        runner = _CompiledModel(model, _get_model_structure_key(model))
        runner.train()
        reference = model(coord, atype)
        out = runner(coord, atype)
        torch.testing.assert_close(out["energy"], reference["energy"])
        torch.testing.assert_close(out["force"], reference["force"])
        return
    out = model(coord, atype)
    for frame in range(2):
        alone = model(coord[frame : frame + 1], atype[frame : frame + 1])
        torch.testing.assert_close(alone["energy"][0], out["energy"][frame])
    # The output statistics set the bias; an atom without neighbors and in
    # its reference state then has exactly the bias of its type.
    bias = model.atomic_model.out_bias.clone()
    bias[0, :, 0] = torch.tensor([-3.0, 0.5])
    model.atomic_model.set_out_bias(bias)
    isolated = model(
        coord[:1, :1],
        atype[:1, :1],
        charge_spin=_tensor(np.array([[0.0, 3.0]])),
    )
    torch.testing.assert_close(
        isolated["energy"].reshape(()),
        torch.tensor(-3.0, dtype=torch.float64, device=env.DEVICE),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_self_label_bias_adjustment_is_invariant(kernel: str) -> None:
    """Labelling a sample with the model's own energies leaves its bias unchanged."""
    model = _model(kernel)
    with torch.no_grad():
        model.atomic_model.long_range.charge_net.bias_atom_e[:, 0] = torch.tensor(
            [0.8, -0.6]
        )
    coord, atype = _batch()
    coord, atype = coord[:1], atype[:1]
    energy = model(coord, atype)["energy"].detach().cpu().numpy()
    natoms = np.array([[6, 6, 3, 3]], dtype=np.int32)
    sample = [
        {
            "coord": coord.cpu().numpy(),
            "atype": atype.cpu().numpy(),
            "box": None,
            "natoms": natoms,
            "energy": energy.reshape(1, 1),
            "find_energy": np.float32(1.0),
        }
    ]
    bias_before = model.atomic_model.out_bias.detach().clone()
    model_change_out_bias(model, lambda: sample, "change-by-statistic")
    torch.testing.assert_close(
        model.atomic_model.out_bias, bias_before, atol=1e-10, rtol=0
    )
    after = model(coord, atype)["energy"].detach().cpu().numpy()
    np.testing.assert_allclose(after, energy, atol=1e-10)


@pytest.mark.parametrize("weights", ["sum", "mean", [0.3, 1.4]])
def test_linear_composition_combines_complete_energies(weights: object) -> None:
    """A linear composition weights the complete energy of each child.

    The second child is an ordinary model, which reads no topology.
    """
    child = _config("les")
    child["type"] = "standard"
    other = copy.deepcopy(child)
    other.pop("long_range")
    other["descriptor"]["seed"] = 31
    composition = (
        get_model(
            {
                "type": "linear_ener",
                "type_map": child["type_map"],
                "weights": weights,
                "models": [child, other],
            }
        )
        .to(env.DEVICE)
        .eval()
    )
    standalone = [_model("les"), get_model(other).to(env.DEVICE).eval()]
    for index, model in enumerate(standalone):
        model.atomic_model.load_state_dict(
            composition.atomic_model.models[index].state_dict()
        )
    coord, atype = _batch()
    out = composition(coord, atype)
    if weights == "sum":
        w = [1.0, 1.0]
    elif weights == "mean":
        w = [0.5, 0.5]
    else:
        w = weights
    energy = sum(
        ww * m(coord, atype)["energy"] for ww, m in zip(w, standalone, strict=True)
    )
    force = sum(
        ww * m(coord, atype)["force"] for ww, m in zip(w, standalone, strict=True)
    )
    torch.testing.assert_close(out["energy"], energy)
    torch.testing.assert_close(out["force"], force)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_parameter_gradients_and_freezing(kernel: str) -> None:
    """Energy and force losses reach the charge network and the kernel; frozen parts stay out."""
    model = _model(kernel).train()
    coord, atype = _batch()
    out = model(coord, atype)
    loss = out["energy"].square().sum() + out["force"].square().sum()
    loss.backward()
    long_range = model.atomic_model.long_range
    # The charge constraint fixes the uniform offset of the constrained
    # channel, so its entry in the output bias of the charge net is redundant
    # by construction and carries an exactly zero gradient.
    output_bias = long_range.charge_net.nets._module_networks["0"].layers[-1].b
    constrained = (
        long_range.charge_constraint["channel"]
        if long_range.charge_constraint is not None
        else None
    )
    for name, parameter in long_range.named_parameters():
        assert parameter.grad is not None, name
        grad = parameter.grad.abs()
        if parameter is output_bias and constrained is not None:
            grad = grad[[c for c in range(grad.shape[0]) if c != constrained]]
            if grad.numel() == 0:
                # every channel is constrained
                continue
        assert grad.max() > 0, name

    config = _config(kernel)
    config["long_range"]["kernel"]["trainable"] = False
    config["long_range"]["charge_net"]["trainable"] = False
    frozen = get_model(config).to(env.DEVICE).train()
    names = [
        name
        for name, parameter in frozen.named_parameters()
        if parameter.requires_grad and "long_range" in name
    ]
    assert names == []
    frozen(coord, atype)["energy"].sum().backward()


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_serialization_round_trip(kernel: str) -> None:
    """The portable record and the state dict restore the same model."""
    model = _model(kernel)
    rng = np.random.default_rng(5)
    with torch.no_grad():
        for parameter in model.atomic_model.long_range.parameters():
            parameter.add_(0.01 * _tensor(rng.normal(size=tuple(parameter.shape))))
    coord, atype = _batch()
    expected = model(coord, atype)
    record = model.serialize()
    assert record["@version"] == 3
    restored = type(model).deserialize(copy.deepcopy(record)).to(env.DEVICE).eval()
    torch.testing.assert_close(restored(coord, atype)["energy"], expected["energy"])
    rebuilt = _model(kernel)
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(rebuilt(coord, atype)["force"], expected["force"])


def test_construction_is_reproducible() -> None:
    """Without a seed of its own the charge network follows the fitting seed."""
    config = _config("les")
    del config["long_range"]["charge_net"]["seed"]
    first, second = (get_model(copy.deepcopy(config)) for _ in range(2))
    for (name, a), (_, b) in zip(
        first.named_parameters(), second.named_parameters(), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0, msg=name)


def test_ordinary_record_keeps_version() -> None:
    """A model without a long-range module keeps its record and constructor."""
    config = _config("les")
    config.pop("long_range")
    model = get_model(config)
    record = model.serialize()
    assert record["@version"] == 2
    assert "long_range" not in record
    assert not model.atomic_model.has_long_range()


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_checkpoint_inference_matches_direct_forward(tmp_path, kernel: str) -> None:
    """DeepPot on a checkpoint builds the topology over the real atoms only."""
    model = _model(kernel)
    checkpoint = tmp_path / f"long_range_{kernel}.pt"
    wrapper = ModelWrapper(model, model_params=_config(kernel))
    torch.save({"model": wrapper.state_dict()}, checkpoint)
    coord, atype = _batch(seed=7)
    expected = model(coord, atype, do_atomic_virial=True)
    dp = DeepPot(str(checkpoint), auto_batch_size=False)
    names = ("energy", "force", "virial", "atom_energy", "atom_virial")
    coord_np = coord.cpu().numpy()
    actual = dict(
        zip(
            names,
            dp.eval(coord_np, None, atype.cpu().numpy(), atomic=True, mixed_type=True),
            strict=True,
        )
    )
    for name in names:
        np.testing.assert_allclose(
            actual[name],
            expected[name].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=name,
        )
    coord_np[1, 3:] += 40.0
    moved = dp.eval(coord_np, None, atype.cpu().numpy(), mixed_type=True)
    np.testing.assert_allclose(moved[0], actual["energy"], rtol=0, atol=1e-12)


@pytest.mark.parametrize(
    ("kernel", "source", "suffix"),
    [("les", "neutral", ".pt2"), ("sog", "neutral", ".pte"), ("les", "fparam", ".pte")],
)
def test_frozen_model_matches_direct_forward(
    tmp_path, kernel: str, source: str, suffix: str
) -> None:
    """A frozen model reads the pair topology as named inputs of any length."""
    config = _config(kernel, constraint={"source": source, "fparam_index": 0})
    frame = {}
    if source == "fparam":
        config["fitting_net"]["numb_fparam"] = 1
        frame["fparam"] = np.array([[1.0], [-0.5], [2.0]])
    model = get_model(config).to(env.DEVICE).eval()
    frozen = tmp_path / f"long_range_{kernel}{suffix}"
    deserialize_to_file(
        str(frozen),
        {"model": model.serialize(), "model_def_script": config},
        do_atomic_virial=True,
        lower_kind="graph",
    )
    dp = DeepPot(str(frozen), auto_batch_size=False)
    assert dp.deep_eval.metadata["topologies"] == ["all_pairs"]
    if suffix == ".pte":
        names = [
            node.name
            for node in torch.export.load(str(frozen)).graph_module.graph.nodes
            if node.op == "placeholder"
        ]
        for name, expected_name in zip(
            names[-3:], ("pair_index", "pair_vec", "pair_mask"), strict=True
        ):
            assert name.startswith(expected_name), names
    names = ("energy", "force", "virial", "atom_energy", "atom_virial")
    for seed, nloc in ((7, 6), (8, 4)):
        coord, atype = _batch(seed=seed)
        coord, atype = coord[:, :nloc], atype[:, :nloc]
        expected = model(
            coord,
            atype,
            do_atomic_virial=True,
            **{key: _tensor(value) for key, value in frame.items()},
        )
        actual = dp.eval(
            coord.cpu().numpy(),
            None,
            atype.cpu().numpy(),
            atomic=True,
            mixed_type=True,
            **frame,
        )
        for name, value in zip(names, actual, strict=True):
            np.testing.assert_allclose(
                value,
                expected[name].detach().cpu().numpy().reshape(value.shape),
                rtol=1e-10,
                atol=1e-10,
                err_msg=name,
            )


@pytest.mark.parametrize(
    ("kernel", "source"), [("les", None), ("les", "neutral"), ("sog", "charge_spin")]
)
def test_coordinate_evaluation_matches_pair_graph(
    kernel: str, source: str | None
) -> None:
    """The evaluation from node positions matches the pair-graph evaluation.

    It returns the atomic energy, the derivative of the energy with respect to
    the descriptor through the constrained charges, and the force at a fixed
    descriptor, all of which the pair graph yields by automatic
    differentiation. Two frames, an uncharged node and trailing padding nodes
    exercise the frame sums of the constraint.
    """
    from deepmd.pt_expt.kernels.long_range import (
        op_available,
    )

    if not op_available():
        pytest.skip("the direct-sum operator is not available")
    config = _config(kernel, constraint=None if source is None else {"source": source})
    if source == "charge_spin":
        config["descriptor"]["add_chg_spin_ebd"] = True
    long_range = get_model(config).atomic_model.long_range.eval()
    if source == "charge_spin":
        long_range.fold_charge_state([2.0, 1.0])
    rng = np.random.default_rng(7)
    n_node = torch.tensor([9, 6], device=env.DEVICE)
    node_mask = torch.ones(17, dtype=torch.bool, device=env.DEVICE)
    node_mask[4] = False
    node_mask[15:] = False
    coord = _tensor(rng.uniform(0.0, 6.0, (17, 3)))
    descriptor = _tensor(rng.normal(size=(17, long_range.charge_net.dim_descrpt)))
    atype = _tensor(rng.integers(0, 2, 17), torch.int64)

    atom_energy, descriptor_gradient, force, _, _ = long_range.energy_force_from_coord(
        descriptor, atype, node_mask, n_node, coord
    )

    leaf_descriptor = descriptor.clone().requires_grad_(True)
    leaf_coord = coord.clone().requires_grad_(True)
    pairs = build_pair_graph(leaf_coord, n_node, atype, node_mask)
    expected = long_range.call_graph(
        leaf_descriptor, atype, node_mask, n_node, {ALL_PAIRS: pairs}
    )
    expected_descriptor_gradient, expected_coord_gradient = torch.autograd.grad(
        expected.sum(), (leaf_descriptor, leaf_coord)
    )
    for actual, target in (
        (atom_energy, expected),
        (descriptor_gradient, expected_descriptor_gradient),
        (force, -expected_coord_gradient),
    ):
        target = target.detach()
        scale = target.abs().max().item()
        assert scale > 0.0
        np.testing.assert_allclose(
            actual.detach().to(torch.float64).cpu().numpy(),
            target.cpu().numpy(),
            rtol=0.0,
            atol=2e-5 * scale,
        )


@pytest.mark.parametrize(
    ("kernel", "source"), [("les", "charge_spin"), ("sog", "neutral")]
)
def test_compressed_frozen_model_matches_uncompressed(
    tmp_path, kernel: str, source: str
) -> None:
    """A compressed model keeps its long-range module through the freeze.

    On a CUDA device the freeze selects the compact canonical lower, whose
    direct-sum operator reads the positions of the atoms; on the CPU the
    graph lower reads the pair lists. A constraint that reads the total
    charge from the charge state is folded with the descriptor and follows
    the state the deployment serves. Both match the uncompressed model.
    """
    config = _config(kernel, constraint={"source": source})
    config["descriptor"].update(n_radial=8, precision="float32")
    if source == "charge_spin":
        config["descriptor"].update(add_chg_spin_ebd=True, default_chg_spin=[0.0, 1.0])
    config["fitting_net"].update(
        neuron=[32, 32], activation_function="silu", resnet_dt=False
    )
    config["fitting_net"]["precision"] = "float32"
    config["long_range"]["charge_net"] = {"neuron": [16, 16], "seed": 23}
    model = get_model(config).to(env.DEVICE).eval()
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.add_(0.05 * noise.to(parameter))
    reference = copy.deepcopy(model)
    model.atomic_model.enable_compression(0.5)
    frozen = tmp_path / f"long_range_compressed_{kernel}.pt2"
    deserialize_to_file(
        str(frozen),
        {"model": model.serialize(), "model_def_script": config},
        do_atomic_virial=True,
        lower_kind="auto",
    )
    dp = DeepPot(str(frozen), auto_batch_size=False)
    expected_lower = "dpa4c_canonical" if env.DEVICE.type == "cuda" else "graph"
    assert dp.deep_eval.metadata["lower_input_kind"] == expected_lower
    rng = np.random.default_rng(3)
    coord = rng.uniform(0.0, 7.0, (1, 40, 3)) + 100.0
    atype = rng.integers(0, 2, (1, 40))
    names = ("energy", "force", "virial", "atom_energy", "atom_virial")
    # The tables of the compressed descriptor approximate its networks. On the
    # graph lower the charges of this model amplify that error at a charged
    # state, to about 1e-4 of the scale at [1, 2]; the compact canonical lower
    # stays within a few 1e-6.
    tolerance = 2e-5 if expected_lower == "dpa4c_canonical" else 5e-4
    states = [[0.0, 1.0], [1.0, 2.0], [0.0, 1.0]] if source == "charge_spin" else [None]
    for state in states:
        condition = {} if state is None else {"charge_spin": np.array([state])}
        expected = reference(
            _tensor(coord),
            _tensor(atype, torch.int64),
            do_atomic_virial=True,
            **{key: _tensor(value) for key, value in condition.items()},
        )
        actual = dp.eval(coord, None, atype, atomic=True, **condition)
        for name, value in zip(names, actual, strict=True):
            target = expected[name].detach().cpu().numpy().reshape(value.shape)
            scale = np.abs(target).max()
            assert np.abs(value - target).max() <= tolerance * scale + 1e-5, (
                name,
                state,
            )


@pytest.mark.parametrize("kernel", ["les", "sog"])
@REQUIRES_SUPPORTED_COMPILE
def test_compiled_matches_eager(kernel: str) -> None:
    """The compiled graph lower evaluates the complete model, train and eval.

    In training the gradient of an energy and force loss, whose force part is
    a second-order derivative of the energy, reaches the long-range
    parameters as in eager execution.
    """
    model = _model(kernel)
    coord, atype = _batch()
    parameters = list(model.atomic_model.long_range.parameters())

    def loss(out: dict) -> torch.Tensor:
        return out["energy"].square().sum() + out["force"].square().sum()

    for training in (True, False):
        model.train(training)
        compiled = _CompiledModel(
            model, _get_model_structure_key(model), compile_eval=True
        )
        compiled.train(training)
        reference = model(coord, atype)
        result = compiled(coord, atype)
        assert compiled._compiled_lower_by_mode, "the compiled graph was not used"
        torch.testing.assert_close(result["energy"], reference["energy"])
        torch.testing.assert_close(result["force"], reference["force"])
        torch.testing.assert_close(result["virial"], reference["virial"])
        if training:
            expected = torch.autograd.grad(loss(reference), parameters)
            actual = torch.autograd.grad(loss(result), parameters)
            for expected_grad, actual_grad in zip(expected, actual, strict=True):
                torch.testing.assert_close(actual_grad, expected_grad)


def test_unsupported_combinations_fail() -> None:
    """Unsupported inputs, routes, combinations and descriptors are rejected."""
    model = _model("les")
    coord, atype = _batch()
    box = 10.0 * torch.eye(3, dtype=torch.float64, device=env.DEVICE).reshape(1, 9)
    with pytest.raises(NotImplementedError, match="non-periodic"):
        model(coord[:1], atype[:1], box=box)
    nlist = torch.full((1, 6, 4), -1, dtype=torch.int64, device=env.DEVICE)
    with pytest.raises(NotImplementedError, match="NeighborGraph"):
        model.atomic_model.forward_atomic(coord[:1], atype[:1], nlist)
    config = _config("les")
    config["spin"] = {"use_spin": [True, False], "scheme": "native"}
    with pytest.raises(NotImplementedError):
        get_model(config)
    config = _config("les")
    config["use_srtab"] = "table.txt"
    config["sw_rmin"], config["sw_rmax"] = 0.5, 1.0
    with pytest.raises(NotImplementedError, match="ZBL"):
        get_model(config)
    config = _config("les")
    config["type_map"] = ["O", "H"]
    config["bridging_method"] = "ZBL"
    with pytest.raises(NotImplementedError, match="bridging"):
        get_model(config)
    config = _config("les")
    config["atom_exclude_types"] = [1]
    with pytest.raises(NotImplementedError, match="atom_exclude_types"):
        get_model(config)
    child = _config("les")
    child["type"] = "standard"
    with pytest.raises(NotImplementedError, match="atom_exclude_types"):
        get_model(
            {
                "type": "linear_ener",
                "type_map": child["type_map"],
                "weights": "sum",
                "models": [child],
                "atom_exclude_types": [1],
            }
        )
    config = _config("les")
    config["descriptor"] = {
        "type": "se_e2_a",
        "rcut": 3.0,
        "rcut_smth": 2.0,
        "sel": [8, 8],
    }
    with pytest.raises(NotImplementedError, match="DPA4C"):
        get_model(config)


def test_compressing_the_descriptor_alone_is_rejected() -> None:
    """A folded descriptor and a constraint reading the condition are rejected."""
    config = _config("les", constraint={"source": "charge_spin"})
    config["descriptor"].update(
        n_radial=8,
        precision="float32",
        add_chg_spin_ebd=True,
        default_chg_spin=[0.0, 1.0],
    )
    config["fitting_net"]["precision"] = "float32"
    model = get_model(config).to(env.DEVICE).eval()
    model.atomic_model.descriptor.enable_compression(0.5)
    coord, atype = _batch()
    with pytest.raises(ValueError, match="folds the charge/spin condition"):
        model(coord[:1], atype[:1])


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_dpmodel_reference_matches(kernel: str) -> None:
    """The dpmodel reference evaluates the same energy from the portable record."""
    from deepmd.dpmodel.model.base_model import BaseModel as BaseModelDP

    model = _model(kernel)
    coord, atype = _batch()
    expected = model(coord, atype)
    reference = BaseModelDP.deserialize(model.serialize())
    ret = reference.call_common(
        coord.cpu().numpy(), atype.cpu().numpy(), neighbor_graph_method="dense"
    )
    np.testing.assert_allclose(
        ret["energy_redu"].reshape(-1),
        expected["energy"].detach().cpu().numpy().reshape(-1),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ret["energy"].reshape(3, 6),
        expected["atom_energy"].detach().cpu().numpy().reshape(3, 6),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pair_graph_skips_padding() -> None:
    """The pair topology joins real atoms of one frame only and ends in masked guards."""
    coord = torch.arange(15, dtype=torch.float64).reshape(5, 3)
    pairs = build_pair_graph(
        coord,
        torch.tensor([3, 2]),
        torch.tensor([0, -1, 1, 0, 1]),
        torch.tensor([True, False, True, True, True]),
    )
    real = pairs.index[:, pairs.mask]
    assert real.tolist() == [[0, 3], [2, 4]]
    assert (~pairs.mask).sum() == 2


# ---------------------------------------------------------------------------
# Periodic (k-space) evaluation
# ---------------------------------------------------------------------------


def _periodic_model(kernel: str, **kwargs: object) -> torch.nn.Module:
    config = _config(kernel, **kwargs)
    config["long_range"]["periodic"] = True
    return get_model(config).to(env.DEVICE).eval()


def _periodic_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A compact cluster (all atoms within rcut of each other) in a large triclinic cell."""
    box = np.array([[9.0, 0.4, 0.0], [0.0, 8.5, 0.3], [0.0, 0.0, 9.5]])
    rng = np.random.default_rng(7)
    coord = 3.0 + rng.uniform(0.0, 1.4, (6, 3))
    atype = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
    return coord[None], atype[None], box[None]


def _kspace_reference(
    coord: np.ndarray, charge: np.ndarray, box: np.ndarray, kernel: str
) -> float:
    """Independent k-space evaluation of the periodic long-range energy, in eV.

    E = k_e [ 1/(2V) sum_{k != 0} K_hat(k) |rho(k)|^2 - K(0)/2 sum_i |q_i|^2 ].
    """
    rec = 2.0 * np.pi * np.linalg.inv(box)  # columns g_d: k(n) = rec @ n
    # the module's default kspace_tolerance=1e-8 sets the truncation
    if kernel == "les":
        k_max = 2.0 * 0.8 * np.sqrt(np.log(1e8))
    else:
        k_max = np.sqrt(2.0 * np.log(1e8)) / 0.8
    n_d = np.ceil(k_max * np.linalg.norm(box, axis=1) / (2 * np.pi)).astype(int)
    grids = np.meshgrid(
        *[np.arange(-n, n + 1, dtype=float) for n in n_d], indexing="ij"
    )
    n_all = np.stack(grids, axis=-1).reshape(-1, 3)
    k = n_all @ rec.T
    k2 = np.sum(k * k, axis=-1)
    keep = (k2 > 0) & (k2 <= k_max * k_max)
    k, k2 = k[keep], k2[keep]
    phase = k @ coord.T  # (nk, N)
    rho = (np.cos(phase) + 1j * np.sin(phase)) @ charge  # (nk, C)
    s_sq = np.abs(rho) ** 2
    s_sq = s_sq.sum(axis=-1)
    if kernel == "les":
        k_hat = 4.0 * np.pi * np.exp(-k2 / (4.0 * 0.8**2)) / k2
        k0 = 2.0 * 0.8 / np.sqrt(np.pi)
    else:
        amplitude = np.array([0.7, -0.2, 0.1])
        width = np.array([0.8, 1.7, 3.2])
        k_hat = np.sum(
            amplitude[None, :]
            * (2.0 * np.pi) ** 1.5
            * width[None, :] ** 3
            * np.exp(-0.5 * k2[:, None] * width[None, :] ** 2),
            axis=-1,
        )
        k0 = amplitude.sum()
    volume = abs(np.linalg.det(box))
    return float(
        COULOMB_CONSTANT
        * ((k_hat * s_sq).sum() / (2 * volume) - 0.5 * k0 * (charge * charge).sum())
    )


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_energy_matches_kspace_reference(kernel: str) -> None:
    """With fixed charges the periodic energy is the documented k-space sum."""
    model = _periodic_model(kernel, constraint=None)
    _set_constant_charges(model, [1.0, -0.5])
    coord, atype, box = _periodic_case()
    out = model(_tensor(coord), _tensor(atype, torch.int64), box=_tensor(box))
    charge = np.zeros((6, CHANNELS[kernel]))
    charge[:, 0] = [1.0, -0.5, 1.0, -0.5, -0.5, 1.0]
    expected = _kspace_reference(coord[0], charge, box[0], kernel)
    np.testing.assert_allclose(out["energy"].item(), expected, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        out["atom_energy"].sum().item(), expected, rtol=1e-9, atol=1e-9
    )


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_force_virial_hessian_match_finite_differences(kernel: str) -> None:
    """Force, strain virial and Hessian derive from the complete periodic energy."""
    model = _periodic_model(kernel)
    coord, atype, box = _periodic_case()
    coord_t, atype_t, box_t = (
        _tensor(coord),
        _tensor(atype, torch.int64),
        _tensor(box),
    )
    out = model(coord_t, atype_t, box=box_t)

    def energy(c: np.ndarray, b: np.ndarray) -> float:
        return model(_tensor(c), atype_t, box=_tensor(b))["energy"].item()

    h = 1e-5
    force_fd = np.zeros((6, 3))
    for i in range(6):
        for d in range(3):
            plus, minus = coord.copy(), coord.copy()
            plus[0, i, d] += h
            minus[0, i, d] -= h
            force_fd[i, d] = -(energy(plus, box) - energy(minus, box)) / (2 * h)
    np.testing.assert_allclose(
        out["force"][0].detach().cpu().numpy(), force_fd, atol=1e-6
    )

    # The virial is -dE/d(strain) at zero strain, for r -> (I + eps) r and
    # B -> B (I + eps)^T (the strain family of the k-space strain leaf).
    virial_fd = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            strain = np.zeros((3, 3))
            strain[a, b] = h
            plus = coord @ (np.eye(3) + strain).T
            minus = coord @ (np.eye(3) - strain).T
            box_plus = box @ (np.eye(3) + strain).T
            box_minus = box @ (np.eye(3) - strain).T
            virial_fd[a, b] = -(energy(plus, box_plus) - energy(minus, box_minus)) / (
                2 * h
            )
    np.testing.assert_allclose(
        out["virial"][0].detach().cpu().numpy().reshape(3, 3), virial_fd, atol=1e-5
    )

    hessian_model = _periodic_model(kernel)
    hessian_model.load_state_dict(model.state_dict())
    hessian_model.enable_hessian()
    hessian = (
        hessian_model(coord_t, atype_t, box=box_t)["hessian"].detach().cpu().numpy()
    )
    hessian = hessian.reshape(18, 18)
    hessian_fd = np.zeros((18, 18))
    for kk in range(18):
        plus, minus = coord.copy(), coord.copy()
        plus.reshape(-1)[kk] += h
        minus.reshape(-1)[kk] -= h
        f_plus = (
            model(_tensor(plus), atype_t, box=box_t)["force"].detach().cpu().numpy()
        )
        f_minus = (
            model(_tensor(minus), atype_t, box=box_t)["force"].detach().cpu().numpy()
        )
        hessian_fd[:, kk] = -(f_plus - f_minus).reshape(-1) / (2 * h)
    np.testing.assert_allclose(hessian, hessian_fd, atol=1e-4)
    np.testing.assert_allclose(hessian, hessian.T, atol=1e-10)


@pytest.mark.parametrize("kernel", ["les", "sog"])
@REQUIRES_SUPPORTED_COMPILE
def test_periodic_compiled_matches_eager(kernel: str) -> None:
    """The compiled graph lower evaluates the periodic model like the eager one."""
    model = _periodic_model(kernel)
    coord, atype, box = _periodic_case()
    coord_t, atype_t, box_t = (
        _tensor(coord),
        _tensor(atype, torch.int64),
        _tensor(box),
    )
    model.train()
    runner = _CompiledModel(model, _get_model_structure_key(model))
    runner.train()
    reference = model.eval()(coord_t, atype_t, box=box_t)
    model.train()
    out = runner(coord_t, atype_t, box=box_t)
    for key in ("energy", "force", "virial"):
        torch.testing.assert_close(out[key], reference[key], rtol=1e-6, atol=1e-8)
    # the backward reaches the charge network and the kernel through the
    # compiled graph
    loss = out["energy"].square().sum() + out["force"].square().sum()
    loss.backward()
    grads = [parameter.grad for parameter in model.atomic_model.long_range.parameters()]
    assert grads and all(grad is not None for grad in grads)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_model_serves_non_periodic_batches(kernel: str) -> None:
    """A periodic-enabled model keeps its non-periodic evaluation."""
    model = _periodic_model(kernel)
    plain = _model(kernel)
    plain.load_state_dict(model.state_dict())
    coord, atype = _batch()
    periodic_enabled = model(coord, atype)
    plain_out = plain(coord, atype)
    torch.testing.assert_close(periodic_enabled["energy"], plain_out["energy"])
    torch.testing.assert_close(periodic_enabled["force"], plain_out["force"])


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_frames_are_independent(kernel: str) -> None:
    """Each frame's k grid and energy are independent of the rest of the batch."""
    model = _periodic_model(kernel)
    coord, atype, box = _periodic_case()
    coord2 = np.concatenate([coord, coord + 0.1], axis=0)
    atype2 = np.concatenate([atype, atype], axis=0)
    box2 = np.concatenate([box, box * 1.1], axis=0)
    batched = model(_tensor(coord2), _tensor(atype2, torch.int64), box=_tensor(box2))
    for frame in range(2):
        alone = model(
            _tensor(coord2[frame : frame + 1]),
            _tensor(atype2[frame : frame + 1], torch.int64),
            box=_tensor(box2[frame : frame + 1]),
        )
        torch.testing.assert_close(
            alone["energy"][0], batched["energy"][frame], rtol=1e-10, atol=1e-10
        )


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_serialization_round_trip(kernel: str) -> None:
    """The record keeps the periodic flag and evaluates identically."""
    from deepmd.dpmodel.model.base_model import BaseModel as BaseModelDP

    model = _periodic_model(kernel)
    assert model.atomic_model.required_topologies() == ("all_pairs", "k_grid")
    record = model.serialize()
    assert record["long_range"]["periodic"] is True
    reference = BaseModelDP.deserialize(record)
    assert reference.atomic_model.required_topologies() == ("all_pairs", "k_grid")
    coord, atype, box = _periodic_case()
    expected = model(_tensor(coord), _tensor(atype, torch.int64), box=_tensor(box))
    ret = reference.call_common(
        coord,
        atype,
        box,
        neighbor_graph_method="dense",
    )
    np.testing.assert_allclose(
        ret["energy_redu"].reshape(-1),
        expected["energy"].detach().cpu().numpy().reshape(-1),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_compression_is_rejected(kernel: str) -> None:
    """The compressed path does not serve the k-space evaluation."""
    config = _config(kernel)
    config["descriptor"]["precision"] = "float32"
    config["fitting_net"]["precision"] = "float32"
    config["long_range"]["charge_net"]["precision"] = "float32"
    config["long_range"]["periodic"] = True
    model = get_model(config).to(env.DEVICE).eval()
    with pytest.raises(NotImplementedError, match="reciprocal-space"):
        model.atomic_model.enable_compression(0.5)


# ---------------------------------------------------------------------------
# Periodic fixed k-grid mode
# ---------------------------------------------------------------------------


def _periodic_fixed_model(
    kernel: str, boxes: np.ndarray | None = None, **kwargs: object
) -> torch.nn.Module:
    """A periodic model in the fixed k-grid mode, sized by the given cells."""
    config = _config(kernel, **kwargs)
    config["long_range"]["periodic"] = True
    config["long_range"]["k_grid_mode"] = "fixed"
    model = get_model(config).to(env.DEVICE).eval()
    if boxes is not None:
        boxes = torch.as_tensor(np.asarray(boxes, dtype=np.float64).reshape(-1, 9))
        model.atomic_model.long_range.compute_input_stats([{"box": boxes}])
    return model


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_fixed_grid_matches_ragged(kernel: str) -> None:
    """The fixed grid with its mask evaluates like the ragged enumeration."""
    coord, atype, box = _periodic_case()
    ragged = _periodic_model(kernel)
    fixed = _periodic_fixed_model(kernel, box)
    # the state dicts of the two grid modes differ by the fixed grid buffer
    fixed.load_state_dict(ragged.state_dict(), strict=False)
    coord_t, atype_t, box_t = (
        _tensor(coord),
        _tensor(atype, torch.int64),
        _tensor(box),
    )
    expected = ragged(coord_t, atype_t, box=box_t)
    got = fixed(coord_t, atype_t, box=box_t)
    for key in ("energy", "force", "virial"):
        torch.testing.assert_close(got[key], expected[key], rtol=1e-9, atol=1e-12)
    # the gradient reaches the charge network and the kernel without NaN,
    # though masked points carry a singular kernel transform nowhere
    fixed.train()
    out = fixed(coord_t, atype_t, box=box_t)
    loss = out["energy"].square().sum() + out["force"].square().sum()
    loss.backward()
    grads = [
        parameter.grad
        for parameter in fixed.atomic_model.long_range.parameters()
        if parameter.grad is not None
    ]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_fixed_grid_mixed_cells(kernel: str) -> None:
    """One shared grid serves a batch of different cells, each masked to its sphere."""
    coord, atype, box = _periodic_case()
    coord2 = np.concatenate([coord, coord + 0.1], axis=0)
    atype2 = np.concatenate([atype, atype], axis=0)
    box2 = np.concatenate([box, box * 1.4], axis=0)
    fixed = _periodic_fixed_model(kernel, box2)
    ragged = _periodic_model(kernel)
    ragged.load_state_dict(fixed.state_dict(), strict=False)
    coord_t, atype_t, box_t = (
        _tensor(coord2),
        _tensor(atype2, torch.int64),
        _tensor(box2),
    )
    batched = fixed(coord_t, atype_t, box=box_t)
    expected = ragged(coord_t, atype_t, box=box_t)
    torch.testing.assert_close(
        batched["energy"], expected["energy"], rtol=1e-9, atol=1e-12
    )
    torch.testing.assert_close(
        batched["force"], expected["force"], rtol=1e-9, atol=1e-12
    )
    # the batched energy of a frame equals its standalone evaluation
    for frame in range(2):
        alone = fixed(
            _tensor(coord2[frame : frame + 1]),
            _tensor(atype2[frame : frame + 1], torch.int64),
            box=_tensor(box2[frame : frame + 1]),
        )
        torch.testing.assert_close(
            alone["energy"][0], batched["energy"][frame], rtol=1e-10, atol=1e-10
        )


def _kspace_reference_bounded(
    coord: np.ndarray,
    charge: np.ndarray,
    box: np.ndarray,
    kernel: str,
    bounds: np.ndarray,
) -> float:
    """The k-space reference of :func:`_kspace_reference` restricted to the
    integer indices within ``bounds`` — the truncated sum of a fixed grid.
    """
    rec = 2.0 * np.pi * np.linalg.inv(box)
    if kernel == "les":
        k_max = 2.0 * 0.8 * np.sqrt(np.log(1e8))
    else:
        k_max = np.sqrt(2.0 * np.log(1e8)) / 0.8
    grids = np.meshgrid(
        *[np.arange(-n, n + 1, dtype=float) for n in bounds], indexing="ij"
    )
    n_all = np.stack(grids, axis=-1).reshape(-1, 3)
    k = n_all @ rec.T
    k2 = np.sum(k * k, axis=-1)
    keep = (k2 > 0) & (k2 <= k_max * k_max)
    k, k2 = k[keep], k2[keep]
    phase = k @ coord.T
    rho = (np.cos(phase) + 1j * np.sin(phase)) @ charge
    s_sq = (np.abs(rho) ** 2).sum(axis=-1)
    if kernel == "les":
        k_hat = 4.0 * np.pi * np.exp(-k2 / (4.0 * 0.8**2)) / k2
        k0 = 2.0 * 0.8 / np.sqrt(np.pi)
    else:
        amplitude = np.array([0.7, -0.2, 0.1])
        width = np.array([0.8, 1.7, 3.2])
        k_hat = np.sum(
            amplitude[None, :]
            * (2.0 * np.pi) ** 1.5
            * width[None, :] ** 3
            * np.exp(-0.5 * k2[:, None] * width[None, :] ** 2),
            axis=-1,
        )
        k0 = amplitude.sum()
    volume = abs(np.linalg.det(box))
    return float(
        COULOMB_CONSTANT
        * ((k_hat * s_sq).sum() / (2 * volume) - 0.5 * k0 * (charge * charge).sum())
    )


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_fixed_grid_truncates_oversized_cells(kernel: str) -> None:
    """A cell beyond the grid is evaluated with the grid alone, not an error."""
    coord, atype, box = _periodic_case()
    model = _periodic_fixed_model(kernel, box, constraint=None)
    _set_constant_charges(model, [1.0, -0.5])
    big_box = box * 3.0
    with pytest.warns(UserWarning, match="truncated"):
        out = model(_tensor(coord), _tensor(atype, torch.int64), box=_tensor(big_box))
    grid_n = model.atomic_model.long_range.k_grid_n
    grid_n = grid_n.detach().cpu().numpy() if hasattr(grid_n, "detach") else grid_n
    bounds = np.abs(np.asarray(grid_n)).max(axis=0)
    charge = np.zeros((6, CHANNELS[kernel]))
    charge[:, 0] = [1.0, -0.5, 1.0, -0.5, -0.5, 1.0]
    expected = _kspace_reference_bounded(coord[0], charge, big_box[0], kernel, bounds)
    np.testing.assert_allclose(out["energy"].item(), expected, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_fixed_grid_needs_statistics_or_bounds(kernel: str) -> None:
    """Without statistics or explicit bounds the grid is not sized; bounds skip them."""
    coord, atype, box = _periodic_case()
    unsized = _periodic_fixed_model(kernel)
    with pytest.raises(ValueError, match="not sized yet"):
        unsized(_tensor(coord), _tensor(atype, torch.int64), box=_tensor(box))
    config = _config(kernel)
    config["long_range"]["periodic"] = True
    config["long_range"]["k_grid_mode"] = "fixed"
    config["long_range"]["k_grid_bounds"] = [12, 12, 12]
    model = get_model(config).to(env.DEVICE).eval()
    assert model.atomic_model.long_range.k_grid_n.shape == (25**3 - 1, 3)
    out = model(_tensor(coord), _tensor(atype, torch.int64), box=_tensor(box))
    reference = _periodic_model(kernel)
    reference.load_state_dict(model.state_dict(), strict=False)
    expected = reference(_tensor(coord), _tensor(atype, torch.int64), box=_tensor(box))
    torch.testing.assert_close(out["energy"], expected["energy"], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_fixed_grid_serialization_round_trip(kernel: str) -> None:
    """The record keeps the sized grid and evaluates identically."""
    from deepmd.dpmodel.model.base_model import BaseModel as BaseModelDP

    coord, atype, box = _periodic_case()
    model = _periodic_fixed_model(kernel, box)
    record = model.serialize()
    assert record["long_range"]["k_grid_mode"] == "fixed"
    assert "k_grid_n" in record["long_range"]
    reference = BaseModelDP.deserialize(record)
    assert reference.atomic_model.long_range.k_grid_mode == "fixed"
    expected = model(_tensor(coord), _tensor(atype, torch.int64), box=_tensor(box))
    ret = reference.call_common(coord, atype, box, neighbor_graph_method="dense")
    np.testing.assert_allclose(
        ret["energy_redu"].reshape(-1),
        expected["energy"].detach().cpu().numpy().reshape(-1),
        rtol=1e-9,
        atol=1e-9,
    )


@pytest.mark.parametrize("kernel", ["les", "sog"])
@REQUIRES_SUPPORTED_COMPILE
def test_periodic_fixed_grid_compiled_matches_eager(kernel: str) -> None:
    """The compiled graph lower evaluates the fixed-grid model like the eager one."""
    coord, atype, box = _periodic_case()
    model = _periodic_fixed_model(kernel, box)
    coord_t, atype_t, box_t = (
        _tensor(coord),
        _tensor(atype, torch.int64),
        _tensor(box),
    )
    model.train()
    runner = _CompiledModel(model, _get_model_structure_key(model))
    runner.train()
    reference = model.eval()(coord_t, atype_t, box=box_t)
    model.train()
    out = runner(coord_t, atype_t, box=box_t)
    for key in ("energy", "force", "virial"):
        torch.testing.assert_close(out[key], reference[key], rtol=1e-6, atol=1e-8)
    loss = out["energy"].square().sum() + out["force"].square().sum()
    loss.backward()
    grads = [parameter.grad for parameter in model.atomic_model.long_range.parameters()]
    assert grads and all(grad is not None for grad in grads)


# ---------------------------------------------------------------------------
# Periodic decoupled k grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["les", "sog"])
def test_periodic_decoupled_grid_ignores_kernel_drift(kernel: str) -> None:
    """A decoupled model selects the k points with the initial kernel but
    evaluates them with the live one: the drift moves the energy, not the grid.
    """
    config = _config(kernel)
    config["long_range"]["periodic"] = True
    config["long_range"]["decouple_k_grid"] = True
    model = get_model(config).to(env.DEVICE).eval()
    coord, atype, box = _periodic_case()
    coord_t, atype_t, box_t = (
        _tensor(coord),
        _tensor(atype, torch.int64),
        _tensor(box),
    )
    long_range = model.atomic_model.long_range
    grid_args = (
        coord_t.reshape(-1, 3),
        torch.tensor([6], dtype=torch.int64, device=env.DEVICE),
        box_t,
    )
    before = long_range.build_topologies(*grid_args, atype=atype_t.reshape(-1))[
        "k_grid"
    ]
    energy_before = model(coord_t, atype_t, box=box_t)["energy"]
    # move the kernel far enough that a coupled grid would change
    with torch.no_grad():
        if kernel == "les":
            long_range.kernel.log_alpha.add_(1.0)
        else:
            long_range.kernel.log_width.sub_(1.0)
    after = long_range.build_topologies(*grid_args, atype=atype_t.reshape(-1))["k_grid"]
    torch.testing.assert_close(after.vec, before.vec)
    torch.testing.assert_close(after.mask, before.mask)
    # the kernel values still follow the live parameters
    energy_after = model(coord_t, atype_t, box=box_t)["energy"]
    assert not torch.allclose(energy_after, energy_before)
    # and the drifted kernel still takes a finite gradient through the graph
    model.train()
    out = model(coord_t, atype_t, box=box_t)
    loss = out["energy"].square().sum() + out["force"].square().sum()
    loss.backward()
    kernel_grads = [parameter.grad for parameter in long_range.kernel.parameters()]
    assert kernel_grads and all(
        grad is not None and torch.isfinite(grad).all() for grad in kernel_grads
    )
