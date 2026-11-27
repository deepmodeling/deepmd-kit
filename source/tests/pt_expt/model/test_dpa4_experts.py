# SPDX-License-Identifier: LGPL-3.0-or-later
"""pt_expt DPA4 with MoE: dpmodel parity, forces, merge, sharing and export."""

import copy

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.model.base_model import BaseModel as BaseModelDP
from deepmd.dpmodel.model.expert_merge import (
    branch_local_parameters,
    check_expert_state,
    element_banks,
    learned_atomic_models,
)
from deepmd.dpmodel.utils.expert_linear import (
    is_expert_parameter,
)
from deepmd.dpmodel.utils.expert_route import (
    frame_states,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.main import (
    parse_args,
)
from deepmd.pt.optimizer.hybrid_muon import (
    adam_route_patterns,
)
from deepmd.pt_expt.entrypoints.main import (
    change_bias,
    freeze,
    main,
)
from deepmd.pt_expt.entrypoints.moe_merge import (
    moe_merge,
)
from deepmd.pt_expt.model import (
    BaseModel,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.train.training import (
    _get_model_structure_key,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.pt_expt.utils.multi_task import (
    preprocess_shared_params,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.model_preset import (
    expand_model_preset,
)

from ..compile_utils import (
    REQUIRES_SUPPORTED_COMPILE,
)

EXPERTS = {"n_experts": 3}
# Every training frame uses the weights of its elements.
EVERY_FRAME = {**EXPERTS, "mat_prob": 0.0}
EXPERT_NAMES = (
    "task_weight",
    "task_bias",
    "element_weight",
    "element_bias",
    "expert_weight",
    "expert_bias",
    "adam_task_embedding",
    "expert_router",
)


def model_params(descriptor_moe=None, fitting_moe=None):
    model = expand_model_preset(
        {"preset": "dpa4.5-nano-v20261111", "type_map": ["H", "C", "O"]}
    )
    model["descriptor"].update(
        sel=40,
        seed=1,
        precision="float64",
        use_amp=False,
        add_chg_spin_ebd=True,
        default_chg_spin=[0.0, 1.0],
    )
    model["fitting_net"].update(seed=1, precision="float64", vacuum_ref=True)
    # The reference acts on an output whose bias the preset fixes.
    model["preset_out_bias"] = {"energy": {"H": -13.6, "C": -150.0, "O": -430.1}}
    if descriptor_moe is not None:
        model["descriptor"]["moe"] = copy.deepcopy(descriptor_moe)
    if fitting_moe is not None:
        model["fitting_net"]["moe"] = copy.deepcopy(fitting_moe)
    config = normalize(
        {
            "model": model,
            "learning_rate": {"start_lr": 1e-3},
            "loss": {},
            "training": {"training_data": {"systems": ["x"]}, "numb_steps": 1},
        }
    )
    return config["model"]


def multitask_models(bridging_method=None, spin=None, bulk_spin=None):
    """Two branches sharing an MoE descriptor and fitting network.

    With ``spin``, the first branch ``gas`` is a native-spin model, and with
    ``bulk_spin`` the second branch ``bulk`` as well.
    """
    branch = {
        "type_map": "type_map",
        "descriptor": "descriptor",
        "fitting_net": "fitting",
    }
    if bridging_method is not None:
        branch["bridging_method"] = bridging_method
    gas = copy.deepcopy(branch)
    if spin is not None:
        gas["spin"] = spin
    if bulk_spin is not None:
        branch = {**branch, "spin": bulk_spin}
    model = expand_model_preset(
        {
            "preset": "dpa4.5-nano-v20261111",
            "shared_dict": {
                "type_map": ["H", "C", "O"],
                "descriptor": {
                    "sel": 40,
                    "seed": 1,
                    "use_amp": False,
                    "add_chg_spin_ebd": True,
                    "default_chg_spin": [0.0, 1.0],
                    "moe": EXPERTS,
                },
                "fitting": {"seed": 1, "moe": EXPERTS},
            },
            "model_dict": {"gas": gas, "bulk": branch},
        }
    )
    model, shared_links = preprocess_shared_params(model)
    config = normalize(
        {
            "model": model,
            "learning_rate": {"start_lr": 1e-3},
            "loss_dict": {key: {} for key in model["model_dict"]},
            "training": {
                "data_dict": {
                    key: {"training_data": {"systems": ["x"]}}
                    for key in model["model_dict"]
                },
                "model_prob": dict.fromkeys(model["model_dict"], 0.5),
                "numb_steps": 1,
            },
        },
        multi_task=True,
    )
    models = {
        key: get_model(copy.deepcopy(params)).to(DEVICE)
        for key, params in config["model"]["model_dict"].items()
    }
    wrapper = ModelWrapper(models)
    wrapper.share_params(
        shared_links, model_key_prob_map=dict.fromkeys(models, 0.5), resume=True
    )
    return models, config["model"]["model_dict"]


def expert_parameters(model):
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if any(key in name for key in EXPERT_NAMES)
    }


def trained_like(model, seed=2):
    """Give the MoE weights and the task of the model nonzero values."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for parameter in expert_parameters(model).values():
            parameter.copy_(0.05 * torch.randn(parameter.shape, generator=generator))
    return model


def load_checkpoint(path):
    state = torch.load(path, map_location=DEVICE, weights_only=True)["model"]
    params = state["_extra_state"]["model_params"]
    model = get_model(copy.deepcopy(params)).to(DEVICE)
    ModelWrapper(model).load_state_dict(state)
    return model.eval(), params


def save_checkpoint(path, models, params) -> None:
    torch.save({"model": ModelWrapper(models, model_params=params).state_dict()}, path)


def assert_same_output(actual, expected, atol=1e-10) -> None:
    for key in ("atom_energy", "force"):
        np.testing.assert_allclose(
            actual[key].detach().cpu().numpy(),
            expected[key].detach().cpu().numpy(),
            atol=atol,
        )


rng = np.random.default_rng(0)
COORD = torch.tensor(rng.uniform(0.0, 7.0, (2, 18)), device=DEVICE)
ATYPE = torch.tensor([[0, 1, 2, 0, 1, 0], [1, 1, 0, 0, 2, 2]], device=DEVICE)
BOX = torch.tensor(np.tile((np.eye(3) * 7.0).reshape(1, 9), (2, 1)), device=DEVICE)


def test_parity_with_dpmodel_and_forces() -> None:
    model = get_model(model_params(EXPERTS, EXPERTS)).to(DEVICE).eval()
    parameters = expert_parameters(model)
    # Every component of the MoE weights is a trainable parameter.
    for key in EXPERT_NAMES:
        assert any(key in name for name in parameters), key
    assert all(parameter.requires_grad for parameter in parameters.values())
    assert not [
        name
        for name, _ in model.named_buffers()
        if "expert_w" in name or "task_" in name
    ]
    trained_like(model)
    out = model(COORD, ATYPE, BOX)
    reference = BaseModelDP.deserialize(model.serialize()).call_common(
        COORD.cpu().numpy(),
        ATYPE.cpu().numpy(),
        BOX.cpu().numpy(),
        neighbor_graph_method="dense",
    )
    np.testing.assert_allclose(
        out["energy"].detach().cpu().numpy().ravel(),
        reference["energy_redu"].ravel(),
        atol=1e-10,
    )
    step = 1e-5
    shifted = [COORD.clone(), COORD.clone()]
    shifted[0][0, 4] += step
    shifted[1][0, 4] -= step
    upper, lower = (model(c, ATYPE, BOX)["energy"][0, 0].item() for c in shifted)
    assert out["force"].abs().max().item() > 1e-6
    np.testing.assert_allclose(
        -(upper - lower) / (2 * step),
        out["force"][0, 1, 1].item(),
        rtol=1e-5,
        atol=1e-8,
    )
    # A state dict restores the model.
    restored = get_model(model_params(EXPERTS, EXPERTS)).to(DEVICE).eval()
    restored.load_state_dict(model.state_dict())
    assert_same_output(restored(COORD, ATYPE, BOX), out)


@pytest.mark.parametrize(
    ("descriptor", "fitting", "vacuum_ref"),
    [
        (EVERY_FRAME, EVERY_FRAME, True),
        (EVERY_FRAME, EVERY_FRAME, False),
        (EVERY_FRAME, None, True),
        (None, EVERY_FRAME, True),
        ({**EVERY_FRAME, "element": ["O", "C"]}, EVERY_FRAME, True),
    ],
)
def test_absent_elements_train_no_row(descriptor, fitting, vacuum_ref) -> None:
    """The element rows of the types absent from a batch have zero gradient."""
    params = model_params(descriptor, fitting)
    params["fitting_net"]["vacuum_ref"] = vacuum_ref
    model = trained_like(get_model(params).to(DEVICE)).train()
    # Neither H nor C occurs, and one atom of every frame is padding.
    atype = torch.full_like(ATYPE, 2)
    atype[:, -1] = -1
    out = model(COORD, atype, BOX)
    (out["energy"].sum() + out["force"].square().sum()).backward()
    components = element_banks(model)
    assert len(components) == (descriptor is not None) + (fitting is not None)
    trained = 0
    for banks, slots in components:
        for bank in banks:
            rows = bank.grad.reshape(bank.shape[0], -1).abs().amax(dim=1)
            kept = torch.zeros_like(rows, dtype=torch.bool)
            kept[slots[2] - 1] = True
            assert not rows[~kept].any()
            trained += int(rows[kept].any())
    assert trained


def test_merge_aware_training_acts_in_training_only() -> None:
    """MAT drops the element weights of both components in the same frames."""
    moe = {"n_experts": 2, "mat_prob": 0.5}
    model = trained_like(get_model(model_params(moe, moe)).to(DEVICE))

    def energies(count):
        return torch.stack(
            [model(COORD, ATYPE, BOX)["energy"].detach() for _ in range(count)]
        )

    model.eval()
    settled = energies(4)
    # Evaluation uses the weights of every element, so repeated forwards
    # agree to the reduction order of the device.
    assert (settled - settled[0]).abs().max().item() < 1e-12
    torch.manual_seed(0)
    model.train()
    drawn = energies(16)[:, 0, 0]
    # A frame is evaluated either with or without its element weights.
    levels = torch.unique(torch.round(drawn, decimals=8))
    assert levels.numel() == 2
    assert (levels - settled[0, 0, 0]).abs().min().item() < 1e-8


def test_output_statistics_evaluate_the_model() -> None:
    """A model in training mode fits its bias to the energies it evaluates."""
    moe = {"n_experts": 2, "mat_prob": 0.9}
    params = model_params(moe, moe)
    # Without a preset, the bias follows the statistics of the model.
    params.pop("preset_out_bias")
    model = trained_like(get_model(params).to(DEVICE)).eval()
    energy = model(COORD, ATYPE, BOX)["energy"].detach().cpu().numpy()
    sample = [
        {
            "coord": COORD[frame : frame + 1].cpu().numpy(),
            "atype": ATYPE[frame : frame + 1].cpu().numpy(),
            "box": BOX[frame : frame + 1].cpu().numpy(),
            "natoms": np.array([[6, 6, *np.bincount(ATYPE[frame].cpu().numpy())]]),
            "energy": energy[frame : frame + 1],
            "find_energy": np.float32(1.0),
        }
        for frame in range(2)
    ]
    bias = model.get_out_bias().detach().clone()
    model.train()
    model.change_out_bias(lambda: sample, bias_adjust_mode="change-by-statistic")
    # Labelled with its own energies, the model keeps its bias.
    torch.testing.assert_close(model.get_out_bias(), bias, atol=1e-8, rtol=0)


def test_merge_entry_point(tmp_path) -> None:
    params = model_params(EXPERTS, EXPERTS)
    model = trained_like(get_model(copy.deepcopy(params)).to(DEVICE).eval())
    checkpoint = tmp_path / "moe.pt"
    save_checkpoint(checkpoint, model, params)
    state_cs = torch.tensor([[1.0, 2.0]] * 2, device=DEVICE)
    expected = model(COORD, ATYPE, BOX, charge_spin=state_cs)

    # Keeping the weights of every element reproduces the model.
    moe_merge(str(checkpoint), str(tmp_path / "all.pt"), ["all"], charge_spin=[1, 2])
    merged, merged_params = load_checkpoint(tmp_path / "all.pt")
    for key in ("descriptor", "fitting_net"):
        assert merged_params[key]["moe"]["n_experts"] == 0
        assert merged_params[key]["moe"]["element"] == ["H", "C", "O"]
    assert merged_params["descriptor"]["default_chg_spin"] == [1.0, 2.0]
    for kwargs in ({}, {"charge_spin": state_cs}):
        assert_same_output(merged(COORD, ATYPE, BOX, **kwargs), expected)

    # Chosen elements keep their weights; the merge is exact on their systems.
    moe_merge(str(checkpoint), str(tmp_path / "ho.pt"), ["H", "O"], charge_spin=[1, 2])
    merged, merged_params = load_checkpoint(tmp_path / "ho.pt")
    assert merged_params["descriptor"]["moe"]["element"] == ["H", "O"]
    atype = torch.tensor([[0, 2, 2, 0, 0, 0], [2, 2, 0, 0, 2, 2]], device=DEVICE)
    assert_same_output(
        merged(COORD, atype, BOX), model(COORD, atype, BOX, charge_spin=state_cs)
    )

    # Without elements the merged model has no MoE; the
    # command line reaches the same merge.
    main(["moe-merge", str(checkpoint), "-o", str(tmp_path / "plain.pt")])
    merged, merged_params = load_checkpoint(tmp_path / "plain.pt")
    assert "moe" not in merged_params["descriptor"]
    assert "moe" not in merged_params["fitting_net"]
    assert not expert_parameters(merged)
    plain = model_params()
    assert {
        name: tuple(value.shape) for name, value in merged.state_dict().items()
    } == {
        name: tuple(value.shape)
        for name, value in get_model(plain).state_dict().items()
    }


def test_merge_at_another_state_keeps_the_reference(tmp_path) -> None:
    """Merging for another state keeps the reference the preset resolved.

    The merge writes the state as the inference default of the merged model,
    but the reference of the preset belongs to the model: the merged model
    evaluates a structure at that state exactly as the trained model did.
    """
    params = model_params(EXPERTS, EXPERTS)
    model = trained_like(get_model(copy.deepcopy(params)).to(DEVICE).eval())
    checkpoint = tmp_path / "moe.pt"
    save_checkpoint(checkpoint, model, params)
    state_cs = torch.tensor([[1.0, 2.0]] * 2, device=DEVICE)

    moe_merge(str(checkpoint), str(tmp_path / "ion.pt"), ["all"], charge_spin=[1, 2])
    merged, merged_params = load_checkpoint(tmp_path / "ion.pt")
    assert merged_params["descriptor"]["default_chg_spin"] == [1.0, 2.0]
    # the resolved reference of the preset does not follow the default
    np.testing.assert_array_equal(
        merged_params["preset_out_bias"]["@charge_spin"], [[0.0, 1.0]] * 3
    )
    assert_same_output(
        merged(COORD, ATYPE, BOX), model(COORD, ATYPE, BOX, charge_spin=state_cs)
    )
    # A continued training saves the resolved reference with the model, and a
    # reload evaluates the same structure the same way.
    restored = BaseModel.deserialize(merged.serialize()).to(DEVICE).eval()
    np.testing.assert_array_equal(
        restored.atomic_model.vacuum_charge_spin.cpu().numpy(), [[0.0, 1.0]] * 3
    )
    assert_same_output(
        restored(COORD, ATYPE, BOX), model(COORD, ATYPE, BOX, charge_spin=state_cs)
    )


def test_merged_models_freeze(tmp_path) -> None:
    """The ordinary freeze exports a merged model, with and without kept elements."""
    params = model_params(EXPERTS, EXPERTS)
    model = trained_like(get_model(copy.deepcopy(params)).to(DEVICE).eval())
    checkpoint = tmp_path / "moe.pt"
    save_checkpoint(checkpoint, model, params)
    for name, elements in (("plain", None), ("elements", ["H", "O"])):
        merged_file = tmp_path / f"{name}.pt"
        moe_merge(str(checkpoint), str(merged_file), elements)
        merged, _ = load_checkpoint(merged_file)
        assert merged.atomic_model.fitting_net.needs_vacuum_descriptor()
        expected = merged(COORD, ATYPE, BOX)
        frozen_file = tmp_path / f"{name}.pte"
        freeze(str(merged_file), str(frozen_file))
        potential = DeepPot(str(frozen_file))
        for frame in range(COORD.shape[0]):
            energy, force, _ = potential.eval(
                COORD[frame : frame + 1].cpu().numpy(),
                BOX[frame : frame + 1].cpu().numpy(),
                ATYPE[frame].cpu().numpy(),
            )
            np.testing.assert_allclose(
                energy.ravel(),
                expected["energy"][frame].detach().cpu().numpy().ravel(),
                atol=1e-9,
            )
            np.testing.assert_allclose(
                force.reshape(-1),
                expected["force"][frame].detach().cpu().numpy().reshape(-1),
                atol=1e-9,
            )


def test_merge_selects_the_task_of_the_head(tmp_path) -> None:
    """``--task`` merges the branch it names, with the task of that branch."""
    models, params = multitask_models()
    for seed, model in enumerate(models.values()):
        model.eval()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for parameter in branch_local_parameters(model):
                parameter.copy_(
                    0.05 * torch.randn(parameter.shape, generator=generator)
                )
    with torch.no_grad():
        shared = set(map(id, branch_local_parameters(models["gas"]))) | set(
            map(id, branch_local_parameters(models["bulk"]))
        )
        generator = torch.Generator(device="cpu").manual_seed(7)
        for parameter in expert_parameters(models["gas"]).values():
            if id(parameter) not in shared:
                parameter.copy_(
                    0.05 * torch.randn(parameter.shape, generator=generator)
                )
    checkpoint = tmp_path / "moe.pt"
    save_checkpoint(checkpoint, models, {"model_dict": params})
    with pytest.raises(ValueError, match="--task"):
        moe_merge(str(checkpoint), str(tmp_path / "none.pt"))
    energies = {}
    for head in ("gas", "bulk"):
        merged_file = tmp_path / f"{head}.pt"
        moe_merge(str(checkpoint), str(merged_file), ["all"], head=head)
        merged, merged_params = load_checkpoint(merged_file)
        assert merged_params["task"] == head
        expected = models[head](COORD, ATYPE, BOX)
        actual = merged(COORD, ATYPE, BOX)
        assert_same_output(actual, expected, atol=1e-5)
        energies[head] = actual["energy"][0].item()
    assert abs(energies["gas"] - energies["bulk"]) > 1e-4
    single = tmp_path / "gas.pt"
    with pytest.raises(ValueError, match="single-task"):
        moe_merge(str(single), str(tmp_path / "again.pt"), ["all"], head="gas")


def test_bias_change_selects_the_branch(tmp_path) -> None:
    """A bias change of one branch keeps the other branch and every task."""
    models, params = multitask_models()
    checkpoint, output = tmp_path / "moe.pt", tmp_path / "bias.pt"
    save_checkpoint(checkpoint, models, {"model_dict": params})
    change_bias(
        str(checkpoint),
        bias_value=[0.1, 0.2, 0.3],
        model_branch="gas",
        output=str(output),
    )
    before, after = (
        torch.load(path, map_location=DEVICE, weights_only=True)["model"]
        for path in (checkpoint, output)
    )
    changed = [
        key
        for key in before
        if key != "_extra_state" and not torch.equal(before[key], after[key])
    ]
    assert changed == ["model.gas.atomic_model.out_bias"]
    np.testing.assert_allclose(
        after[changed[0]].detach().cpu().numpy().reshape(-1), [0.1, 0.2, 0.3]
    )


def test_branches_share_networks_and_keep_their_task() -> None:
    models, _ = multitask_models("zbl")
    gas, bulk = (
        learned_atomic_models(models[key].atomic_model)[0] for key in ("gas", "bulk")
    )
    assert gas.descriptor.blocks is bulk.descriptor.blocks
    assert gas.descriptor.expert_router is bulk.descriptor.expert_router
    assert gas.fitting_net.nets is bulk.fitting_net.nets
    assert gas.fitting_net.expert_router is bulk.fitting_net.expert_router
    for mine, other in zip(
        branch_local_parameters(models["gas"]),
        branch_local_parameters(models["bulk"]),
        strict=True,
    ):
        assert mine is not other
    assert gas.descriptor.contact_clamp is not None
    assert gas.descriptor.contact_clamp is not bulk.descriptor.contact_clamp


def test_tasks_never_share_a_compiled_graph() -> None:
    models, _ = multitask_models()
    gas, bulk = (_get_model_structure_key(models[key]) for key in ("gas", "bulk"))
    # The keys differ by the task of the branches, which the trace bakes in.
    assert gas[-1] and gas[-1] != bulk[-1]
    assert gas[-3:-1] == bulk[-3:-1]


@REQUIRES_SUPPORTED_COMPILE
def test_frame_states_compile_once_for_several_frames() -> None:
    """One compiled graph of the frame states serves every count of several frames."""
    frames = torch.tensor(
        [[0, 1], [0, 1], [1, 2], [1, 2], [2, 1], [2, 1], [2, 3]],
        dtype=torch.float64,
        device="cpu",
    )
    traced = make_fx(
        frame_states, tracing_mode="symbolic", _allow_non_fake_inputs=True
    )(frames)
    compiled = torch.compile(traced, backend="eager", dynamic=True)
    compiled(frames.clone())
    with torch.compiler.set_stance("fail_on_recompile"):
        for count in (5, 3, 2):
            batch = frames[:count].clone()
            for got, expected in zip(compiled(batch), frame_states(batch), strict=True):
                assert torch.equal(got, expected)


def test_export_refuses_a_router(tmp_path) -> None:
    model = get_model(model_params(EXPERTS))
    with pytest.raises(ValueError, match="moe-merge"):
        deserialize_to_file(str(tmp_path / "model.pt2"), {"model": model.serialize()})


def test_spin_free_branch_holds_its_own_state() -> None:
    """A spin-free branch shares the common layers of its native-spin base."""
    models, params = multitask_models(spin={"scheme": "native", "use_spin": ["O"]})
    shared = ModelWrapper(models).state_dict()
    own = ModelWrapper(
        {key: get_model(copy.deepcopy(value)) for key, value in params.items()}
    ).state_dict()
    assert shared.keys() == own.keys()
    gas, bulk = (
        learned_atomic_models(models[key].atomic_model)[0].descriptor
        for key in ("gas", "bulk")
    )
    assert bulk.spin_embedding is None
    assert bulk.env_seed_embedding.g_layer1 is gas.env_seed_embedding.g_layer1


def test_merge_arguments() -> None:
    flags = parse_args(["moe-merge", "model.pt"])
    assert flags.elements is None and flags.charge_spin is None
    flags = parse_args(
        ["moe-merge", "model.pt", "--elements", "H", "C", "--charge-spin", "1", "2"]
    )
    assert flags.elements == ["H", "C"]
    assert flags.charge_spin == [1.0, 2.0]
    assert parse_args(["moe-merge", "model.pt", "--elements", "all"]).elements == [
        "all"
    ]


def test_merge_refusals(tmp_path) -> None:
    params = model_params(EXPERTS, EXPERTS)
    checkpoint = tmp_path / "moe.pt"
    save_checkpoint(checkpoint, get_model(copy.deepcopy(params)), params)
    output = str(tmp_path / "merged.pt")
    with pytest.raises(ValueError, match="stands alone"):
        moe_merge(str(checkpoint), output, ["all", "H"])
    with pytest.raises(ValueError, match="single-task checkpoint"):
        moe_merge(str(checkpoint), output, head="water")
    with pytest.raises(ValueError, match="no weights of their own"):
        moe_merge(str(checkpoint), output, ["Xx"])
    # Without a router nothing reads the charge and spin multiplicity.
    params = model_params({"n_experts": 0}, {"n_experts": 0})
    save_checkpoint(checkpoint, get_model(copy.deepcopy(params)), params)
    with pytest.raises(ValueError, match="no router"):
        moe_merge(str(checkpoint), output, charge_spin=[1, 2])
    assert not (tmp_path / "merged.pt").exists()


def test_magnetic_types_separate_compiled_graphs() -> None:
    """Native-spin branches with different magnetic types bake different masks."""
    models, _ = multitask_models(
        spin={"scheme": "native", "use_spin": ["O"]},
        bulk_spin={"scheme": "native", "use_spin": ["H"]},
    )
    gas, bulk = (_get_model_structure_key(models[key]) for key in ("gas", "bulk"))
    assert gas[1] != bulk[1]


def test_task_increments_take_the_adam_path() -> None:
    """HybridMuon grows the task increments only where a task labels them.

    An orthogonalized step moves every row of a matrix at the same rate, so a
    task increment on the Muon route grows into the directions no task covers.
    The descriptor and the fitting declare their increments, and the decoupled
    decay pulls the ones no data push back toward the shared weights.
    """
    models, _ = multitask_models()
    model = models["gas"]
    patterns = tuple(pattern.lower() for pattern in adam_route_patterns([model]))
    assert any(pattern.endswith("descriptor.task_weight.") for pattern in patterns)
    assert any(pattern.endswith("fitting_net.task_weight.") for pattern in patterns)

    def routed(name: str) -> bool:
        return any(pattern in name.lower() for pattern in patterns)

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".task_weight." in name:
            assert routed(name), name
        elif ".task_bias." in name or "task_embedding" in name:
            # Leaf names that name no matrix keep the undecayed Adam path.
            assert not routed(name), name
        elif "element_weight" in name or "expert_weight" in name:
            assert not routed(name), name


def _plain_parameters() -> dict:
    """Model parameters of the MoE model without the ``moe`` options."""
    model = model_params(EXPERTS, EXPERTS)
    for region in ("descriptor", "fitting_net"):
        model[region].pop("moe")
    return model


def test_a_plain_state_is_the_start_of_every_moe_model() -> None:
    """Loading a plain state leaves the MoE model the plain function.

    Every term MoE adds starts at zero, so a model trained without MoE is the
    state a model with MoE starts from: the increments keep their zeros and
    the energy and force are those of the plain model.
    """
    target = model_params(EXPERTS, EXPERTS)
    plain_params = _plain_parameters()
    with pytest.raises(RuntimeError, match="Missing key"):
        get_model(copy.deepcopy(target)).to(DEVICE).load_state_dict(
            get_model(copy.deepcopy(plain_params)).to(DEVICE).state_dict()
        )
    moe = get_model(copy.deepcopy(target)).to(DEVICE).eval()
    plain = get_model(copy.deepcopy(plain_params)).to(DEVICE).eval()
    plain_state = plain.state_dict()
    moe_state = moe.state_dict()
    check_expert_state(
        [key for key in moe_state if key not in plain_state],
        [key for key in plain_state if key not in moe_state],
    )
    moe.load_state_dict(plain_state, strict=False)
    # The increments keep their zeros, and the function is the plain one.
    for name, parameter in moe.named_parameters():
        if is_expert_parameter(name) and ".expert_router." not in name:
            assert float(parameter.abs().max()) == 0.0, name
    assert_same_output(moe(COORD, ATYPE, BOX), plain(COORD, ATYPE, BOX), atol=1e-10)
