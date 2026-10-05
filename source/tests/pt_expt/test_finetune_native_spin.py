# SPDX-License-Identifier: LGPL-3.0-or-later
"""Spin-free checkpoint transfer through both native-spin training backends."""

import json
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import pytest
import torch

from deepmd.pt.entrypoints.main import main as pt_main
from deepmd.pt.model.model import get_model as get_pt_model
from deepmd.pt.train.wrapper import ModelWrapper as PTModelWrapper
from deepmd.pt.utils.finetune import (
    _learned_part_path,
)
from deepmd.pt.utils.spin_finetune import (
    get_native_spin_finetune_state,
)
from deepmd.pt_expt.entrypoints.main import (
    main,
)
from deepmd.pt_expt.model import (
    get_model,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)
from deepmd.utils.model_preset import (
    get_model_preset,
)

from .test_finetune import (
    _make_config,
    _native_spin_config,
)

_BACKENDS = [("pt", "dpa4"), ("pt_expt", "dpa4"), ("pt_expt", "dpa4c")]


def _config(descriptor: str, native: bool, bridged: bool = False) -> dict:
    config = _native_spin_config(descriptor, [True, False])
    if not native:
        config.pop("spin")
    if bridged:
        config["bridging_method"] = "zbl"
    config["data_stat_nbatch"] = 1
    return config


def _build(backend: str, config: dict) -> torch.nn.Module:
    builder = get_pt_model if backend == "pt" else get_model
    return builder(deepcopy(config)).to(DEVICE)


def _learned(model: torch.nn.Module) -> torch.nn.Module:
    return model.get_submodule(_learned_part_path(model).removesuffix("."))


def _pretrained(backend: str, config: dict) -> torch.nn.Module:
    """Build a geometric model whose nonzero forces make transfer observable."""
    model = _build(backend, config)
    generator = torch.Generator(device=DEVICE).manual_seed(31)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                0.02
                * torch.randn(
                    parameter.shape,
                    dtype=parameter.dtype,
                    device=parameter.device,
                    generator=generator,
                )
            )
        descriptor = _learned(model).descriptor
        for name, start, stop in (("mean", -0.2, 0.2), ("stddev", 0.8, 1.2)):
            array = getattr(descriptor, name)
            array.copy_(
                torch.linspace(
                    start, stop, array.numel(), dtype=array.dtype, device=array.device
                ).reshape(array.shape)
            )
    return model


def _inputs() -> dict[str, torch.Tensor]:
    return {
        "coord": torch.tensor(
            [[[0.0, 0.0, 0.0], [1.3, 0.2, 0.1], [0.3, 1.5, 0.4]]],
            dtype=torch.float64,
            device=DEVICE,
        ),
        "atype": torch.tensor([[0, 1, 0]], device=DEVICE),
        "box": (6.0 * torch.eye(3, dtype=torch.float64, device=DEVICE)).reshape(1, 9),
        "spin": torch.tensor(
            [[[0.6, 0.2, -0.7], [0.1, 0.5, 0.1], [-0.4, 0.8, 0.3]]],
            dtype=torch.float64,
            device=DEVICE,
        ),
    }


@pytest.mark.parametrize("backend, descriptor", _BACKENDS)
@pytest.mark.parametrize("bridged", [False, True])
def test_spin_free_transfer_preserves_function_and_trains_spin(
    backend: str, descriptor: str, bridged: bool
) -> None:
    source = _pretrained(backend, _config(descriptor, False, bridged))
    target = _build(backend, _config(descriptor, True, bridged))
    target.load_state_dict(get_native_spin_finetune_state(target, source))
    inputs = _inputs()
    spin = inputs.pop("spin")
    expected = source(**inputs, do_atomic_virial=True)
    assert expected["force"].abs().max() > 1e-8
    for scale in (0.0, 1.0, 2.0):
        actual = target(**inputs, spin=scale * spin, do_atomic_virial=True)
        for key in ("energy", "force", "virial", "atom_virial"):
            torch.testing.assert_close(
                actual[key], expected[key], rtol=1e-11, atol=1e-11
            )
        torch.testing.assert_close(
            actual["force_mag"], torch.zeros_like(actual["force_mag"]), rtol=0, atol=0
        )

    spin_params = [param for name, param in target.named_parameters() if "spin" in name]
    optimizer = torch.optim.SGD(spin_params, lr=1e-3)
    (actual["force_mag"] - 0.2).square().sum().backward()
    assert any(
        param.grad is not None and param.grad.abs().max() > 1e-8
        for param in spin_params
    )
    optimizer.step()
    trained = target(**inputs, spin=spin)
    assert trained["force_mag"].abs().max() > 1e-10
    assert torch.count_nonzero(trained["force_mag"][:, 1]) == 0


@pytest.mark.parametrize("backend, descriptor", _BACKENDS)
@pytest.mark.parametrize("random_fitting", [False, True])
def test_spin_free_transfer_preserves_target_bias_and_bridging(
    backend: str, descriptor: str, random_fitting: bool
) -> None:
    source = _pretrained(backend, _config(descriptor, False))
    target = _build(backend, _config(descriptor, True, bridged=True))
    initial = {key: value.clone() for key, value in target.state_dict().items()}
    with torch.no_grad():
        for key, value in source.state_dict().items():
            if key.rsplit(".", 1)[-1] in {"out_bias", "out_std", "bias_atom_e"}:
                value.fill_(1234.0)
    state = get_native_spin_finetune_state(
        target, source, random_fitting=random_fitting
    )
    target.load_state_dict(state)
    for key, value in state.items():
        if key.rsplit(".", 1)[-1] in {"out_bias", "out_std", "bias_atom_e"} or (
            random_fitting and ".fitting_net." in key
        ):
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    source_descriptor = _learned(source).descriptor.state_dict()
    target_descriptor = _learned(target).descriptor.state_dict()
    for key, value in source_descriptor.items():
        if value.shape == target_descriptor[key].shape:
            torch.testing.assert_close(value, target_descriptor[key], rtol=0, atol=0)
    np.testing.assert_array_equal(target.spin.use_spin, [True, False])


@pytest.mark.parametrize("backend, descriptor", _BACKENDS)
@pytest.mark.parametrize("width", [4, 16])
def test_spin_free_transfer_rejects_unrelated_shape_change(
    backend: str, descriptor: str, width: int
) -> None:
    source = _build(backend, _config(descriptor, False))
    config = _config(descriptor, True)
    config["fitting_net"]["neuron"] = [width]
    target = _build(backend, config)
    with pytest.raises(ValueError, match="Incompatible spin-free transfer"):
        get_native_spin_finetune_state(target, source)


@pytest.mark.parametrize("resnet_dt", [False, True])
@pytest.mark.parametrize("residual_side", ["source", "target"])
@pytest.mark.parametrize("width_multiple", [1, 2])
def test_dpa4c_transfer_rejects_changed_input_residual(
    resnet_dt: bool, residual_side: str, width_multiple: int
) -> None:
    source_config = _config("dpa4c", False)
    target_config = _config("dpa4c", True)
    width_config = source_config if residual_side == "source" else target_config
    width = _learned(_build("pt_expt", width_config)).descriptor.get_dim_out()
    for config in (source_config, target_config):
        config["fitting_net"].update(
            neuron=[width_multiple * width], resnet_dt=resnet_dt
        )
    source = _pretrained("pt_expt", source_config)
    target = _build("pt_expt", target_config)
    initial = {key: value.clone() for key, value in target.state_dict().items()}
    with pytest.raises(ValueError, match=r"residual.*input"):
        get_native_spin_finetune_state(target, source)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)

    # A random fitting net does not inherit the source function or its residual.
    state = get_native_spin_finetune_state(target, source, random_fitting=True)
    target.load_state_dict(state)
    for key, value in state.items():
        if ".fitting_net." in key:
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)


@pytest.mark.parametrize("fitting_type", ["ener", "dpa4_ener", "sezm_ener"])
@pytest.mark.parametrize("neuron", [[], [8], [8, 8]])
def test_dpa4c_transfer_fitting_input_projection(
    fitting_type: str, neuron: list[int]
) -> None:
    source_config = _config("dpa4c", False)
    source_config["fitting_net"].update(type=fitting_type, neuron=neuron)
    target_config = deepcopy(source_config)
    target_config["spin"] = {"scheme": "native", "use_spin": [True, False]}
    source = _pretrained("pt_expt", source_config)
    target = _build("pt_expt", target_config)
    # DPA4C's descriptor selects a shared fitting network at model construction.
    assert _learned(source).fitting_net.nets.ndim == 0
    assert _learned(target).fitting_net.nets.ndim == 0
    state = get_native_spin_finetune_state(target, source)
    assert all(not value.requires_grad for value in state.values())
    target.load_state_dict(state)
    inputs = _inputs()
    spin = inputs.pop("spin")
    expected = source(**inputs, do_atomic_virial=True)
    assert expected["force"].abs().max() > 1e-8
    for scale in (0.0, 1.0, 2.0):
        actual = target(**inputs, spin=scale * spin, do_atomic_virial=True)
        for key in ("energy", "force", "virial", "atom_virial"):
            torch.testing.assert_close(
                actual[key], expected[key], rtol=1e-11, atol=1e-11
            )
        assert torch.count_nonzero(actual["force_mag"]) == 0

    (actual["force_mag"] - 0.2).square().sum().backward()
    gate = _learned(target).descriptor.spin.spin_gate
    assert gate.grad is not None and gate.grad.abs().max() > 1e-8
    with torch.no_grad():
        gate.add_(-1e-3 * gate.grad)
    trained = target(**inputs, spin=spin)
    assert trained["force_mag"].abs().max() > 1e-10
    assert torch.count_nonzero(trained["force_mag"][:, 1]) == 0

    restored = type(target).deserialize(target.serialize()).to(DEVICE)
    reloaded = restored(**inputs, spin=spin)
    for key in ("energy", "force", "virial", "force_mag"):
        torch.testing.assert_close(reloaded[key], trained[key], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    "source_type, target_type", [("ener", "dpa4_ener"), ("dpa4_ener", "ener")]
)
def test_dpa4c_transfer_rejects_changed_fitting_family(
    source_type: str, target_type: str
) -> None:
    source_config = _config("dpa4c", False)
    target_config = _config("dpa4c", True)
    for config, fitting_type in (
        (source_config, source_type),
        (target_config, target_type),
    ):
        config["fitting_net"].update(
            type=fitting_type, neuron=[16 if fitting_type == "ener" else 8]
        )
    source = _build("pt_expt", source_config)
    target = _build("pt_expt", target_config)
    with pytest.raises(ValueError, match="matching fitting network types"):
        get_native_spin_finetune_state(target, source)


@pytest.mark.parametrize("fitting_type", ["ener", "dpa4_ener"])
def test_dpa4c_spin_transfer_preserves_fitting_parameter_tail(
    fitting_type: str,
) -> None:
    source_config = _config("dpa4c", False)
    source_config["fitting_net"].update(type=fitting_type, numb_fparam=2, numb_aparam=1)
    target_config = deepcopy(source_config)
    target_config["spin"] = {"scheme": "native", "use_spin": [True, False]}
    source = _pretrained("pt_expt", source_config)
    target = _build("pt_expt", target_config)
    target.load_state_dict(get_native_spin_finetune_state(target, source))
    inputs = _inputs()
    spin = inputs.pop("spin")
    inputs["fparam"] = torch.tensor([[0.2, -0.1]], dtype=torch.float64, device=DEVICE)
    inputs["aparam"] = torch.tensor(
        [[[0.3], [0.5], [0.7]]], dtype=torch.float64, device=DEVICE
    )
    expected = source(**inputs)
    actual = target(**inputs, spin=spin)
    for key in ("energy", "force", "virial"):
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("grade", ["nano", "mini", "neo", "air", "plus"])
def test_dpa4c_preset_spin_free_transfer(grade: str) -> None:
    source_config = get_model_preset(f"dpa4c-{grade}-v20260911")
    source_config["descriptor"]["seed"] = 17
    source_config["fitting_net"]["seed"] = 19
    target_config = deepcopy(source_config)
    target_config["spin"] = {"scheme": "native", "use_spin": ["Fe"]}
    source = _pretrained("pt_expt", source_config)
    target = _build("pt_expt", target_config)
    target.load_state_dict(get_native_spin_finetune_state(target, source))
    inputs = _inputs()
    spin = inputs.pop("spin")
    inputs["atype"] = torch.tensor(
        [[source.get_type_map().index(name) for name in ("Fe", "C", "Fe")]],
        device=DEVICE,
    )
    expected = source(**inputs)
    actual = target(**inputs, spin=spin)
    for key in ("energy", "force", "virial"):
        torch.testing.assert_close(actual[key], expected[key], rtol=2e-5, atol=2e-5)
    assert torch.count_nonzero(actual["force_mag"]) == 0


@pytest.mark.parametrize(
    "backend, descriptor, fitting_type",
    [
        ("pt", "dpa4", "dpa4_ener"),
        ("pt_expt", "dpa4", "dpa4_ener"),
        ("pt_expt", "dpa4c", "ener"),
        ("pt_expt", "dpa4c", "dpa4_ener"),
    ],
)
@pytest.mark.parametrize("bridged", [False, True])
@pytest.mark.parametrize("multitask", [False, True])
def test_spin_free_native_finetune_cli(
    backend: str,
    descriptor: str,
    fitting_type: str,
    bridged: bool,
    multitask: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_config = _config(descriptor, False)
    source_config["fitting_net"]["type"] = fitting_type
    source = _pretrained(backend, source_config)
    wrapper_cls = PTModelWrapper if backend == "pt" else ModelWrapper
    wrapper = wrapper_cls(
        {"magnetic": source} if multitask else source,
        model_params={"model_dict": {"magnetic": source_config}}
        if multitask
        else source_config,
    )
    wrapper.train_infos["step"] = 100
    checkpoint = tmp_path / "pretrained.pt"
    torch.save(
        {
            "model": wrapper.state_dict(),
            "optimizer": {"source_only": True},
            "ema": {"source_only": True},
        },
        checkpoint,
    )

    inputs = _inputs()
    spin = inputs.pop("spin")
    expected = source(**inputs)
    system = tmp_path / "data" / "data_0"
    arrays = system / "set.000"
    arrays.mkdir(parents=True)
    (system / "type.raw").write_text("0\n1\n0\n")
    (system / "type_map.raw").write_text("Fe\nC\n")
    labels = {
        "coord": inputs["coord"],
        "box": inputs["box"],
        "spin": spin,
        "energy": expected["energy"] + 1.75,
        "force": expected["force"],
        "force_mag": torch.full_like(spin, 0.2),
    }
    for key, value in labels.items():
        data = value.detach().cpu().numpy().reshape(1, -1)
        np.save(arrays / f"{key}.npy", np.repeat(data, 2, axis=0))

    target_config = _config(descriptor, True, bridged)
    target_config["fitting_net"]["type"] = fitting_type
    if bridged:
        target_config["preset_out_bias"] = {"energy": {"Fe": 1.25, "C": -0.75}}
    config = _make_config(str(system.parent), target_config, numb_steps=2)
    config["loss"] = {
        "type": "ener_spin",
        "start_pref_e": 1.0,
        "limit_pref_e": 1.0,
        "start_pref_fr": 1.0,
        "limit_pref_fr": 1.0,
        "start_pref_fm": 1.0,
        "limit_pref_fm": 1.0,
    }
    config["training"].update(enable_ema=True, ema_decay=0.9)
    if multitask:
        config["model"] = {"model_dict": {"magnetic": target_config}}
        config["loss_dict"] = {"magnetic": config.pop("loss")}
        config["training"]["data_dict"] = {
            "magnetic": {
                key: config["training"].pop(key)
                for key in ("training_data", "validation_data")
            }
        }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(config))
    monkeypatch.chdir(tmp_path)
    entrypoint = pt_main if backend == "pt" else main
    entrypoint(
        [
            "train",
            str(input_path),
            "--finetune",
            str(checkpoint),
            "--skip-neighbor-stat",
        ]
    )
    saved = torch.load(
        tmp_path / "model.ckpt.pt", map_location=DEVICE, weights_only=True
    )
    params = saved["model"]["_extra_state"]["model_params"]
    model_params = params["model_dict"]["magnetic"] if multitask else params
    assert model_params["spin"]["scheme"] == "native"
    assert model_params["spin"]["use_spin"] == [True, False]
    # PT checkpoints record the zero-based step; pt_expt records the step count.
    assert saved["model"]["_extra_state"]["train_infos"]["step"] == (
        1 if backend == "pt" else 2
    )
    assert "source_only" not in saved["optimizer"]
    assert "source_only" not in saved["ema"]
    trained = _build(backend, model_params)
    loaded = wrapper_cls(
        {"magnetic": trained} if multitask else trained, model_params=params
    )
    loaded.load_state_dict(saved["model"])
    bias = trained.get_out_bias().reshape(-1)
    if bridged:
        torch.testing.assert_close(
            bias,
            torch.tensor([1.25, -0.75], dtype=torch.float64, device=DEVICE),
            rtol=0,
            atol=0,
        )
    else:
        torch.testing.assert_close(
            2.0 * bias[0] + bias[1],
            torch.tensor(1.75, dtype=torch.float64, device=DEVICE),
            rtol=1e-11,
            atol=1e-11,
        )
    assert trained(**inputs, spin=spin)["force_mag"].abs().max() > 1e-10
