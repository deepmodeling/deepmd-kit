# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASE checkpoint export and eager-inference behavior."""

import copy
import json
import logging
import os
import zipfile
from pathlib import (
    Path,
)
from unittest import (
    mock,
)

import numpy as np
import pytest
import torch
from ase import (
    Atoms,
)

from deepmd.calculator import (
    DP,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.pt.entrypoints import (
    freeze_pt2,
)
from deepmd.pt.utils import env as pt_env
from deepmd.pt_expt.utils import (
    env,
    serialization,
)

from ..pt.model.test_sezm_export import (
    _tiny_sezm_model_params,
)
from .model.test_dpa4_export import (
    _DPA4_CONFIG,
)
from .model.test_dpa4c_graph_lower import _config as dpa4c_config
from .model.test_export_with_comm import (
    _DPA3_CONFIG,
)
from .model.test_graph_export_with_comm import (
    DPA1_CONFIG,
    DPA2_CONFIG,
)
from .test_dp_freeze import (
    model_se_e2_a,
)

_LEVELS = ("DP_CUDA_INFER", "DP_TRITON_INFER", "DP_CUTILE_INFER", "DP_CUTE_INFER")


@pytest.fixture(autouse=True, params=["cpu"])
def inference_device(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> str:
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    for module in (pt_env, env, freeze_pt2):
        monkeypatch.setattr(module, "DEVICE", torch.device(device))
    for name in (*_LEVELS, "DP_COMPILE_INFER"):
        monkeypatch.delenv(name, raising=False)
    return device


def _config(kind: str) -> dict:
    if kind in ("pt", "pt-dpa4"):
        config = _tiny_sezm_model_params()
        if kind == "pt-dpa4":
            config["type"] = config["descriptor"]["type"] = "dpa4"
    elif kind == "dpa4":
        config = copy.deepcopy(_DPA4_CONFIG)
        config["descriptor"].update(channels=4, n_radial=3, lmax=1, n_blocks=1)
        config["fitting_net"]["neuron"] = [8]
    elif kind == "dpa4c":
        config = dpa4c_config()
        config["descriptor"].update(channels=8, lmax=2, precision="float32")
        config["fitting_net"]["neuron"] = [8]
        config["fitting_net"]["precision"] = "float32"
    else:
        config = copy.deepcopy(
            {
                "dpa1": DPA1_CONFIG,
                "dpa2": DPA2_CONFIG,
                "dpa3": _DPA3_CONFIG,
                "dense": model_se_e2_a,
                "pt-dense": model_se_e2_a,
            }[kind]
        )
    config["type_map"] = ["O", "H"]
    if kind in ("dense", "pt-dense"):
        config["descriptor"]["sel"] = config["descriptor"]["sel"][:2]
    if env.DEVICE.type == "cuda":
        config["descriptor"]["precision"] = "float32"
        config["fitting_net"]["precision"] = "float32"
        if kind in ("pt", "pt-dpa4", "dpa4"):
            config["descriptor"]["channels"] = 32
    return config


def _write_checkpoint(
    path: Path,
    kind: str,
    head: str | None = None,
    *,
    config: dict | None = None,
) -> Path:
    if kind.startswith("pt"):
        from deepmd.pt.model.model import (
            get_model,
        )
        from deepmd.pt.train.wrapper import (
            ModelWrapper,
        )
    else:
        from deepmd.pt_expt.model.get_model import (
            get_model,
        )
        from deepmd.pt_expt.train.wrapper import (
            ModelWrapper,
        )

    params = _config(kind) if config is None else copy.deepcopy(config)
    model = get_model(copy.deepcopy(params)).to("cpu").eval()
    generator = torch.Generator(device="cpu").manual_seed(17)
    with torch.no_grad():
        for parameter in model.parameters():
            if torch.count_nonzero(parameter) == 0:
                parameter.copy_(
                    0.05
                    * torch.randn(
                        parameter.shape,
                        generator=generator,
                        dtype=parameter.dtype,
                        device="cpu",
                    )
                )
    wrapper = ModelWrapper(
        model if head is None else {head: model},
        model_params=params if head is None else {"model_dict": {head: params}},
    )
    torch.save({"model": wrapper.state_dict()}, path)
    return path


@pytest.mark.parametrize("kind", ["pt", "pt-dpa4", "dpa4", "dpa4c", "dpa1", "dpa2"])
def test_checkpoint_default_warns_without_export(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt", kind)
    with caplog.at_level(logging.WARNING, logger="deepmd.calculator"):
        calculator = DP(checkpoint)

    assert calculator.dp.deep_eval.model_path == str(checkpoint)
    assert not checkpoint.with_name("model-ase_single.pt2").exists()
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "deepmd.calculator"
    ]
    assert len(warnings) == 1
    assert "SUBSTANTIALLY slower" in warnings[0]
    assert "freeze=True" in warnings[0]
    assert "manually" in warnings[0]
    assert "debugging" in warnings[0]


@pytest.mark.parametrize(
    ("kind", "head"),
    [("pt", "branch"), ("dpa4", "branch"), ("dpa4c", "branch"), ("dpa4", None)],
)
@pytest.mark.parametrize("override", [None, ("0", "0"), ("2", "3")])
def test_automatic_freeze_reuses_kernel_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    head: str | None,
    override: tuple[str, str] | None,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt", kind, head=head or "Default")
    monkeypatch.setattr(freeze_pt2, "DEVICE", torch.device("cuda"))
    monkeypatch.setattr(
        "deepmd.pt_expt.kernels.utils.backend_device_type", lambda: "cuda"
    )
    if override is not None:
        for name, value in zip(_LEVELS[:2], override, strict=True):
            monkeypatch.setenv(name, value)
    captured = {}

    def capture(*args: object, **kwargs: object) -> None:
        captured["levels"] = tuple(os.environ.get(name) for name in _LEVELS[:2])
        captured["with_comm"] = kwargs["with_comm"]

    if kind == "pt":
        monkeypatch.setattr(freeze_pt2, "_freeze_sezm_to_pt2", capture)
    else:
        monkeypatch.setattr(serialization, "_deserialize_to_file_pt2", capture)

    def load(model_file: str, **kwargs: object) -> object:
        if model_file.endswith(".pt"):
            return DeepPot(model_file, **kwargs)
        result = mock.Mock()
        result.get_type_map.return_value = ["O", "H"]
        result.get_ntypes.return_value = 2
        return result

    with mock.patch("deepmd.calculator.DeepPot", side_effect=load) as deep_pot:
        DP(checkpoint, freeze=True, head=head)

    expected = override or ("1", "2")
    if kind == "dpa4c":
        expected = ("2", None if override is None else override[1])
    assert captured == {"levels": expected, "with_comm": False}
    assert tuple(os.environ.get(name) for name in _LEVELS[:2]) == (
        override or (None, None)
    )
    assert deep_pot.call_count == 2
    deep_pot.assert_called_with(
        str(tmp_path / "model-ase_single.pt2"),
        neighbor_list=None,
        nlist_backend="auto",
    )


@pytest.mark.parametrize("kind", ["pt-dense", "dense", "dpa3"])
@pytest.mark.parametrize("freeze", [False, True])
def test_ineligible_checkpoint_keeps_eager_inference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, kind: str, freeze: bool
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt", kind)
    with caplog.at_level(logging.WARNING, logger="deepmd.calculator"):
        calculator = DP(checkpoint, freeze=freeze)
    assert calculator.dp.deep_eval.model_path == str(checkpoint)
    assert not any(record.name == "deepmd.calculator" for record in caplog.records)
    assert not checkpoint.with_name("model-ase_single.pt2").exists()


def test_single_task_pt_freeze_ignores_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-task head arguments retain the eager checkpoint's semantics."""
    checkpoint = _write_checkpoint(tmp_path / "model.pt", "pt")
    loaded = DeepPot(str(checkpoint), head="Default")
    export = mock.Mock()
    monkeypatch.setattr(freeze_pt2, "freeze_sezm_to_pt2", export)
    with mock.patch("deepmd.calculator.DeepPot", return_value=loaded):
        DP(checkpoint, head="Default", freeze=True)
    export.assert_called_once_with(
        str(checkpoint),
        str(tmp_path / "model-ase_single.pt2"),
        head=None,
        with_comm=False,
    )


@pytest.mark.parametrize("kind", ["pt", "dpa4", "dpa4c", "dpa1", "dpa2"])
@pytest.mark.parametrize("inference_device", ["cpu", "cuda"], indirect=True)
def test_automatic_freeze_single_artifact_matches_eager(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    kind: str,
    inference_device: str,
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt", kind)
    original = checkpoint.read_bytes()
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.8, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.calc = DP(checkpoint)
    expected = (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
    caplog.clear()
    with mock.patch(
        "torch._inductor.aoti_compile_and_package",
        wraps=torch._inductor.aoti_compile_and_package,
    ) as compile_package:
        atoms.calc = DP(checkpoint, freeze=True)
    assert compile_package.call_count == 1
    if kind == "dpa4c":
        exported = compile_package.call_args.args[0]
        targets = {str(node.target) for node in exported.graph_module.graph.nodes}
        assert any("dpa4c_" in target and "compress" in target for target in targets)
    frozen = tmp_path / "model-ase_single.pt2"
    assert checkpoint.read_bytes() == original
    assert atoms.calc.dp.deep_eval.model_path == str(frozen)
    with zipfile.ZipFile(frozen) as archive:
        metadata = json.loads(archive.read("model/extra/metadata.json"))
        assert metadata["has_comm_artifact"] is False
        assert "model/extra/forward_lower_with_comm.pt2" not in archive.namelist()
    actual = (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
    assert np.max(np.abs(expected[1])) > 1e-8
    fp32 = inference_device == "cuda" or kind == "dpa4c"
    rtol, atol = (1e-4, 1e-5) if fp32 else (1e-6, 1e-8)
    for reference, result in zip(expected, actual, strict=True):
        np.testing.assert_allclose(result, reference, rtol=rtol, atol=atol)
    assert not any(
        record.name == "deepmd.calculator" and record.levelno >= logging.WARNING
        for record in caplog.records
    )
    atoms.calc = DP(frozen, freeze=True)
    reload_tol = 1e-6 if fp32 else 1e-12
    np.testing.assert_allclose(
        atoms.get_forces(), actual[1], rtol=reload_tol, atol=reload_tol
    )
    assert not (tmp_path / "model-ase_single-ase_single.pt2").exists()
