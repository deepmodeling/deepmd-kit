# SPDX-License-Identifier: LGPL-3.0-or-later
"""ASE checkpoint export and eager-inference behavior."""

import contextlib
import copy
import gc
import json
import logging
import os
import zipfile
from collections.abc import (
    Callable,
    Iterator,
)
from concurrent.futures import (
    ThreadPoolExecutor,
)
from functools import (
    partial,
)
from pathlib import (
    Path,
)
from threading import (
    Barrier,
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

from deepmd import calculator as calculator_module
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

from ..common.sezm_config import (
    _tiny_sezm_model_params,
    _tiny_sezm_native_spin_model_params,
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


@pytest.fixture
def file_freezer(monkeypatch: pytest.MonkeyPatch) -> mock.Mock:
    """Exercise artifact ownership without running threaded compilation."""

    class FileEvaluator:
        def __init__(self, model_file: str) -> None:
            self.model_path = model_file

        def serialize(self) -> str:
            return Path(self.model_path).read_text()

    def load(model_file: str, **kwargs: object) -> mock.Mock:
        return mock.Mock(
            deep_eval=FileEvaluator(model_file),
            get_type_map=mock.Mock(return_value=["O", "H"]),
            get_ntypes=mock.Mock(return_value=2),
        )

    def export(source: str, output: str, *, head: str | None) -> None:
        Path(output).write_text(head or "Default")

    freezer = mock.Mock(side_effect=export)

    def select(
        evaluator: object, head: str | None, **kwargs: object
    ) -> Callable[..., None]:
        return partial(freezer, head=head)

    monkeypatch.setattr(calculator_module, "DeepPot", load)
    monkeypatch.setattr(calculator_module, "_checkpoint_freezer", select)
    return freezer


def test_automatic_freeze_retains_owned_artifact(
    tmp_path: Path, file_freezer: mock.Mock
) -> None:
    checkpoint = tmp_path / "model.pt"
    first = DP(checkpoint, head="first", freeze=True)
    second = DP(checkpoint, head="second", freeze=True)
    evaluator = first.dp.deep_eval
    private = Path(evaluator.model_path)
    assert evaluator.serialize() == "first"
    assert second.dp.deep_eval.serialize() == "second"
    assert (tmp_path / "model-ase_single.pt2").read_text() == "second"
    (tmp_path / "model-ase_single.pt2").write_text("external overwrite")
    assert evaluator.serialize() == "first"
    assert second.dp.deep_eval.serialize() == "second"
    del first
    gc.collect()
    assert private.exists()
    del evaluator
    gc.collect()
    assert not private.parent.exists()


def test_concurrent_automatic_freeze_loads_each_head(
    tmp_path: Path, file_freezer: mock.Mock
) -> None:
    barrier = Barrier(2)

    def export(source: str, output: str, *, head: str) -> None:
        Path(output).write_text(head)
        barrier.wait(timeout=10)

    file_freezer.side_effect = export

    def construct(head: str) -> DP:
        return DP(tmp_path / "model.pt", head=head, freeze=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(construct, ["first", "second"])
    assert first.dp.deep_eval.serialize() == "first"
    assert second.dp.deep_eval.serialize() == "second"
    assert (tmp_path / "model-ase_single.pt2").read_text() in {"first", "second"}


@pytest.mark.parametrize("stage", ["export", "load", "publish"])
def test_failed_automatic_freeze_preserves_published_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_freezer: mock.Mock,
    stage: str,
) -> None:
    output = tmp_path / "model-ase_single.pt2"
    output.write_text("original")

    def fail_export(source: str, output: str, **kwargs: object) -> None:
        Path(output).write_text("incomplete")
        raise RuntimeError("export failed")

    load = calculator_module.DeepPot

    def fail_load(model_file: str, **kwargs: object) -> object:
        if model_file.endswith(".pt2"):
            raise RuntimeError("load failed")
        return load(model_file, **kwargs)

    if stage == "export":
        file_freezer.side_effect = fail_export
    elif stage == "load":
        monkeypatch.setattr(calculator_module, "DeepPot", fail_load)
    else:
        monkeypatch.setattr(
            Path, "replace", mock.Mock(side_effect=RuntimeError("publish failed"))
        )
    with pytest.raises(RuntimeError, match=f"{stage} failed"):
        DP(tmp_path / "model.pt", freeze=True)
    assert output.read_text() == "original"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("skin", [None, 0.75])
def test_graph_neighbor_options_reach_evaluator(
    tmp_path: Path, skin: float | None
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt", "dpa4")
    options = {"neighbor_skin": skin} if skin is not None else {}
    with mock.patch("deepmd.calculator.DeepPot", wraps=DeepPot) as deep_pot:
        DP(checkpoint, neighbor_graph_method="vesin", **options)
    deep_pot.assert_called_once_with(
        str(checkpoint.resolve()),
        neighbor_list=None,
        head=None,
        nlist_backend="auto",
        neighbor_graph_method="vesin",
        neighbor_skin=1.0 if skin is None else skin,
    )


def test_native_spin_edge_freeze_preserves_neighbor_options(tmp_path: Path) -> None:
    config = _tiny_sezm_native_spin_model_params()
    # Edge inputs retain all cutoff neighbors independently of dense capacity.
    config["descriptor"]["sel"] = [1, 0]
    checkpoint = _write_checkpoint(tmp_path / "native.pt", "pt", config=config)
    original = DeepPot(
        str(checkpoint), neighbor_graph_method="vesin", neighbor_skin=1.0
    )
    frozen = tmp_path / "native.pt2"
    freeze_pt2.freeze_sezm_to_pt2(str(checkpoint), str(frozen), with_comm=False)
    coords = np.array([[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.8, 0.0]]])
    cells = np.eye(3).reshape(1, 9) * 8.0
    types = np.array([0, 1, 0])
    spins = np.array([[[0.2, 0.3, 0.9], [0.0, 0.0, 0.0], [-0.1, 0.6, 0.7]]])
    for method in ("vesin", "auto"):
        exported = DeepPot(str(frozen), neighbor_graph_method=method, neighbor_skin=1.0)
        for displacement in (0.0, 0.1, 0.7):
            moved = coords.copy()
            moved[0, 1, 0] += displacement
            expected = original.eval(moved, cells, types, spin=spins, atomic=True)
            actual = exported.eval(moved, cells, types, spin=spins, atomic=True)
            for value, reference in zip(actual, expected, strict=True):
                np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-8)

    native = DeepPot(str(frozen), nlist_backend="native", auto_batch_size=False)
    batch_coords = np.repeat(coords, 2, axis=0)
    batch_cells = np.repeat(cells, 2, axis=0)
    batch_types = np.array([[0, 1, 0], [1, 0, 0]])
    batch_spins = np.repeat(spins, 2, axis=0)
    actual = native.eval(
        batch_coords,
        batch_cells,
        batch_types,
        spin=batch_spins,
        atomic=True,
        mixed_type=True,
    )
    frames = [
        original.eval(
            batch_coords[frame : frame + 1],
            batch_cells[frame : frame + 1],
            batch_types[frame],
            spin=batch_spins[frame : frame + 1],
            atomic=True,
        )
        for frame in range(2)
    ]
    for value, first, second in zip(actual, *frames, strict=True):
        np.testing.assert_allclose(
            value, np.concatenate((first, second), axis=0), rtol=1e-6, atol=1e-8
        )


@pytest.mark.parametrize("skin", [-1.0, float("inf"), float("nan")])
def test_invalid_neighbor_skin_rejected(skin: float) -> None:
    with pytest.raises(ValueError, match="neighbor_skin"):
        DP("unused.pt", neighbor_skin=skin)


@pytest.mark.parametrize(
    "kind", ["pt", "pt-dpa4", "dpa4", "dpa4c", "dpa1", "dpa2", "pt-dense", "dense"]
)
@pytest.mark.parametrize("inference_device", ["cpu", "cuda"], indirect=True)
def test_cached_neighbors_and_types_match_fresh_evaluation(
    tmp_path: Path, kind: str, inference_device: str
) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt", kind)
    options = (
        {"nlist_backend": "vesin"}
        if kind in ("pt-dense", "dense")
        else {"neighbor_graph_method": "vesin"}
    )
    cached = DP(checkpoint, **options)
    fresh = DP(checkpoint, neighbor_skin=0.0, **options)
    positions = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.8, 0.0]])
    for step in range(5):
        points = positions.copy()
        points[1, 0] += 0.1 * step
        symbols = "OH2" if step < 2 else "HOH"
        cell = [8.0, 8.0, 8.0] if step < 3 else [7.5, 8.0, 8.0]
        if step == 4:
            cached.type_dict = fresh.type_dict = {"O": 1, "H": 0}
        atoms = Atoms(symbols, positions=points, cell=cell, pbc=True)
        results = []
        for calculator in (cached, fresh):
            atoms.calc = calculator
            results.append(
                (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
            )
        fp32 = kind == "dpa4c" or inference_device == "cuda"
        rtol, atol = (1e-4, 1e-5) if fp32 else (1e-7, 1e-8)
        for actual, expected in zip(*results, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


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
    export_policy = serialization._fused_operators_for_export

    @contextlib.contextmanager
    def cuda_export_policy() -> Iterator[None]:
        # Only policy selection sees CUDA; lazy imports retain the real query.
        with contextlib.ExitStack() as stack:
            with mock.patch(
                "deepmd.pt_expt.kernels.utils.backend_device_type", return_value="cuda"
            ):
                stack.enter_context(export_policy())
            yield

    monkeypatch.setattr(
        serialization, "_fused_operators_for_export", cuda_export_policy
    )
    if override is not None:
        for name, value in zip(_LEVELS[:2], override, strict=True):
            monkeypatch.setenv(name, value)
    captured = {}

    def capture(*args: object, **kwargs: object) -> None:
        captured["levels"] = tuple(os.environ.get(name) for name in _LEVELS[:2])
        captured["with_comm"] = kwargs["with_comm"]
        Path(args[1] if kind == "pt" else args[0]).write_bytes(b"export")

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
        DP(
            checkpoint,
            freeze=True,
            head=head,
            neighbor_graph_method="vesin",
            neighbor_skin=0.75,
        )

    expected = override or ("1", "2")
    if kind == "dpa4c":
        expected = ("2", None if override is None else override[1])
    assert captured == {"levels": expected, "with_comm": False}
    assert tuple(os.environ.get(name) for name in _LEVELS[:2]) == (
        override or (None, None)
    )
    assert deep_pot.call_count == 2
    private = Path(deep_pot.call_args.args[0])
    assert private.name == "model-ase_single.pt2"
    assert private.parent != tmp_path
    assert (tmp_path / private.name).read_bytes() == b"export"
    deep_pot.assert_called_with(
        str(private),
        neighbor_list=None,
        nlist_backend="auto",
        neighbor_graph_method="vesin",
        neighbor_skin=0.75,
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

    def write_artifact(source: str, output: str, **kwargs: object) -> None:
        Path(output).write_bytes(b"export")

    export = mock.Mock(side_effect=write_artifact)
    monkeypatch.setattr(freeze_pt2, "freeze_sezm_to_pt2", export)
    with mock.patch("deepmd.calculator.DeepPot", return_value=loaded):
        DP(checkpoint, head="Default", freeze=True)
    export.assert_called_once_with(
        str(checkpoint),
        mock.ANY,
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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(kind)
    options = {}
    if kind == "pt":
        config["descriptor"]["sel"] = [1, 0]
        config["pair_exclude_types"] = [[0, 1]]
        options["nlist_backend"] = "native"
    checkpoint = _write_checkpoint(tmp_path / "model.pt", kind, config=config)
    original = checkpoint.read_bytes()
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.8, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.calc = DP(checkpoint, **options)
    expected = (atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress())
    if kind == "pt":
        np.testing.assert_array_equal(expected[1][0], 0.0)
        from deepmd.pt_expt.utils import (
            auto_neighbor_list,
        )

        monkeypatch.setattr(auto_neighbor_list, "is_nv_available", lambda: False)
        monkeypatch.setattr(
            auto_neighbor_list, "is_vesin_torch_available", lambda: False
        )
    caplog.clear()
    with mock.patch(
        "torch._inductor.aoti_compile_and_package",
        wraps=torch._inductor.aoti_compile_and_package,
    ) as compile_package:
        atoms.calc = DP(checkpoint, freeze=True, **options)
    assert compile_package.call_count == 1
    if kind == "dpa4c":
        exported = compile_package.call_args.args[0]
        targets = {str(node.target) for node in exported.graph_module.graph.nodes}
        assert any("dpa4c_" in target and "compress" in target for target in targets)
    frozen = tmp_path / "model-ase_single.pt2"
    assert checkpoint.read_bytes() == original
    private = Path(atoms.calc.dp.deep_eval.model_path)
    assert private != frozen
    assert private.read_bytes() == frozen.read_bytes()
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
