# SPDX-License-Identifier: LGPL-3.0-or-later
"""Single-pass compressed export through the CLI and ASE calculator."""

import json
import logging
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
from deepmd.main import (
    parse_args,
)
from deepmd.pt_expt.entrypoints import compress as compression
from deepmd.pt_expt.entrypoints import main as entrypoints
from deepmd.pt_expt.utils import (
    env,
    serialization,
)

from .test_calculator import (
    _config,
    _write_checkpoint,
)


@pytest.fixture(autouse=True)
def cpu_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(env, "DEVICE", torch.device("cpu"))
    for name in ("DP_CUDA_INFER", "DP_TRITON_INFER", "DP_COMPILE_INFER"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("compress", [False, True])
def test_freeze_cli_compression_option(tmp_path: Path, compress: bool) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    args = ["--pt-expt", "freeze", "-c", str(checkpoint)]
    if compress:
        args.append("--compress")
    flags = parse_args(args)
    with mock.patch.object(entrypoints, "freeze") as freeze:
        entrypoints.main(flags)
    assert freeze.call_args.kwargs["compress"] is compress


@pytest.mark.parametrize("compress", [False, True])
def test_freeze_compression_changes_snapshot_before_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compress: bool
) -> None:
    config = _config("dpa4c")
    config["fitting_net"]["vacuum_ref"] = True
    config["preset_out_bias"] = {"energy": {"O": 0.0, "H": 0.0}}
    checkpoint = _write_checkpoint(tmp_path / "model.pt", "dpa4c", config=config)
    original = checkpoint.read_bytes()
    captured = {}

    def capture(model_file: str, data: dict, **kwargs: object) -> None:
        captured.update(data=data, options=kwargs)

    monkeypatch.setattr(serialization, "deserialize_to_file", capture)
    monkeypatch.setattr(compression, "deserialize_to_file", capture)
    entrypoints.freeze(
        str(checkpoint),
        str(tmp_path / "model.pt2"),
        compress=compress,
        with_comm=False,
    )
    model = captured["data"]["model"]
    assert ("compress" in model["descriptor"]) is compress
    assert model["fitting"]["vacuum_ref"] is (not compress)
    assert captured["options"]["with_comm"] is False
    assert captured["options"]["lower_kind"] == ("auto" if compress else "graph")
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("case", ["precision", "exclusions", "charge_state"])
def test_ase_falls_back_but_explicit_compression_reports_ineligibility(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, case: str
) -> None:
    config = _config("dpa4c")
    if case == "precision":
        config["descriptor"]["precision"] = "float64"
        config["fitting_net"]["precision"] = "float64"
        error = "requires descriptor precision"
    elif case == "exclusions":
        config["descriptor"]["exclude_types"] = [[0, 1]]
        error = "non-empty `exclude_types`"
    else:
        config["descriptor"]["add_chg_spin_ebd"] = True
        config["descriptor"]["default_chg_spin"] = None
        error = "requires `default_chg_spin`"
    checkpoint = _write_checkpoint(tmp_path / "model.pt", "dpa4c", config=config)
    loaded = DeepPot(str(checkpoint))
    with (
        mock.patch("deepmd.calculator.DeepPot", return_value=loaded),
        mock.patch.object(entrypoints, "freeze") as freeze,
        caplog.at_level(logging.WARNING, logger="deepmd.calculator"),
    ):
        DP(checkpoint, freeze=True)
    assert freeze.call_args.kwargs["compress"] is False
    assert "Freezing without compression" in caplog.text
    assert error in caplog.text
    output = tmp_path / "explicit.pt2"
    with pytest.raises(ValueError, match=error):
        entrypoints.freeze(str(checkpoint), str(output), compress=True)
    assert not output.exists()


def test_compressed_freeze_compiles_one_cpu_artifact(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path / "model.pt", "dpa4c")
    output = tmp_path / "model.pt2"

    def compile_package(
        exported: torch.export.ExportedProgram, package_path: str, **kwargs: object
    ) -> None:
        with zipfile.ZipFile(package_path, "w") as archive:
            archive.writestr("model/data.pkl", b"")

    with mock.patch(
        "torch._inductor.aoti_compile_and_package", side_effect=compile_package
    ) as compile_mock:
        entrypoints.freeze(str(checkpoint), str(output), compress=True, with_comm=False)
    assert compile_mock.call_count == 1
    with zipfile.ZipFile(output) as archive:
        metadata = json.loads(archive.read("model/extra/metadata.json"))
        assert metadata["has_comm_artifact"] is False
        assert metadata["lower_input_kind"] == "graph"
        assert "model/extra/forward_lower_with_comm.pt2" not in archive.namelist()
    snapshot = serialization.serialize_from_file(str(output))
    assert "compress" in snapshot["model"]["descriptor"]


def test_ase_float64_fallback_exports_and_evaluates(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = _config("dpa4c")
    config["descriptor"]["precision"] = "float64"
    config["fitting_net"]["precision"] = "float64"
    checkpoint = _write_checkpoint(tmp_path / "model.pt", "dpa4c", config=config)
    atoms = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.3, 0.8, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
        calculator=DP(checkpoint),
    )
    expected = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="deepmd.calculator"):
        atoms.calc = DP(checkpoint, freeze=True)
    actual = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
    assert "Freezing without compression" in caplog.text
    snapshot = serialization.serialize_from_file(str(tmp_path / "model-ase_single.pt2"))
    assert "compress" not in snapshot["model"]["descriptor"]
    for reference, result in zip(expected, actual, strict=True):
        np.testing.assert_allclose(result, reference, rtol=1e-6, atol=1e-8)
