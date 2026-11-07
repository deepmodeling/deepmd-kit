# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import subprocess
from pathlib import (
    Path,
)

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
LAMMPS_SKILL = ROOT / "skills" / "lammps-deepmd" / "SKILL.md"
LAMMPS_ASSET = ROOT / "skills" / "lammps-deepmd" / "assets" / "input.nvt.lammps"
HELD_OUT_REFERENCE = (
    ROOT
    / "skills"
    / "deepmd-python-inference"
    / "references"
    / "held-out-evaluation.md"
)


def test_lammps_mapping_data_and_dump_contract() -> None:
    skill = LAMMPS_SKILL.read_text(encoding="utf-8")
    example = skill.split("## Example: annotated NVT input", 1)[1].split(
        "### What every command means", 1
    )[0]

    assert example.index("atom_modify     map yes") < example.index(
        "read_data       data.system"
    )
    assert "pair_coeff      * * Si O" in example
    assert "dump_modify     1 element Si O sort id" in example


def test_held_out_command_runs_from_clean_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    held_out = HELD_OUT_REFERENCE.read_text(encoding="utf-8")
    command = held_out.split("```bash", 1)[1].split("```", 1)[0].strip()
    bash = shutil.which("bash")
    assert bash is not None

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_dp = fake_bin / "dp"
    fake_dp.write_text(
        """#!/bin/sh
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-d" ]; then
    shift
    detail_prefix=$1
  fi
  shift
done
: "${detail_prefix:?missing detail prefix}"
touch "${detail_prefix}.e.out" "${detail_prefix}.e_peratom.out" \
  "${detail_prefix}.f.out"
""",
        encoding="utf-8",
    )
    fake_dp.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"

    first = subprocess.run(
        [bash, "-eu", "-c", command], cwd=tmp_path, env=env, check=False
    )
    second = subprocess.run(
        [bash, "-eu", "-c", command], cwd=tmp_path, env=env, check=False
    )

    assert first.returncode == 0
    assert second.returncode != 0
    assert (tmp_path / "details" / "selected-SHA256" / "system_000.e.out").is_file()


def test_real_detail_writer_keeps_two_system_outputs(tmp_path: Path) -> None:
    pytest.importorskip("deepmd.lib", reason="requires a built DeePMD checkout")
    from deepmd.infer.model_test.ener import (
        _write_energy_test_details,
    )

    for index, energy in enumerate((10.0, 11.0)):
        detail_path = tmp_path / f"system_{index:03d}"
        _write_energy_test_details(
            detail_path=detail_path,
            system=detail_path.name,
            natoms=1,
            append_detail=False,
            reference_energy=np.array([energy]),
            prediction_energy=np.array([energy + 0.5]),
            reference_force=np.zeros((1, 3)),
            prediction_force=np.zeros((1, 3)),
            reference_virial=None,
            prediction_virial=None,
            out_put_spin=False,
        )

    outputs = sorted(tmp_path.glob("system_*.e.out"))
    assert [path.name for path in outputs] == [
        "system_000.e.out",
        "system_001.e.out",
    ]
    assert np.loadtxt(outputs[0], ndmin=2)[0, 0] == pytest.approx(10.0)
    assert np.loadtxt(outputs[1], ndmin=2)[0, 0] == pytest.approx(11.0)


def test_dpa4_and_dpa4c_preset_examples_expand_and_validate() -> None:
    pytest.importorskip("deepmd.lib", reason="requires a built DeePMD checkout")
    from deepmd.utils.argcheck import (
        normalize,
    )
    from deepmd.utils.compat import (
        update_deepmd_input,
    )
    from deepmd.utils.model_preset import (
        expand_model_preset,
    )

    examples = (
        (
            ROOT / "examples" / "water" / "dpa4" / "input_preset.json",
            "dpa4",
            "dpa4_ener",
        ),
        (
            ROOT / "examples" / "water" / "dpa4c" / "input.json",
            "standard",
            "ener",
        ),
    )
    for path, expected_model_type, expected_fitting_type in examples:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        config["model"] = expand_model_preset(config["model"])
        normalized = normalize(update_deepmd_input(config, warning=False))

        assert "preset" not in normalized["model"]
        assert normalized["model"]["type"] == expected_model_type
        assert normalized["model"]["fitting_net"]["type"] == expected_fitting_type


def test_dpa4c_reference_configuration_initializes_trainer(
    tmp_path: Path,
) -> None:
    pytest.importorskip("deepmd.lib", reason="requires a built DeePMD checkout")
    from deepmd.pt_expt.entrypoints.main import (
        get_trainer,
    )
    from deepmd.utils.argcheck import (
        normalize,
    )
    from deepmd.utils.compat import (
        update_deepmd_input,
    )

    example = ROOT / "examples" / "water" / "dpa4c" / "input.json"
    with example.open(encoding="utf-8") as handle:
        config = json.load(handle)
    data_root = ROOT / "examples" / "water" / "data"
    config["training"]["training_data"]["systems"] = [str(data_root / "data_0")]
    config["training"].pop("validation_data", None)
    config["training"]["numb_steps"] = 1
    config["training"]["enable_compile"] = False
    config["training"]["stat_file"] = str(tmp_path / "dpa4c.hdf5")

    config = normalize(update_deepmd_input(config, warning=False))
    trainer = get_trainer(config)

    assert "preset" not in config["model"]
    assert config["model"]["descriptor"]["type"] == "dpa4c"
    assert trainer.lr_schedule is not None


def test_lammps_asset_matches_mapping_contract() -> None:
    asset = LAMMPS_ASSET.read_text(encoding="utf-8")

    assert asset.index("atom_modify     map yes") < asset.index(
        "read_data       data.system"
    )
    assert "pair_coeff      * * Si O" in asset
    assert "id type element x y z" in asset
    assert "dump_modify     1 element Si O sort id" in asset
