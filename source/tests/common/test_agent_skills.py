# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import re
import shutil
import subprocess
from pathlib import (
    Path,
)

import pytest

ROOT = Path(__file__).resolve().parents[3]
LAMMPS_SKILL = ROOT / "skills" / "lammps-deepmd" / "SKILL.md"
LAMMPS_DEPLOYMENT = (
    ROOT / "skills" / "lammps-deepmd" / "references" / "model-deployment.md"
)
LAMMPS_WORKFLOW = (
    ROOT / "skills" / "lammps-deepmd" / "references" / "commands-and-workflow.md"
)
LAMMPS_ASSET = ROOT / "skills" / "lammps-deepmd" / "assets" / "input.nvt.lammps"
DPA4_TRAIN_REFERENCE = ROOT / "skills" / "deepmd-train" / "models" / "dpa4.md"
DPA4_FREEZE_POLICY = (
    ROOT / "skills" / "deepmd-python-inference" / "references" / "dpa4-freeze-policy.md"
)
DPA4_FINETUNE_SKILL = ROOT / "skills" / "deepmd-finetune-dpa4" / "SKILL.md"
INFERENCE_SKILL = ROOT / "skills" / "deepmd-python-inference" / "SKILL.md"
HELD_OUT_REFERENCE = (
    ROOT
    / "skills"
    / "deepmd-python-inference"
    / "references"
    / "held-out-evaluation.md"
)
DP_TEST_ENTRYPOINT = ROOT / "deepmd" / "entrypoints" / "test.py"
ENERGY_TESTER = ROOT / "deepmd" / "infer" / "model_test" / "ener.py"
FINETUNE_SOURCE = ROOT / "deepmd" / "utils" / "finetune.py"


def test_lammps_skill_uses_capability_gated_runtime() -> None:
    text = LAMMPS_SKILL.read_text(encoding="utf-8")

    assert "3.2.0b0" not in text
    assert "uvx --from" not in text
    assert "record the resolved Git commit SHA" in text
    assert "do not claim support from an" in text
    assert "unreleased version number" in text
    assert "do not install or upgrade packages silently" in text


def test_lammps_required_inputs_remain_nested() -> None:
    text = LAMMPS_SKILL.read_text(encoding="utf-8")
    section = text.split("1. Confirm the minimum simulation inputs:", 1)[1].split(
        "1. Write the LAMMPS input script", 1
    )[0]

    for required in (
        "structure/data file",
        "DeePMD model artifact",
        "atom type to element mapping",
        "target ensemble",
        "temperature, pressure",
    ):
        assert any(
            line.startswith("   - ") and required in line
            for line in section.splitlines()
        )


def test_complete_held_out_evaluation_is_routed_and_evidence_complete() -> None:
    finetune = DPA4_FINETUNE_SKILL.read_text(encoding="utf-8")
    inference = INFERENCE_SKILL.read_text(encoding="utf-8")
    held_out = HELD_OUT_REFERENCE.read_text(encoding="utf-8")

    assert "references/held-out-evaluation.md" in inference
    assert "held-out-evaluation.md" in finetune
    assert "dp --pt test -m selected.pt" in held_out
    assert '-n 0 -d "$detail_prefix"' in held_out
    assert "one command per held-out system" in held_out
    assert "population standard deviation (`ddof=0`)" in held_out
    assert "do not average per-system RMSE values" in held_out
    assert "Training logs, a successful freeze" in held_out


def test_complete_held_out_evaluation_checks_dataset_shapes() -> None:
    held_out = HELD_OUT_REFERENCE.read_text(encoding="utf-8")

    assert "number of whitespace-separated entries in `type.raw`" in held_out
    assert "coordinate and force widths of `3 * natoms`" in held_out
    assert "force rows equal `frames * natoms`" in held_out
    assert "divided by `natoms` exactly once" in held_out
    assert "zero-based indices" in held_out


def test_held_out_contract_matches_dp_test_source() -> None:
    entrypoint = DP_TEST_ENTRYPOINT.read_text(encoding="utf-8")
    energy_tester = ENERGY_TESTER.read_text(encoding="utf-8")

    assert "if numb_test == 0:" in entrypoint
    assert 'detail_path.with_suffix(".e.out")' in energy_tester
    assert 'detail_path.with_suffix(".e_peratom.out")' in energy_tester
    assert 'detail_path.with_suffix(".f.out")' in energy_tester


def test_dpa4_freeze_and_lammps_stay_on_target_node() -> None:
    deployment = LAMMPS_DEPLOYMENT.read_text(encoding="utf-8")

    assert "same target physical compute node" in deployment
    assert "inspect the native checkpoint -> freeze `.pt2` -> `run 0`" in deployment
    assert "move the archive to B" in deployment
    assert "artifact portability is proven" in deployment


def test_lammps_mapping_data_and_dump_contract() -> None:
    skill = LAMMPS_SKILL.read_text(encoding="utf-8")
    deployment = LAMMPS_DEPLOYMENT.read_text(encoding="utf-8")
    workflow = LAMMPS_WORKFLOW.read_text(encoding="utf-8")
    example = skill.split("## Example: annotated NVT input", 1)[1].split(
        "### What every command means", 1
    )[0]

    assert example.index("atom_modify     map yes") < example.index(
        "read_data       data.system"
    )
    assert "pair_coeff      * * Si O" in example
    assert "dump_modify     1 element Si O sort id" in example
    assert "zero-based `type.raw` index" in deployment
    assert "Do not sort atoms by element" in deployment
    assert "`xy = b_x`" in deployment
    assert "`xz = c_x`" in deployment
    assert "`yz = c_y`" in deployment
    assert "first line of a LAMMPS data file is a title" in deployment
    assert "first line of a LAMMPS data file is a skipped title" in workflow


def test_lammps_canary_requires_physical_stability() -> None:
    deployment = LAMMPS_DEPLOYMENT.read_text(encoding="utf-8")
    workflow = LAMMPS_WORKFLOW.read_text(encoding="utf-8")

    assert "short NVE when physically appropriate" in deployment
    assert "Exit code zero alone is not a passed canary" in deployment
    assert "early temperature, pressure, and controlled variables" in deployment
    assert "do not barostat that direction" in workflow
    assert "shell environment variables and LAMMPS variables distinct" in workflow


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
    assert (tmp_path / "details" / "selected-SHA256" / "system.000.e.out").is_file()


def test_dpa4_minimal_model_configuration_normalizes() -> None:
    pytest.importorskip("deepmd.lib", reason="requires a built DeePMD checkout")
    from deepmd.utils.argcheck import (
        normalize,
    )
    from deepmd.utils.compat import (
        update_deepmd_input,
    )

    text = DPA4_TRAIN_REFERENCE.read_text(encoding="utf-8")
    section = text.split("## Minimal model configuration", 1)[1]
    fenced_json = re.search(r"```json\n(.*?)\n```", section, flags=re.DOTALL)
    assert fenced_json is not None
    model_fragment = json.loads(fenced_json.group(1))
    config = {
        **model_fragment,
        "training": {
            "training_data": {"systems": ["dummy"]},
            "numb_steps": 1,
        },
        "loss": {"type": "ener"},
        "learning_rate": {"type": "exp", "start_lr": 1e-3},
    }

    normalized = normalize(update_deepmd_input(config, warning=False))

    assert normalized["model"]["fitting_net"]["type"] == "dpa4_ener"


def test_lammps_asset_matches_mapping_contract() -> None:
    asset = LAMMPS_ASSET.read_text(encoding="utf-8")

    assert asset.index("atom_modify     map yes") < asset.index(
        "read_data       data.system"
    )
    assert "pair_coeff      * * Si O" in asset
    assert "id type element x y z" in asset
    assert "dump_modify     1 element Si O sort id" in asset


def test_dpa4_freeze_policy_is_explicit_and_routed() -> None:
    policy = DPA4_FREEZE_POLICY.read_text(encoding="utf-8")
    train = DPA4_TRAIN_REFERENCE.read_text(encoding="utf-8")
    finetune = DPA4_FINETUNE_SKILL.read_text(encoding="utf-8")
    deployment = LAMMPS_DEPLOYMENT.read_text(encoding="utf-8")

    for variable in ("DP_TRITON_INFER", "DP_TF32_INFER", "DP_AMP_INFER"):
        assert f"export {variable}=" in policy
    assert "Levels 1 and 2 keep" in policy
    assert "Level 3, TF32, and AMP" in policy
    assert "dpa4-freeze-policy.md" in train
    assert "dpa4-freeze-policy.md" in finetune
    assert "dpa4-freeze-policy.md" in deployment


def test_held_out_multitask_and_optional_type_map_contracts() -> None:
    held_out = HELD_OUT_REFERENCE.read_text(encoding="utf-8")

    assert "When `type_map.raw` is present" in held_out
    assert "is absent, require provenance" in held_out
    assert "--head SELECTED_BRANCH" in held_out
    assert "already single-head; do not pass `--head`" in held_out


def test_use_pretrain_script_guidance_matches_source_scope() -> None:
    guidance = DPA4_FINETUNE_SKILL.read_text(encoding="utf-8")
    source = FINETUNE_SOURCE.read_text(encoding="utf-8")
    function = source.split("def _apply_pretrained_model_params", 1)[1].split(
        "\ndef ", 1
    )[0]

    assert "does not restore the complete model" in guidance
    assert "model.descriptor" in guidance
    assert "model.fitting_net" in guidance
    assert 'pretrained_config["descriptor"]' in function
    assert 'pretrained_config["fitting_net"]' in function
