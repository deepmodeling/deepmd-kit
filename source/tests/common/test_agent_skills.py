# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)

ROOT = Path(__file__).resolve().parents[3]
LAMMPS_SKILL = ROOT / "skills" / "lammps-deepmd" / "SKILL.md"
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
    assert "-n 0 -d details/system.000" in held_out
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
