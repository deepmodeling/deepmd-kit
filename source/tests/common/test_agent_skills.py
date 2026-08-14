# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)

ROOT = Path(__file__).resolve().parents[3]
LAMMPS_SKILL = ROOT / "skills" / "lammps-deepmd" / "SKILL.md"


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
