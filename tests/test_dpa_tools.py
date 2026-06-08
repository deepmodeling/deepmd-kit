# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for dpa_tools data conversion pipelines."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_fake_poscar(path: str) -> None:
    """Write a minimal 2×2×1 NiO₂H₂ slab POSCAR (~12 atoms)."""
    content = """Ni O H slab
1.0
   5.0 0.0 0.0
   0.0 5.0 0.0
   0.0 0.0 10.0
Ni O H
4 6 2
direct
0.00 0.00 0.00 Ni
0.50 0.00 0.00 Ni
0.00 0.50 0.00 Ni
0.50 0.50 0.00 Ni
0.25 0.25 0.10 O
0.75 0.25 0.10 O
0.25 0.75 0.10 O
0.75 0.75 0.10 O
0.25 0.25 0.20 O
0.75 0.75 0.20 O
0.40 0.40 0.15 H
0.60 0.60 0.15 H
"""
    Path(path).write_text(content)


def _write_formula_csv(path: str, *, with_header: bool = False) -> list[str]:
    """Write a 3-row formula CSV.  Returns the formula strings for assertions."""
    formulas = [
        "Ni0.75Co0.25O2H1",
        "Ni0.50Co0.50O2H1",
        "Ni1.00O2H1",
    ]
    values = ["1.5", "2.0", "0.8"]
    lines = []
    if with_header:
        lines.append("formula,overpotential")
    for f, v in zip(formulas, values):
        lines.append(f"{f},{v}")
    Path(path).write_text("\n".join(lines))
    return formulas


# ---------------------------------------------------------------------------
# formula_to_npy
# ---------------------------------------------------------------------------


class TestFormulaCsvToNpy:
    def test_basic(self) -> None:
        """3 formulas × 2 sets → 6 valid deepmd/npy systems."""
        with tempfile.TemporaryDirectory() as tmp:
            poscar_path = os.path.join(tmp, "POSCAR")
            csv_path = os.path.join(tmp, "data.csv")
            out_dir = os.path.join(tmp, "output")

            _write_fake_poscar(poscar_path)
            _write_formula_csv(csv_path, with_header=False)

            from dpa_tools.data.formula import formula_to_npy

            systems = formula_to_npy(
                csv_path=csv_path,
                output_dir=out_dir,
                poscar=poscar_path,
                property_name="overpotential",
                sets=2,
                seed=0,
            )

            assert len(systems) == 6, f"Expected 6 systems, got {len(systems)}"

            # Verify each output is a valid deepmd/npy directory.
            for i, sys_dir in enumerate(systems):
                d = Path(sys_dir)
                set000 = d / "set.000"
                assert d.is_dir(), f"sys_{i:04d} not a directory"
                assert (d / "type.raw").is_file(), f"sys_{i:04d}: missing type.raw"
                assert (set000 / "coord.npy").is_file(), f"sys_{i:04d}: missing set.000/coord.npy"
                assert (set000 / "box.npy").is_file(), f"sys_{i:04d}: missing set.000/box.npy"
                label_file = set000 / "overpotential.npy"
                assert label_file.is_file(), f"sys_{i:04d}: missing overpotential.npy"

                # Verify label value is a float.
                label = np.load(str(label_file))
                assert label.shape == (1,)

    def test_with_header(self) -> None:
        """Header row is auto-skipped; still produces 6 systems."""
        with tempfile.TemporaryDirectory() as tmp:
            poscar_path = os.path.join(tmp, "POSCAR")
            csv_path = os.path.join(tmp, "data.csv")
            out_dir = os.path.join(tmp, "output")

            _write_fake_poscar(poscar_path)
            _write_formula_csv(csv_path, with_header=True)

            from dpa_tools.data.formula import formula_to_npy

            systems = formula_to_npy(
                csv_path=csv_path,
                output_dir=out_dir,
                poscar=poscar_path,
                property_name="overpotential",
                sets=2,
                seed=0,
            )

            assert len(systems) == 6, f"Expected 6 systems (header skipped), got {len(systems)}"
            for sys_dir in systems:
                assert (Path(sys_dir) / "set.000" / "overpotential.npy").is_file()


# ---------------------------------------------------------------------------
# parse_formula
# ---------------------------------------------------------------------------


class TestParseFormula:
    def test_basic(self) -> None:
        from dpa_tools.data.formula import parse_formula

        r = parse_formula("Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1")
        assert r == pytest.approx({
            "Ni": 0.65, "Gd": 0.15, "Fe": 0.10, "Co": 0.05, "Yb": 0.05,
            "O": 2.0, "H": 1.0,
        })

    def test_base_element_inference(self) -> None:
        from dpa_tools.data.formula import parse_formula

        # Co=0.25 total < 1.0 → Ni infers as 0.75 remainder.
        r = parse_formula("Co0.25O2H1", base_element="Ni")
        assert "Ni" in r
        assert r["Co"] == pytest.approx(0.25)
        assert r["Ni"] == pytest.approx(0.75)

    def test_normalisation(self) -> None:
        from dpa_tools.data.formula import parse_formula

        r = parse_formula("Ni0.5Co0.5O2H1")
        sub_sum = sum(v for k, v in r.items() if k not in ("O", "H"))
        assert sub_sum == pytest.approx(1.0)

    def test_empty_raises(self) -> None:
        from dpa_tools.data.formula import parse_formula

        with pytest.raises(ValueError, match="Could not parse"):
            parse_formula("")


# ---------------------------------------------------------------------------
# infer_base_element
# ---------------------------------------------------------------------------


class TestInferBaseElement:
    def test_basic(self) -> None:
        from dpa_tools.data.formula import infer_base_element

        assert infer_base_element(["Ni", "Ni", "O", "H"]) == "Ni"
        assert infer_base_element(["Co", "Co", "Ni", "O"]) == "Co"

    def test_only_o_h(self) -> None:
        from dpa_tools.data.formula import infer_base_element

        assert infer_base_element(["O", "H", "O"]) is None
