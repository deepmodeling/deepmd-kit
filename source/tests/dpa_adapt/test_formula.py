# SPDX-License-Identifier: LGPL-3.0-or-later
"""End-to-end tests for the formula -> deepmd/npy conversion pipeline.

Exercises ``dpa_adapt.data.formula.formula_to_npy()`` for real. (``test_convert``
covers ``convert()`` routing with ``formula_to_npy`` mocked, and unit-tests
``parse_formula()`` / ``infer_base_element()``.)
"""

from pathlib import (
    Path,
)

import numpy as np


def _write_fake_poscar(path: str) -> None:
    r"""Write a minimal 2x2x1 NiO2H2 slab POSCAR (~12 atoms)."""
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
    for f, v in zip(formulas, values, strict=True):
        lines.append(f"{f},{v}")
    Path(path).write_text("\n".join(lines))
    return formulas


class TestFormulaCsvToNpy:
    def test_basic(self, tmp_path) -> None:
        """3 formulas x 2 sets -> 6 valid deepmd/npy systems."""
        poscar_path = str(tmp_path / "POSCAR")
        csv_path = str(tmp_path / "data.csv")
        out_dir = str(tmp_path / "output")

        _write_fake_poscar(poscar_path)
        _write_formula_csv(csv_path, with_header=False)

        from dpa_adapt.data.formula import (
            formula_to_npy,
        )

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
            assert (set000 / "coord.npy").is_file(), (
                f"sys_{i:04d}: missing set.000/coord.npy"
            )
            assert (set000 / "box.npy").is_file(), (
                f"sys_{i:04d}: missing set.000/box.npy"
            )
            label_file = set000 / "overpotential.npy"
            assert label_file.is_file(), f"sys_{i:04d}: missing overpotential.npy"

            # Verify label value is a float.
            label = np.load(str(label_file))
            assert label.shape == (1,)

    def test_with_header(self, tmp_path) -> None:
        """Header row is auto-skipped; still produces 6 systems."""
        poscar_path = str(tmp_path / "POSCAR")
        csv_path = str(tmp_path / "data.csv")
        out_dir = str(tmp_path / "output")

        _write_fake_poscar(poscar_path)
        _write_formula_csv(csv_path, with_header=True)

        from dpa_adapt.data.formula import (
            formula_to_npy,
        )

        systems = formula_to_npy(
            csv_path=csv_path,
            output_dir=out_dir,
            poscar=poscar_path,
            property_name="overpotential",
            sets=2,
            seed=0,
        )

        assert len(systems) == 6, (
            f"Expected 6 systems (header skipped), got {len(systems)}"
        )
        for sys_dir in systems:
            assert (Path(sys_dir) / "set.000" / "overpotential.npy").is_file()
