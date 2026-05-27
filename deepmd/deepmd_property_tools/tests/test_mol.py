# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)

import numpy as np
from deepmd_property_tools.data.mol import (
    build_used_type_map,
    has_overlapping_atoms,
    parse_property_value,
    read_mol_coords,
    records_from_direct_data,
)


def test_parse_property_value_accepts_text_with_units() -> None:
    assert parse_property_value("gap = -1.25 eV") == -1.25


def test_overlap_detection() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    assert has_overlapping_atoms(coords, 1e-6)


def test_type_map_uses_periodic_table_order() -> None:
    assert build_used_type_map({"O", "C", "H"}) == ["H", "C", "O"]


def test_records_from_direct_data() -> None:
    records, rows = records_from_direct_data(
        {
            "atoms": [["O", "H", "H"]],
            "coordinates": [np.zeros((3, 3), dtype=np.float32)],
            "target": [1.5],
        }
    )

    assert records[0][0] == ["O", "H", "H"]
    assert records[0][2] == 1.5
    assert rows == [{"sample_id": 0, "target": 1.5}]


def test_read_mol_coords(tmp_path: Path) -> None:
    mol_path = tmp_path / "id0.mol"
    mol_path.write_text(
        "\n".join(
            [
                "methane",
                "",
                "",
                "  1  0  0  0  0  0            999 V2000",
                "    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "M  END",
            ]
        ),
        encoding="utf-8",
    )

    symbols, coords = read_mol_coords(mol_path)

    assert symbols == ["C"]
    assert coords.shape == (1, 3)
