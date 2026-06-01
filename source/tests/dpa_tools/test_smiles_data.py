# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

from pathlib import (
    Path,
)
from unittest import (
    mock,
)

import numpy as np
from deepmd.dpa_tools.data import smiles as mol_module
from deepmd.dpa_tools.data.smiles import (
    _build_type_map_from_elements,
    _has_overlapping_atoms,
    _parse_property_value,
    _records_from_csv_mol,
    _records_from_csv_smiles,
    predict_records_from_data,
    read_mol_coords,
    records_from_direct_data,
)


def test__parse_property_value_accepts_text_with_units() -> None:
    assert _parse_property_value("gap = -1.25 eV") == -1.25


def test_overlap_detection() -> None:
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)

    assert _has_overlapping_atoms(coords, 1e-6)


def test_type_map_uses_periodic_table_order() -> None:
    assert _build_type_map_from_elements({"O", "C", "H"}) == ["H", "C", "O"]


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


def test_records_from_csv_smiles_generates_coordinates(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("SMILES,Property\nO,1.5\n", encoding="utf-8")

    with mock.patch.object(
        mol_module,
        "smiles_to_3d_coords",
        return_value=(
            ["O", "H", "H"],
            np.array(
                [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.2, 0.9, 0.0]],
                dtype=np.float32,
            ),
        ),
    ) as smiles_mock:
        records, failed_rows, skipped_zero, skipped_overlap, rows = (
            _records_from_csv_smiles(
                dataset=dataset,
                property_col="Property",
            )
        )

    smiles_mock.assert_called_once_with("O", random_seed=42)
    assert records[0][0] == ["O", "H", "H"]
    assert records[0][2] == 1.5
    assert failed_rows == []
    assert skipped_zero == 0
    assert skipped_overlap == 0
    assert rows == [{"SMILES": "O", "Property": "1.5"}]


def test_records_from_csv_smiles_collects_failed_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("SMILES,Property\nbad,1.5\n", encoding="utf-8")

    with mock.patch.object(
        mol_module,
        "smiles_to_3d_coords",
        side_effect=ValueError("bad smiles"),
    ):
        records, failed_rows, skipped_zero, skipped_overlap, rows = (
            _records_from_csv_smiles(
                dataset=dataset,
                property_col="Property",
            )
        )

    assert records == []
    assert failed_rows == [(0, "bad", "bad smiles")]
    assert skipped_zero == 0
    assert skipped_overlap == 0
    assert rows == []


def test_csv_mol_path_does_not_use_smiles_generation(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("SMILES,Property\nbad,1.5\n", encoding="utf-8")
    mol_dir = tmp_path / "mol"
    mol_dir.mkdir()
    mol_path = mol_dir / "id0.mol"
    mol_path.write_text(
        "\n".join(
            [
                "methane",
                "",
                "",
                "  1  0  0  0  0  0            999 V2000",
                "    0.1000    0.2000    0.3000 C   0  0  0  0  0  0  0  0  0  0  0  0",
                "M  END",
            ]
        ),
        encoding="utf-8",
    )

    with mock.patch.object(
        mol_module,
        "smiles_to_3d_coords",
        side_effect=AssertionError("SMILES generation should not be used"),
    ):
        records, failed_rows, skipped_zero, skipped_overlap, rows = (
            _records_from_csv_mol(
                dataset=dataset,
                mol_dir=mol_dir,
                property_col="Property",
            )
        )
        atoms, coords, pred_rows = predict_records_from_data(
            {"dataset": dataset, "mol_dir": mol_dir},
            property_col=None,
        )

    assert records[0][0] == ["C"]
    assert failed_rows == []
    assert skipped_zero == 0
    assert skipped_overlap == 0
    assert rows == [{"SMILES": "bad", "Property": "1.5"}]
    assert atoms == [["C"]]
    assert coords[0].shape == (1, 3)
    assert pred_rows == [{"SMILES": "bad", "Property": "1.5"}]


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
