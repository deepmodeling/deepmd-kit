# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest import mock

import numpy as np

from deepmd_property_tools import PropertyPredict
from deepmd_property_tools.data.mol import predict_records_from_data


def _write_mol(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "water",
                "  deepmd_property_tools",
                "",
                "  3  2  0  0  0  0            999 V2000",
                "    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0",
                "    0.9572    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "   -0.2390    0.9270    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0",
                "M  END",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_predict_records_from_csv_without_property_column(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    dataset.write_text("SMILES\nO\n", encoding="utf-8")
    mol_dir = tmp_path / "mol"
    mol_dir.mkdir()
    _write_mol(mol_dir / "id0.mol")

    atoms, coords, rows = predict_records_from_data(
        {"dataset": dataset, "mol_dir": mol_dir},
        property_col=None,
    )

    assert atoms == [["O", "H", "H"]]
    assert coords[0].shape == (3, 3)
    assert rows == [{"SMILES": "O"}]


def test_predict_directory_uses_latest_checkpoint(tmp_path: Path) -> None:
    old_checkpoint = tmp_path / "model.ckpt-1.pt"
    old_checkpoint.write_text("old", encoding="utf-8")
    time.sleep(0.01)
    latest_checkpoint = tmp_path / "model.ckpt-2.pt"
    latest_checkpoint.write_text("new", encoding="utf-8")
    (tmp_path / "property_tools_config.json").write_text(
        json.dumps({"type_map": ["H", "O"], "property_name": "Property"}),
        encoding="utf-8",
    )

    predictor = PropertyPredict(tmp_path)

    assert predictor.load_model == latest_checkpoint
    assert predictor.type_map == ["H", "O"]


def test_predict_directory_prefers_frozen_model(tmp_path: Path) -> None:
    frozen_model = tmp_path / "frozen_model.pth"
    frozen_model.write_text("frozen", encoding="utf-8")
    checkpoint = tmp_path / "model.ckpt-1.pt"
    checkpoint.write_text("checkpoint", encoding="utf-8")
    (tmp_path / "property_tools_config.json").write_text(
        json.dumps({"type_map": ["H"], "property_name": "Property"}),
        encoding="utf-8",
    )

    predictor = PropertyPredict(tmp_path)

    assert predictor.load_model == frozen_model


def test_predict_save_handles_single_output(tmp_path: Path) -> None:
    from deepmd_property_tools import predictor as predictor_module

    class DummyModel:
        def __init__(self, model_path: Path) -> None:
            self.model_path = model_path

        def eval(self, *args, **kwargs):
            return (np.array([[1.25]], dtype=float),)

    with mock.patch.object(predictor_module, "PropertyModel", DummyModel):
        predictor = predictor_module.Predictor(
            model_path=tmp_path / "model.ckpt-1.pt",
            type_map=["H"],
            property_name="Property",
        )
        y_pred = predictor.predict(
            atoms=[["H"]],
            coordinates=[np.array([[0.0, 0.0, 0.0]], dtype=np.float32)],
            rows=[{"SMILES": "[H]"}],
            save_path=tmp_path,
        )

    assert y_pred.tolist() == [[1.25]]
    assert (tmp_path / "test.predict.0.csv").read_text(encoding="utf-8").splitlines() == [
        "SMILES,predict_Property",
        "[H],1.25",
    ]
