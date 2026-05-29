# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prediction pipeline implementation."""

from __future__ import (
    annotations,
)

import csv
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np
from deepmd_property_tools.models import (
    PropertyModel,
)


class Predictor:
    def __init__(
        self, *, model_path: str | Path, type_map: list[str], property_name: str
    ) -> None:
        self.model_path = Path(model_path)
        self.type_map = type_map
        self.type_index = {element: idx for idx, element in enumerate(type_map)}
        self.property_name = property_name

    def predict(
        self,
        atoms: list[list[str]],
        coordinates: list[np.ndarray],
        rows: list[dict[str, Any]],
        save_path: str | Path | None = None,
        prefix: str = "test",
    ) -> np.ndarray:
        coords, atom_types = self.standardize(atoms, coordinates)
        y_pred = PropertyModel(self.model_path).eval(
            coords, None, atom_types, mixed_type=True
        )[0]
        if save_path is not None:
            self.save_predict(rows, y_pred, Path(save_path), prefix)
        return y_pred

    def standardize(
        self, atoms: list[list[str]], coordinates: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not atoms:
            raise ValueError("No samples to predict")
        max_natoms = max(len(symbols) for symbols in atoms)
        coords = np.zeros((len(atoms), max_natoms, 3), dtype=np.float32)
        atom_types = np.full((len(atoms), max_natoms), -1, dtype=np.int32)
        for frame_idx, (symbols, coord) in enumerate(zip(atoms, coordinates)):
            if coord.shape != (len(symbols), 3):
                raise ValueError(
                    f"coordinates shape mismatch at sample {frame_idx}: {coord.shape}"
                )
            for atom_idx, symbol in enumerate(symbols):
                if symbol not in self.type_index:
                    raise ValueError(
                        f"Element {symbol!r} is not present in type_map {self.type_map}"
                    )
                atom_types[frame_idx, atom_idx] = self.type_index[symbol]
            coords[frame_idx, : len(symbols), :] = coord
        return coords, atom_types

    def save_predict(
        self,
        rows: list[dict[str, Any]],
        y_pred: np.ndarray,
        save_path: Path,
        prefix: str,
    ) -> Path:
        save_path.mkdir(parents=True, exist_ok=True)
        out_path = save_path / f"{prefix}.predict.0.csv"
        run_id = 0
        while out_path.exists():
            run_id += 1
            out_path = save_path / f"{prefix}.predict.{run_id}.csv"

        predict_cols = [f"predict_{self.property_name}"]
        if y_pred.shape[1] > 1:
            predict_cols = [
                f"predict_{self.property_name}_{idx}" for idx in range(y_pred.shape[1])
            ]
        fieldnames = list(rows[0].keys()) if rows else []
        for col in predict_cols:
            if col not in fieldnames:
                fieldnames.append(col)
        with out_path.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row, pred in zip(rows, y_pred):
                out_row = dict(row)
                for col, value in zip(predict_cols, pred):
                    out_row[col] = float(value)
                writer.writerow(out_row)
        return out_path
