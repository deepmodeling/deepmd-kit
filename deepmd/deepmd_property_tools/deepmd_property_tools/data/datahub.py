# SPDX-License-Identifier: LGPL-3.0-or-later
"""Uni-Mol-style data hub for DeePMD property workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .converter import (
    PropertyDataResult,
    prepare_property_data,
)
from .mol import (
    predict_records_from_data,
)


class DataHub:
    def __init__(
        self,
        data: dict[str, Any] | str | Path,
        *,
        is_train: bool,
        save_path: str | Path,
        property_name: str = "Property",
        property_col: str | None = "Property",
        train_ratio: float = 0.9,
        mol_dir: str | Path | None = None,
        mol_template: str = "id{row}.mol",
        overlap_tol: float = 1e-6,
        seed: int = 42,
        overwrite: bool = False,
        numb_steps: int = 1000000,
        input_updates: dict[str, Any] | None = None,
    ) -> None:
        self.data_input = data
        self.is_train = is_train
        self.save_path = Path(save_path)
        self.property_name = property_name
        self.property_col = property_col
        if is_train:
            self.result: PropertyDataResult | None = prepare_property_data(
                data,
                output_dir=self.save_path / "prepared_data",
                input_out=self.save_path / "input.json",
                property_name=property_name,
                property_col=property_col,
                train_ratio=train_ratio,
                mol_dir=mol_dir,
                mol_template=mol_template,
                overlap_tol=overlap_tol,
                seed=seed,
                overwrite=overwrite,
                numb_steps=numb_steps,
                input_updates=input_updates,
            )
            self.type_map = self.result.type_map
            self.raw_data = self.result.raw_data
        else:
            self.result = None
            self.atoms, self.coordinates, self.raw_data = predict_records_from_data(
                data,
                property_col=property_col,
                mol_dir=mol_dir,
                mol_template=mol_template,
            )
