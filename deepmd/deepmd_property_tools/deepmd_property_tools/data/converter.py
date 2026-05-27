# SPDX-License-Identifier: LGPL-3.0-or-later
"""DeepMD mixed-npy conversion for property labels."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from deepmd_property_tools.config import ConfigHandler

from .mol import (
    build_used_type_map,
    records_from_csv_mol,
    records_from_direct_data,
)


@dataclass
class PropertyDataResult:
    input_path: Path
    output_dir: Path
    train_systems: list[str]
    valid_systems: list[str]
    type_map: list[str]
    failed_rows: list[tuple[int, str, str]]
    samples_used: int
    skipped_zero: int
    skipped_overlap: int
    raw_data: list[dict[str, Any]]


def register_extra_dtypes(property_name: str) -> None:
    import dpdata
    from dpdata.data_type import Axis, DataType

    datatypes = [
        DataType(property_name, np.ndarray, shape=(Axis.NFRAMES, 1), required=False),
        DataType("stru_id", np.ndarray, shape=(Axis.NFRAMES, 1), required=False),
    ]
    for dtype in datatypes:
        dpdata.System.register_data_type(dtype)
        dpdata.LabeledSystem.register_data_type(dtype)


def to_relative_path(path: Path, base: Path) -> str:
    path_abs = path.resolve()
    base_abs = base.resolve()
    try:
        return str(path_abs.relative_to(base_abs))
    except ValueError:
        return os.path.relpath(path_abs, base_abs)


def build_frame(
    *,
    symbols: list[str],
    coords: np.ndarray,
    property_value: float,
    stru_id: int,
    property_name: str,
    type_map: list[str],
    type_index: dict[str, int],
) -> dict[str, Any]:
    natoms = len(symbols)
    if coords.shape != (natoms, 3):
        raise ValueError(f"coords shape mismatch for stru_id={stru_id}: {coords.shape}")

    atom_types = np.array([type_index[s] for s in symbols], dtype=np.int32)
    atom_numbs = np.zeros(len(type_map), dtype=np.int32)
    for idx in atom_types:
        atom_numbs[idx] += 1

    return {
        "orig": np.array([0, 0, 0], dtype=np.int32),
        "atom_names": type_map,
        "atom_numbs": atom_numbs.tolist(),
        "atom_types": atom_types,
        "cells": np.array([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]]),
        "nopbc": True,
        "coords": coords[np.newaxis, :, :].astype(np.float32),
        "energies": np.zeros((1,), dtype=np.float32),
        "forces": np.zeros((1, natoms, 3), dtype=np.float32),
        property_name: np.array([[property_value]], dtype=np.float32),
        "stru_id": np.array([[stru_id]], dtype=np.int64),
    }


def default_input(
    *,
    property_name: str,
    train_systems: list[str],
    valid_systems: list[str],
    type_map: list[str],
    numb_steps: int = 1000000,
    input_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = ConfigHandler().read()
    config["model"]["type_map"] = type_map
    config["model"]["fitting_net"]["property_name"] = property_name
    config["training"]["training_data"]["systems"] = train_systems
    config["training"]["validation_data"]["systems"] = valid_systems
    config["training"]["numb_steps"] = numb_steps
    return ConfigHandler.merge(config, input_updates)


def prepare_property_data(
    data: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    input_out: str | Path,
    property_name: str = "Property",
    property_col: str = "Property",
    train_ratio: float = 0.9,
    mol_dir: str | Path | None = None,
    mol_template: str = "id{row}.mol",
    overlap_tol: float = 1e-6,
    seed: int = 42,
    overwrite: bool = False,
    numb_steps: int = 1000000,
    input_updates: dict[str, Any] | None = None,
) -> PropertyDataResult:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0, 1)")

    import dpdata

    register_extra_dtypes(property_name)

    failed_rows: list[tuple[int, str, str]] = []
    skipped_zero = 0
    skipped_overlap = 0
    if isinstance(data, (str, Path)) or (isinstance(data, dict) and "dataset" in data):
        dataset = Path(data if isinstance(data, (str, Path)) else data["dataset"])
        mol_dir_value = mol_dir if mol_dir is not None else data.get("mol_dir")
        if mol_dir_value is None:
            raise ValueError("mol_dir is required for CSV/MOL data")
        records, failed_rows, skipped_zero, skipped_overlap, raw_data = records_from_csv_mol(
            dataset=dataset,
            mol_dir=mol_dir_value,
            property_col=property_col,
            mol_template=mol_template,
            overlap_tol=overlap_tol,
        )
    else:
        records, raw_data = records_from_direct_data(data)

    used_elements = {symbol for symbols, _, _, _ in records for symbol in symbols}
    type_map = build_used_type_map(used_elements)
    if not type_map:
        raise RuntimeError("No usable elements found after filtering.")
    type_index = {el: i for i, el in enumerate(type_map)}

    systems: list[dpdata.LabeledSystem] = []
    for symbols, coords, property_value, row_idx in records:
        frame_data = build_frame(
            symbols=symbols,
            coords=coords,
            property_value=property_value,
            stru_id=row_idx,
            property_name=property_name,
            type_map=type_map,
            type_index=type_index,
        )
        systems.append(dpdata.LabeledSystem(data=frame_data, type_map=type_map))

    n_total = len(systems)
    if n_total < 2:
        raise RuntimeError(f"Not enough usable samples: {n_total}")

    output_path = Path(output_dir).resolve()
    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    train_count = int(n_total * train_ratio)
    train_count = max(1, min(train_count, n_total - 1))

    ms_train = dpdata.MultiSystems()
    ms_valid = dpdata.MultiSystems()
    for idx in indices[:train_count]:
        ms_train.append(systems[idx])
    for idx in indices[train_count:]:
        ms_valid.append(systems[idx])

    ms_train.to_deepmd_npy_mixed(str(train_dir))
    ms_valid.to_deepmd_npy_mixed(str(valid_dir))

    input_path = Path(input_out).resolve()
    path_base = input_path.parent
    train_systems = sorted(to_relative_path(path, path_base) for path in train_dir.iterdir() if path.is_dir())
    valid_systems = sorted(to_relative_path(path, path_base) for path in valid_dir.iterdir() if path.is_dir())
    if not train_systems or not valid_systems:
        raise RuntimeError("Generated system directories are empty.")

    input_dict = default_input(
        property_name=property_name,
        train_systems=train_systems,
        valid_systems=valid_systems,
        type_map=type_map,
        numb_steps=numb_steps,
        input_updates=input_updates,
    )
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(input_dict, indent=2) + "\n", encoding="utf-8")

    fail_csv = output_path / "failed_rows.csv"
    with fail_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["row_index", "mol_path", "error"])
        writer.writerows(failed_rows)

    return PropertyDataResult(
        input_path=input_path,
        output_dir=output_path,
        train_systems=train_systems,
        valid_systems=valid_systems,
        type_map=type_map,
        failed_rows=failed_rows,
        samples_used=n_total,
        skipped_zero=skipped_zero,
        skipped_overlap=skipped_overlap,
        raw_data=raw_data,
    )
