#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Prepare a tiny grouped assembly-property dataset.

The bundled CSV contains three real rows from a public component-based
molecular dataset. Each row is one supervised sample. Its component SMILES are
converted to structures, stored as DeePMD frames, and pooled through
``group_id.npy``, ``weight.npy``, and ``pool_mask.npy``.
"""

from __future__ import (
    annotations,
)

import logging
import os
import shutil
from pathlib import (
    Path,
)

from dpa_adapt.grouped._polymer import (
    PolymerBuilder,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEMO_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = DEMO_DIR / "data"
DEFAULT_CSV = DATA_DIR / "grouped_assembly_subset.csv"
TARGET = "target_property"

N_ROWS = int(os.environ.get("DPA_GROUP_PROPERTY_ROWS", "3"))
N_VALID = int(os.environ.get("DPA_GROUP_PROPERTY_VALID", "1"))


def _source_csv() -> Path:
    env_path = os.environ.get("DPA_GROUP_PROPERTY_CSV")
    candidates = [Path(env_path)] if env_path else []
    candidates.append(DEFAULT_CSV)
    for path in candidates:
        if path and path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find grouped_assembly_subset.csv. Set DPA_GROUP_PROPERTY_CSV "
        f"or place a compatible dataset at {DEFAULT_CSV}."
    )


def _clone_with_rows(source: PolymerBuilder, rows: list) -> PolymerBuilder:
    builder = PolymerBuilder(
        target=source.target, type_map=source.type_map, seed=source.seed
    )
    builder._rows = rows
    return builder


def _write_list(path: Path, systems: list[str], split: str) -> None:
    path.write_text(
        "".join(f"{split}/{system}\n" for system in systems), encoding="utf-8"
    )


def _ensure_text_newline(path: Path) -> None:
    if path.is_file():
        path.write_text(
            path.read_text(encoding="utf-8").rstrip() + "\n", encoding="utf-8"
        )


def _normalize_generated_json(split_dir: Path) -> None:
    old = split_dir / "polymer_scaler.json"
    scaler = split_dir / "fparam_scaler.json"
    if old.is_file():
        old.rename(scaler)
    _ensure_text_newline(scaler)
    _ensure_text_newline(split_dir / "manifest.json")


def main() -> None:
    csv_path = _source_csv()
    logger.info("reading grouped assembly CSV: %s", csv_path)
    builder = PolymerBuilder.from_csv(csv_path, target=TARGET)
    rows = builder._rows[:N_ROWS]
    if len(rows) <= N_VALID:
        raise ValueError(f"Need more than {N_VALID} usable rows; got {len(rows)}.")

    for split in ("train", "valid"):
        split_dir = DATA_DIR / split
        if split_dir.exists():
            shutil.rmtree(split_dir)

    train_rows = rows[:-N_VALID]
    valid_rows = rows[-N_VALID:]

    train = _clone_with_rows(builder, train_rows)
    train_res = train.write(DATA_DIR / "train", overwrite=True)
    _normalize_generated_json(DATA_DIR / "train")

    valid = _clone_with_rows(builder, valid_rows)
    valid.type_map = train_res["type_map"]
    valid_res = valid.write(
        DATA_DIR / "valid",
        overwrite=True,
        scaler=train_res["scaler"],
    )
    _normalize_generated_json(DATA_DIR / "valid")

    _write_list(DATA_DIR / "train_systems.txt", train_res["systems"], "train")
    _write_list(DATA_DIR / "valid_systems.txt", valid_res["systems"], "valid")

    logger.info(
        "wrote train groups: %d -> %s", train_res["n_groups"], DATA_DIR / "train"
    )
    logger.info(
        "wrote valid groups: %d -> %s", valid_res["n_groups"], DATA_DIR / "valid"
    )
    logger.info("fparam_dim: %d", train_res["fparam_dim"])


if __name__ == "__main__":
    main()
