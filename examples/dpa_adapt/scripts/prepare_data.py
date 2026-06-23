#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
# One-time data preparation script. Data is already included in
# examples/dpa_adapt/data/. Only re-run if you need to regenerate from raw GDB9.
"""Download QM9 GDB9 and prepare deepmd/npy systems for the quickstart demo.

Reads molecules 1-8 from the SDF, reads HOMO-LUMO gaps from the companion
CSV file, stages a small 8-row dataset, converts it with ``dpa_adapt.convert``,
and splits into 5 training and 3 test systems.

Usage::

    python scripts/prepare_data.py

Can be run from anywhere; all paths are resolved relative to the
``examples/dpa_adapt/`` directory (the parent of this script).
"""

from __future__ import (
    annotations,
)

import csv
import shutil
import tarfile
import urllib.request
from pathlib import (
    Path,
)

import numpy as np

from dpa_adapt import (
    convert,
)

# This script lives in examples/dpa_adapt/scripts/; resolve data and raw dirs
# against examples/dpa_adapt/.
DEMO_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DEMO_DIR / "raw"
DATA_DIR = DEMO_DIR / "data"
STAGED_DIR = RAW_DIR / "qm9_8"
STAGED_MOL_DIR = STAGED_DIR / "mol"
STAGED_CSV_PATH = STAGED_DIR / "qm9_8.csv"
SDF_PATH = RAW_DIR / "gdb9.sdf"
CSV_PATH = RAW_DIR / "gdb9.sdf.csv"
TAR_PATH = RAW_DIR / "gdb9.tar.gz"
TAR_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"

N_TRAIN = 5
N_TEST = 3
N_TOTAL = N_TRAIN + N_TEST
BOX_LENGTH = 100.0  # Angstrom, cubic box for non-periodic systems
TYPE_MAP = ["H", "C", "N", "O", "F"]

# Hartree to eV conversion factor
HARTREE_TO_EV = 27.211386245988


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _download_and_extract(force: bool = False) -> None:
    """Download and extract gdb9.tar.gz if the data files don't already exist."""
    if SDF_PATH.exists() and CSV_PATH.exists() and not force:
        print(f"SDF already present: {SDF_PATH}")
        print(f"CSV already present: {CSV_PATH}")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not TAR_PATH.exists() or force:
        print(f"Downloading {TAR_URL} ...")
        urllib.request.urlretrieve(TAR_URL, TAR_PATH)
        print(f"Downloaded -> {TAR_PATH}")

    print("Extracting from tarball ...")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name in ("gdb9.sdf", "gdb9.sdf.csv"):
                if not (RAW_DIR / name).exists() or force:
                    print(
                        f"  Extracting {name} ({member.size / 1024 / 1024:.1f} MB) ..."
                    )
                    tar.extract(member, path=str(RAW_DIR))
    print("Extraction complete.")


def _load_gaps_from_csv(n: int) -> dict[int, float]:
    """Read the first *n* rows from the GDB9 CSV, return {index: gap_ev}.

    The CSV columns include ``mol_id``, ``homo``, ``lumo``, ``gap``.
    Values are in Hartree; returned values are in eV.
    The *mol_id* is ``gdb_N``; we map to 0-based index N-1.
    """
    gaps: dict[int, float] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mol_id = row["mol_id"]  # e.g. "gdb_1"
            idx = int(mol_id.split("_")[1]) - 1  # 0-based
            if idx >= n:
                break
            # Use pre-computed gap if available; otherwise lumo - homo.
            if row.get("gap"):
                gap_ha = float(row["gap"])
            else:
                gap_ha = float(row["lumo"]) - float(row["homo"])
            gaps[idx] = gap_ha * HARTREE_TO_EV
    return gaps


def _read_sdf_blocks(n: int) -> list[str]:
    """Read the first *n* molecule blocks from the SDF file.

    GDB9 molecules are separated by ``$$$$``.
    """
    print(f"Reading {SDF_PATH} ...")
    raw_text = SDF_PATH.read_text(encoding="utf-8")

    blocks = raw_text.split("$$$$")
    blocks = [b.strip() for b in blocks if b.strip()]
    print(f"Found {len(blocks)} molecules in SDF.")

    if len(blocks) < n:
        raise RuntimeError(f"Expected at least {n} molecules, found {len(blocks)}")
    return blocks[:n]


def _stage_qm9_subset(
    mol_blocks: list[str],
    gaps: np.ndarray,
) -> None:
    """Write an 8-row CSV plus one single-molecule SDF per row."""
    if STAGED_DIR.exists():
        shutil.rmtree(STAGED_DIR)
    STAGED_MOL_DIR.mkdir(parents=True)

    with STAGED_CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["mol_id", "gap"])
        writer.writeheader()
        for i, (block, gap) in enumerate(zip(mol_blocks, gaps)):
            (STAGED_MOL_DIR / f"id{i}.sdf").write_text(
                block.strip() + "\n$$$$\n",
                encoding="utf-8",
            )
            writer.writerow({"mol_id": f"gdb_{i + 1}", "gap": f"{float(gap):.10f}"})


def _collect_labels(system_dirs: list[str]) -> np.ndarray:
    """Collect all gap labels from generated system directories."""
    chunks = []
    for sys_dir in sorted(Path(p) for p in system_dirs):
        for set_dir in sorted(sys_dir.glob("set.*")):
            chunks.append(np.load(set_dir / "gap.npy").reshape(-1))
    if not chunks:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("DPA Tools - Quickstart Data Preparation")
    print("=" * 60)

    # 1. Download & extract --------------------------------------------------
    _download_and_extract()

    # 2. Read gaps from CSV --------------------------------------------------
    all_gaps = _load_gaps_from_csv(N_TOTAL)
    gaps = np.array([all_gaps[i] for i in range(N_TOTAL)], dtype=np.float32)

    print(
        f"Gap stats (all {N_TOTAL}): mean={gaps.mean():.4f} eV, std={gaps.std():.4f} eV"
    )

    # 3. Read molecules from SDF ---------------------------------------------
    mol_blocks = _read_sdf_blocks(N_TOTAL)

    # 4. Stage the 8-row raw subset ------------------------------------------
    _stage_qm9_subset(mol_blocks, gaps)

    # 5. Convert to deepmd/npy via dpa_adapt.convert --------------------------
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    result = convert(
        str(STAGED_CSV_PATH),
        str(DATA_DIR),
        fmt="smiles",
        mol_dir=str(STAGED_MOL_DIR),
        mol_template="id{row}.sdf",
        property_col="gap",
        property_name="gap",
        train_ratio=N_TRAIN / N_TOTAL,
        split_seed=42,
        overwrite=True,
        verbose=False,
    )

    # Keep the historical demo layout: data/test rather than data/valid.
    valid_dir = DATA_DIR / "valid"
    test_dir = DATA_DIR / "test"
    valid_dir.rename(test_dir)
    train_systems = sorted(result["train_systems"])
    test_systems = sorted(str(p) for p in test_dir.iterdir() if p.is_dir())

    # 6. Write aggregated labels in generated-system order --------------------
    train_labels = _collect_labels(train_systems)
    test_labels = _collect_labels(test_systems)
    np.save(str(DATA_DIR / "train_labels.npy"), train_labels)
    np.save(str(DATA_DIR / "test_labels.npy"), test_labels)
    print(
        f"  train systems -> {DATA_DIR / 'train'} "
        f"({len(train_systems)} dirs, {train_labels.shape[0]} samples)"
    )
    print(
        f"  test systems  -> {test_dir} "
        f"({len(test_systems)} dirs, {test_labels.shape[0]} samples)"
    )

    # 7. Summary --------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"n_train : {N_TRAIN}")
    print(f"n_test  : {N_TEST}")
    print(f"gap mean: {gaps.mean():.4f} eV")
    print(f"gap std : {gaps.std():.4f} eV")
    print("Done. Run one of the evaluation scripts next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
