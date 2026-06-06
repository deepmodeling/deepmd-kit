#!/usr/bin/env python3
# One-time data preparation script. Data is already included in
# demo/data/. Only re-run if you need to regenerate from raw GDB9.
"""Download QM9 GDB9 and prepare deepmd/npy systems for the quickstart demo.

Reads molecules 1–50 from the SDF, reads HOMO-LUMO gaps from the companion
CSV file, converts each molecule to ``deepmd/npy`` format with a 100 Å cubic
box, and splits into 40 training and 10 test systems.

Usage::

    python scripts/prepare_data.py

Can be run from anywhere; all paths are resolved relative to the ``demo/``
directory (the parent of this script).
"""

from __future__ import annotations

import csv
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

# This script lives in demo/scripts/; resolve data and raw dirs against demo/.
DEMO_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = DEMO_DIR / "raw"
DATA_DIR = DEMO_DIR / "data"
SDF_PATH = RAW_DIR / "gdb9.sdf"
CSV_PATH = RAW_DIR / "gdb9.sdf.csv"
TAR_PATH = RAW_DIR / "gdb9.tar.gz"
TAR_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"

N_TRAIN = 40
N_TEST = 10
N_TOTAL = N_TRAIN + N_TEST
BOX_LENGTH = 100.0  # Å — cubic box for non-periodic systems
TYPE_MAP = ["H", "C", "N", "O", "F"]

# Hartree → eV conversion factor
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
        print(f"Downloading {TAR_URL} …")
        urllib.request.urlretrieve(TAR_URL, TAR_PATH)
        print(f"Downloaded → {TAR_PATH}")

    print("Extracting from tarball …")
    with tarfile.open(TAR_PATH, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if name in ("gdb9.sdf", "gdb9.sdf.csv"):
                if not (RAW_DIR / name).exists() or force:
                    print(f"  Extracting {name} ({member.size / 1024 / 1024:.1f} MB) …")
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
            if "gap" in row and row["gap"]:
                gap_ha = float(row["gap"])
            else:
                gap_ha = float(row["lumo"]) - float(row["homo"])
            gaps[idx] = gap_ha * HARTREE_TO_EV
    return gaps


def _read_sdf_blocks(n: int) -> list[str]:
    """Read the first *n* molecule blocks from the SDF file.

    GDB9 molecules are separated by ``$$$$``.
    """
    print(f"Reading {SDF_PATH} …")
    raw_text = SDF_PATH.read_text(encoding="utf-8")

    blocks = raw_text.split("$$$$")
    blocks = [b.strip() for b in blocks if b.strip()]
    print(f"Found {len(blocks)} molecules in SDF.")

    if len(blocks) < n:
        raise RuntimeError(f"Expected at least {n} molecules, found {len(blocks)}")
    return blocks[:n]


# ---------------------------------------------------------------------------
# V2000 SDF parser (dpdata's built-in SDF reader does not support System.from)
# ---------------------------------------------------------------------------

_ELEMENT_TO_Z: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
    "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24,
    "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
    "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38,
    "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45,
    "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50, "Sb": 51, "Te": 52,
    "I": 53, "Xe": 54, "Cs": 55, "Ba": 56,
}


def _parse_v2000_block(mol_block: str) -> tuple[list[str], np.ndarray]:
    """Parse a V2000 SDF molecule block, returning (symbols, coords).

    coords shape: (n_atoms, 3), float32.
    """
    lines = mol_block.strip().split("\n")

    # Find the counts line (contains "V2000" or "V3000")
    counts_idx = None
    for i, line in enumerate(lines):
        if "V2000" in line:
            counts_idx = i
            break
    if counts_idx is None:
        raise ValueError("No V2000 counts line found in SDF block")

    counts_line = lines[counts_idx]
    n_atoms = int(counts_line[:3].strip())

    symbols: list[str] = []
    coords_list: list[tuple[float, float, float]] = []

    for i in range(counts_idx + 1, counts_idx + 1 + n_atoms):
        line = lines[i]
        x = float(line[0:10].strip())
        y = float(line[10:20].strip())
        z = float(line[20:30].strip())
        symbol = line[31:34].strip()
        # Handle two-letter symbols like "Cl", "Br" where the first char
        # might be at column 31 and the second at 32.
        if not symbol:
            # Fallback: try wider extraction
            symbol = line[30:34].strip()
        symbols.append(symbol)
        coords_list.append((x, y, z))

    coords = np.array(coords_list, dtype=np.float32)
    return symbols, coords


def _system_to_npy(
    mol_block: str,
    output_dir: Path,
    gap_ev: float,
) -> None:
    """Convert one SDF molecule block to ``deepmd/npy`` and attach the label.

    Parses the V2000 block manually and creates a dpdata System with a
    100 Å cubic box.
    """
    import dpdata

    symbols, coords = _parse_v2000_block(mol_block)
    n_atoms = len(symbols)

    # Build local type_map index
    _type_to_idx = {s: i for i, s in enumerate(TYPE_MAP)}
    atom_types = np.array([_type_to_idx[s] for s in symbols], dtype=np.int32)

    # Count atoms per type
    atom_numbs = [int((atom_types == i).sum()) for i in range(len(TYPE_MAP))]

    sys = dpdata.System()
    sys.data["atom_names"] = list(TYPE_MAP)
    sys.data["atom_numbs"] = atom_numbs
    sys.data["atom_types"] = atom_types
    sys.data["coords"] = coords.reshape(1, n_atoms, 3)
    sys.data["cells"] = np.tile(np.eye(3) * BOX_LENGTH, (1, 1, 1)).reshape(1, 3, 3)
    sys.data["orig"] = np.zeros(3)
    sys.data["nopbc"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    sys.to("deepmd/npy", str(output_dir))

    # Write the label as gap.npy so DPAFineTuner.evaluate() finds it via
    # target_key="gap".
    set_dir = output_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(set_dir / "gap.npy"), np.array([gap_ev], dtype=np.float32))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("DPA Tools — Quickstart Data Preparation")
    print("=" * 60)

    # 1. Download & extract --------------------------------------------------
    _download_and_extract()

    # 2. Read gaps from CSV --------------------------------------------------
    all_gaps = _load_gaps_from_csv(N_TOTAL)
    gaps = np.array([all_gaps[i] for i in range(N_TOTAL)], dtype=np.float32)

    print(f"Gap stats (all {N_TOTAL}): "
          f"mean={gaps.mean():.4f} eV, std={gaps.std():.4f} eV")

    # 3. Read molecules from SDF ---------------------------------------------
    mol_blocks = _read_sdf_blocks(N_TOTAL)

    # 4. Split ---------------------------------------------------------------
    train_blocks = mol_blocks[:N_TRAIN]
    test_blocks = mol_blocks[N_TRAIN:]
    train_gaps = gaps[:N_TRAIN]
    test_gaps = gaps[N_TRAIN:]

    # 5. Convert to deepmd/npy ------------------------------------------------
    # Train
    train_dir = DATA_DIR / "train"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    for i, (block, gap) in enumerate(zip(train_blocks, train_gaps)):
        out = train_dir / f"sys_{i:04d}"
        print(f"  train [{i + 1}/{N_TRAIN}] → {out}")
        _system_to_npy(block, out, float(gap))

    # Test
    test_dir = DATA_DIR / "test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    for i, (block, gap) in enumerate(zip(test_blocks, test_gaps)):
        out = test_dir / f"sys_{i:04d}"
        print(f"  test  [{i + 1}/{N_TEST}]  → {out}")
        _system_to_npy(block, out, float(gap))

    # 6. Write aggregated labels ---------------------------------------------
    np.save(str(DATA_DIR / "train_labels.npy"), train_gaps.astype(np.float32))
    np.save(str(DATA_DIR / "test_labels.npy"), test_gaps.astype(np.float32))

    # 7. Summary --------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"n_train : {N_TRAIN}")
    print(f"n_test  : {N_TEST}")
    print(f"gap mean: {gaps.mean():.4f} eV")
    print(f"gap std : {gaps.std():.4f} eV")
    print("Done. Run fit_evaluate.py next.")
    print("=" * 60)


if __name__ == "__main__":
    main()
