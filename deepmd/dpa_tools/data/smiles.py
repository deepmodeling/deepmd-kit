# SPDX-License-Identifier: LGPL-3.0-or-later
"""SMILES → 3D coordinates → deepmd/npy conversion.

Provides the molecular data ingestion pipeline originally from
``deepmd_property_tools``:

- Parse CSV files with SMILES (or pre-generated MOL files) and property labels
- Generate 3D conformers via RDKit (ETKDGv3 + MMFF/UFF optimisation)
- Validate structures (zero-coordinate rejection, overlapping-atom detection)
- Write ``deepmd/npy`` directories consumable by ``DPAFineTuner`` and friends
"""

from __future__ import annotations

import csv
import random
import re
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Period table, used to build a consistent per-checkpoint type_map.
ELEMENTS = np.array(
    [
        "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
        "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
        "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ]
)
ELEMENT_INDEX: dict[str, int] = {name: i for i, name in enumerate(ELEMENTS)}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_column(columns: list[str], choices: list[str]) -> str:
    lower_map = {col.lower(): col for col in columns}
    for choice in choices:
        if choice.lower() in lower_map:
            return lower_map[choice.lower()]
    raise KeyError(f"None of columns {choices} found in {columns}")


def _parse_property_value(raw_value: object) -> float:
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    text = str(raw_value).strip()
    try:
        return float(text)
    except ValueError:
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
        if match:
            return float(match.group(0))
        raise


# ---------------------------------------------------------------------------
# MOL file reader
# ---------------------------------------------------------------------------


def read_mol_coords(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Parse a V2000/V3000 MOL file, returning element symbols and (natoms,3) coords."""
    mol_path = Path(path)
    lines = mol_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 4:
        raise ValueError(f"Bad MOL file (too short): {mol_path}")

    counts = lines[3]
    try:
        natoms = int(counts[0:3])
    except ValueError:
        parts = counts.split()
        if not parts:
            raise ValueError(f"Bad MOL counts line: {mol_path}") from None
        natoms = int(parts[0])

    atom_lines = lines[4 : 4 + natoms]
    if len(atom_lines) != natoms:
        raise ValueError(f"Bad MOL atom block length: {mol_path}")

    symbols: list[str] = []
    coords: list[list[float]] = []
    for atom_line in atom_lines:
        if len(atom_line) >= 34:
            x = float(atom_line[0:10])
            y = float(atom_line[10:20])
            z = float(atom_line[20:30])
            symbol = atom_line[31:34].strip()
        else:
            parts = atom_line.split()
            if len(parts) < 4:
                raise ValueError(f"Bad MOL atom line: {mol_path}")
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            symbol = parts[3]
        if symbol not in ELEMENT_INDEX:
            raise ValueError(f"Unknown element {symbol!r} in {mol_path}")
        symbols.append(symbol)
        coords.append([x, y, z])

    return symbols, np.asarray(coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# SMILES → 3D (RDKit, lazy import)
# ---------------------------------------------------------------------------


def smiles_to_3d_coords(
    smiles: str, *, random_seed: int = 42,
) -> tuple[list[str], np.ndarray]:
    """Generate a 3D conformer from a SMILES string via RDKit ETKDGv3."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to generate 3D coordinates from SMILES. "
            "Install rdkit, or provide mol_dir with pre-generated MOL files."
        ) from exc

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = 1000
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(
            mol, randomSeed=int(random_seed), useRandomCoords=True,
            maxAttempts=2000, ignoreSmoothingFailures=True,
            enforceChirality=False,
        )
    if status != 0:
        raise ValueError(
            f"RDKit failed to embed 3D coordinates for SMILES: {smiles!r}"
        )
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass

    conf = mol.GetConformer()
    symbols: list[str] = []
    coords: list[list[float]] = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        if symbol not in ELEMENT_INDEX:
            raise ValueError(
                f"Unknown element {symbol!r} generated from SMILES {smiles!r}"
            )
        symbols.append(symbol)
        coords.append([pos.x, pos.y, pos.z])
    return symbols, np.asarray(coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _has_overlapping_atoms(coords: np.ndarray, tol: float) -> bool:
    if coords.shape[0] < 2:
        return False
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    return float(np.min(dist2)) < tol * tol


def _build_type_map_from_elements(used_elements: set[str]) -> list[str]:
    return [el for el in ELEMENTS.tolist() if el in used_elements]


# ---------------------------------------------------------------------------
# CSV record extractors
# ---------------------------------------------------------------------------

_Record = tuple[list[str], np.ndarray, float, int]  # symbols, coords, value, row_idx


def _records_from_csv_mol(
    dataset: str | Path,
    mol_dir: str | Path,
    property_col: str,
    mol_template: str = "id{row}.mol",
    overlap_tol: float = 1e-6,
) -> tuple[list[_Record], list[tuple[int, str, str]], int, int, list[dict[str, Any]]]:
    with Path(dataset).open("r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset}")
    prop_col = _find_column(list(rows[0].keys()), [property_col, "Property", "property"])

    records: list[_Record] = []
    failed_rows: list[tuple[int, str, str]] = []
    skipped_zero = 0
    skipped_overlap = 0
    kept_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        mol_path = (Path(mol_dir) / mol_template.format(row=row_idx)).resolve()
        try:
            symbols, coords = read_mol_coords(mol_path)
            if np.allclose(coords, 0.0):
                skipped_zero += 1
                continue
            if _has_overlapping_atoms(coords, overlap_tol):
                skipped_overlap += 1
                continue
            records.append(
                (symbols, coords, _parse_property_value(row[prop_col]), row_idx)
            )
            kept_rows.append(dict(row))
        except Exception as exc:
            failed_rows.append((row_idx, str(mol_path), str(exc)))
    return records, failed_rows, skipped_zero, skipped_overlap, kept_rows


def _records_from_csv_smiles(
    dataset: str | Path,
    property_col: str,
    smiles_col: str = "SMILES",
    overlap_tol: float = 1e-6,
    seed: int = 42,
) -> tuple[list[_Record], list[tuple[int, str, str]], int, int, list[dict[str, Any]]]:
    with Path(dataset).open("r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset}")
    prop_col = _find_column(list(rows[0].keys()), [property_col, "Property", "property"])
    smiles_column = _find_column(list(rows[0].keys()), [smiles_col, "SMILES", "smiles"])

    records: list[_Record] = []
    failed_rows: list[tuple[int, str, str]] = []
    skipped_zero = 0
    skipped_overlap = 0
    kept_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        smiles = row[smiles_column]
        try:
            symbols, coords = smiles_to_3d_coords(smiles, random_seed=seed + row_idx)
            if np.allclose(coords, 0.0):
                skipped_zero += 1
                continue
            if _has_overlapping_atoms(coords, overlap_tol):
                skipped_overlap += 1
                continue
            records.append(
                (symbols, coords, _parse_property_value(row[prop_col]), row_idx)
            )
            kept_rows.append(dict(row))
        except Exception as exc:
            failed_rows.append((row_idx, smiles, str(exc)))
    return records, failed_rows, skipped_zero, skipped_overlap, kept_rows


# ---------------------------------------------------------------------------
# public: full pipeline
# ---------------------------------------------------------------------------


@dataclass
class SmilesDataResult:
    output_dir: Path
    train_systems: list[str]
    valid_systems: list[str]
    type_map: list[str]
    failed_rows: list[tuple[int, str, str]]
    samples_used: int
    skipped_zero: int
    skipped_overlap: int


def smiles_to_npy(
    data: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path,
    property_name: str = "Property",
    property_col: str = "Property",
    train_ratio: float = 0.9,
    mol_dir: str | Path | None = None,
    mol_template: str = "id{row}.mol",
    smiles_col: str = "SMILES",
    overlap_tol: float = 1e-6,
    seed: int = 42,
    overwrite: bool = False,
) -> SmilesDataResult:
    """Convert a CSV of molecules (SMILES or MOL files) into ``deepmd/npy``.

    Parameters
    ----------
    data :
        Path to a CSV file, or a dict with ``"dataset"`` key.
    output_dir :
        Root directory for ``train/`` and ``valid/`` subdirectories.
    property_name :
        Name of the property label (stored as ``set.*/{property_name}.npy``).
    property_col :
        CSV column containing the target value.
    train_ratio :
        Fraction of samples used for training (remainder = validation).
    mol_dir :
        Directory containing pre-generated ``.mol`` files.  When omitted,
        SMILES are converted to 3D via RDKit.
    mol_template :
        Template for MOL filenames, e.g. ``"id{row}.mol"``.
    smiles_col :
        CSV column containing SMILES strings.
    overlap_tol :
        Minimum inter-atomic distance (Å) below which a structure is rejected.
    seed :
        Random seed for train/valid split and conformer generation.
    overwrite :
        If True, remove *output_dir* before writing.

    Returns
    -------
    SmilesDataResult
    """
    import dpdata
    from dpdata.data_type import Axis, DataType

    # Register the custom property + stru_id dtypes with dpdata.
    datatypes = [
        DataType(property_name, np.ndarray, shape=(Axis.NFRAMES, 1), required=False),
        DataType("stru_id", np.ndarray, shape=(Axis.NFRAMES, 1), required=False),
    ]
    for dtype in datatypes:
        dpdata.System.register_data_type(dtype)
        dpdata.LabeledSystem.register_data_type(dtype)

    # --- ingest ---
    if isinstance(data, (str, Path)) or (isinstance(data, dict) and "dataset" in data):
        dataset = Path(data if isinstance(data, (str, Path)) else data["dataset"])
        mol_dir_value = (
            mol_dir if mol_dir is not None
            else data.get("mol_dir") if isinstance(data, dict) else None
        )
        smiles_col_value = (
            data.get("smiles_col", smiles_col) if isinstance(data, dict) else smiles_col
        )
        if mol_dir_value is None:
            records, failed_rows, skipped_zero, skipped_overlap, _raw = (
                _records_from_csv_smiles(
                    dataset=dataset, property_col=property_col,
                    smiles_col=smiles_col_value, overlap_tol=overlap_tol, seed=seed,
                )
            )
        else:
            records, failed_rows, skipped_zero, skipped_overlap, _raw = (
                _records_from_csv_mol(
                    dataset=dataset, mol_dir=mol_dir_value,
                    property_col=property_col, mol_template=mol_template,
                    overlap_tol=overlap_tol,
                )
            )
    else:
        atoms = data.get("atoms")
        coordinates = data.get("coordinates")
        targets = data.get("target", data.get("targets"))
        if atoms is None or coordinates is None or targets is None:
            raise ValueError("Direct data requires atoms, coordinates, and target")
        records = [
            (list(s), np.asarray(c, dtype=np.float32), float(t), i)
            for i, (s, c, t) in enumerate(zip(atoms, coordinates, targets))
        ]
        failed_rows, skipped_zero, skipped_overlap = [], 0, 0

    for row_idx, source, error in failed_rows:
        warnings.warn(
            f"Skipping row {row_idx}: {source!r} — {error}", RuntimeWarning,
        )

    # --- deduplicate elements → type_map ---
    used_elements = {symbol for symbols, _, _, _ in records for symbol in symbols}
    type_map = _build_type_map_from_elements(used_elements)
    if not type_map:
        raise RuntimeError("No usable elements found after filtering.")
    type_index = {el: i for i, el in enumerate(type_map)}

    # --- build dpdata systems ---
    systems: list[dpdata.LabeledSystem] = []
    for symbols, coords, property_value, row_idx in records:
        natoms = len(symbols)
        if coords.shape != (natoms, 3):
            raise ValueError(f"coords shape mismatch for row {row_idx}: {coords.shape}")
        atom_types = np.array([type_index[s] for s in symbols], dtype=np.int32)
        frame_data = {
            "orig": np.array([0, 0, 0], dtype=np.int32),
            "atom_names": type_map,
            "atom_numbs": [np.count_nonzero(atom_types == i) for i in range(len(type_map))],
            "atom_types": atom_types,
            "cells": np.array([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 100.0]]]),
            "nopbc": True,
            "coords": coords[np.newaxis, :, :].astype(np.float32),
            "energies": np.zeros((1,), dtype=np.float32),
            "forces": np.zeros((1, natoms, 3), dtype=np.float32),
            property_name: np.array([[property_value]], dtype=np.float32),
            "stru_id": np.array([[row_idx]], dtype=np.int64),
        }
        systems.append(dpdata.LabeledSystem(data=frame_data, type_map=type_map))

    n_total = len(systems)
    if n_total < 2:
        raise RuntimeError(f"Not enough usable samples: {n_total}")

    # --- train / valid split ---
    output_path = Path(output_dir).resolve()
    if overwrite and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    train_count = max(1, min(int(n_total * train_ratio), n_total - 1))

    ms_train = dpdata.MultiSystems()
    ms_valid = dpdata.MultiSystems()
    for idx in indices[:train_count]:
        ms_train.append(systems[idx])
    for idx in indices[train_count:]:
        ms_valid.append(systems[idx])

    train_dir = output_path / "train"
    valid_dir = output_path / "valid"
    ms_train.to_deepmd_npy_mixed(str(train_dir))
    ms_valid.to_deepmd_npy_mixed(str(valid_dir))

    train_systems = sorted(
        str(p) for p in train_dir.iterdir() if p.is_dir()
    )
    valid_systems = sorted(
        str(p) for p in valid_dir.iterdir() if p.is_dir()
    )

    return SmilesDataResult(
        output_dir=output_path,
        train_systems=train_systems,
        valid_systems=valid_systems,
        type_map=type_map,
        failed_rows=failed_rows,
        samples_used=n_total,
        skipped_zero=skipped_zero,
        skipped_overlap=skipped_overlap,
    )
