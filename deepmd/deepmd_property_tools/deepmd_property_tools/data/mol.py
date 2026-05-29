# SPDX-License-Identifier: LGPL-3.0-or-later
"""MOL and direct-coordinate data helpers."""

from __future__ import (
    annotations,
)

import csv
import re
from pathlib import (
    Path,
)
from typing import (
    Any,
)

import numpy as np

ELEMENTS = np.array(
    [
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    ]
)
ELEMENT_INDEX = {name: i for i, name in enumerate(ELEMENTS)}


def find_column(columns: list[str], choices: list[str]) -> str:
    lower_map = {col.lower(): col for col in columns}
    for choice in choices:
        if choice.lower() in lower_map:
            return lower_map[choice.lower()]
    raise KeyError(f"None of columns {choices} found in {columns}")


def parse_property_value(raw_value: object) -> float:
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


def read_mol_coords(path: str | Path) -> tuple[list[str], np.ndarray]:
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


def has_overlapping_atoms(coords: np.ndarray, tol: float) -> bool:
    if coords.shape[0] < 2:
        return False
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist2, np.inf)
    return float(np.min(dist2)) < tol * tol


def build_used_type_map(used_elements: set[str]) -> list[str]:
    return [el for el in ELEMENTS.tolist() if el in used_elements]


def records_from_csv_mol(
    *,
    dataset: str | Path,
    mol_dir: str | Path,
    property_col: str,
    mol_template: str = "id{row}.mol",
    overlap_tol: float = 1e-6,
) -> tuple[
    list[tuple[list[str], np.ndarray, float, int]],
    list[tuple[int, str, str]],
    int,
    int,
    list[dict[str, Any]],
]:
    with Path(dataset).open("r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset}")
    prop_col = find_column(list(rows[0].keys()), [property_col, "Property", "property"])

    records: list[tuple[list[str], np.ndarray, float, int]] = []
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
            if has_overlapping_atoms(coords, overlap_tol):
                skipped_overlap += 1
                continue
            records.append(
                (symbols, coords, parse_property_value(row[prop_col]), row_idx)
            )
            kept_rows.append(dict(row))
        except Exception as exc:
            failed_rows.append((row_idx, str(mol_path), str(exc)))
    return records, failed_rows, skipped_zero, skipped_overlap, kept_rows


def records_from_direct_data(
    data: dict[str, Any],
) -> tuple[list[tuple[list[str], np.ndarray, float, int]], list[dict[str, Any]]]:
    atoms = data.get("atoms")
    coordinates = data.get("coordinates")
    targets = data.get("target", data.get("targets"))
    if atoms is None or coordinates is None or targets is None:
        raise ValueError("Direct training data requires atoms, coordinates, and target")
    if not (len(atoms) == len(coordinates) == len(targets)):
        raise ValueError("atoms, coordinates, and target must have the same length")
    records = []
    rows = []
    for idx, (symbols, coords, target) in enumerate(zip(atoms, coordinates, targets)):
        records.append(
            (list(symbols), np.asarray(coords, dtype=np.float32), float(target), idx)
        )
        rows.append({"sample_id": idx, "target": float(target)})
    return records, rows


def predict_records_from_data(
    data: dict[str, Any] | str | Path,
    *,
    property_col: str | None = "Property",
    mol_dir: str | Path | None = None,
    mol_template: str = "id{row}.mol",
) -> tuple[list[list[str]], list[np.ndarray], list[dict[str, Any]]]:
    if isinstance(data, (str, Path)) or (isinstance(data, dict) and "dataset" in data):
        dataset = Path(data if isinstance(data, (str, Path)) else data["dataset"])
        mol_dir_value = mol_dir if mol_dir is not None else data.get("mol_dir")
        if mol_dir_value is None:
            raise ValueError("mol_dir is required for CSV/MOL data")
        resolved_mol_dir = Path(mol_dir_value)
        with dataset.open("r", encoding="utf-8") as fp:
            rows = list(csv.DictReader(fp))
        if rows and property_col is not None:
            find_column(list(rows[0].keys()), [property_col, "Property", "property"])
        atoms: list[list[str]] = []
        coords: list[np.ndarray] = []
        kept_rows: list[dict[str, Any]] = []
        for row_idx, row in enumerate(rows):
            symbols, coord = read_mol_coords(
                resolved_mol_dir / mol_template.format(row=row_idx)
            )
            atoms.append(symbols)
            coords.append(coord)
            kept_rows.append(dict(row))
        return atoms, coords, kept_rows

    atoms_raw = data.get("atoms")
    coords_raw = data.get("coordinates")
    if atoms_raw is None or coords_raw is None:
        raise ValueError("Prediction data requires atoms and coordinates")
    atoms = [list(symbols) for symbols in atoms_raw]
    coords = [np.asarray(coord, dtype=np.float32) for coord in coords_raw]
    if len(atoms) != len(coords):
        raise ValueError("atoms and coordinates must have the same length")
    rows = [{"sample_id": idx} for idx in range(len(atoms))]
    return atoms, coords, rows
