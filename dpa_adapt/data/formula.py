# SPDX-License-Identifier: LGPL-3.0-or-later
"""Formula CSV + template POSCAR → deepmd/npy conversion.

Converts a CSV of elemental composition formulas (e.g.
``Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1``) and property values, paired with a
template POSCAR, into ``deepmd/npy`` systems via random atomic substitution
on the template's base-element sublattice.
"""

from __future__ import (
    annotations,
)

import csv
import random
import re
from pathlib import (
    Path,
)

import numpy as np

# Regex for one element–fraction pair in a formula string: "Ni0.65", "O2", "H1".
_ELEM_FRAC_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


# ---------------------------------------------------------------------------
# formula parsing
# ---------------------------------------------------------------------------


def parse_formula(
    formula_str: str,
    base_element: str | None = None,
) -> dict[str, float]:
    """Parse a composition formula string into element→fraction dict.

    ``"Ni0.65Gd0.15O2H1"`` → ``{"Ni": 0.65, "Gd": 0.15, "O": 2.0, "H": 1.0}``.

    The **substitution sublattice** fractions (everything except O and H) are
    normalised so they sum to 1.0.  O and H fractions are returned as-is
    (absolute stoichiometric counts).

    If *base_element* is given and is missing from the formula but the
    substitution-sublattice total is ≤ 1.0, the remainder is assigned to
    *base_element*.

    Parameters
    ----------
    formula_str : str
        Composition formula, e.g. ``"Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1"``.
    base_element : str | None
        Host element for the substitution sublattice.  Inferred as remainder
        when missing and total ≤ 1.0.

    Returns
    -------
    dict[str, float]
        Element symbols mapped to their fractions.
    """
    formula_str = formula_str.strip()
    fracs: dict[str, float] = {}
    for m in _ELEM_FRAC_RE.finditer(formula_str):
        elem = m.group(1)
        num_str = m.group(2)
        fracs[elem] = float(num_str) if num_str else 1.0

    if not fracs:
        raise ValueError(f"Could not parse any elements from {formula_str!r}")

    # Separate substitution-sublattice elements (non-O/H) from fixed lattice (O, H).
    sub_fracs = {k: v for k, v in fracs.items() if k not in ("O", "H")}
    fixed_fracs = {k: v for k, v in fracs.items() if k in ("O", "H")}

    total_sub = sum(sub_fracs.values())

    # Infer base_element from remainder BEFORE normalisation.
    if base_element is not None and base_element not in sub_fracs and total_sub < 1.0:
        remainder = round(1.0 - total_sub, 10)
        if remainder > 0:
            sub_fracs[base_element] = remainder
            total_sub = 1.0

    # Normalise substitution sublattice to 1.0.
    if sub_fracs and total_sub > 0:
        sub_fracs = {k: v / total_sub for k, v in sub_fracs.items()}

    # Reconstruct: substitution (normalised) + fixed lattice (unchanged).
    result = dict(sub_fracs)
    result.update(fixed_fracs)
    return result


# ---------------------------------------------------------------------------
# base element inference
# ---------------------------------------------------------------------------


def infer_base_element(symbols: list[str]) -> str | None:
    """Infer the substitution-sublattice host element from a list of atom symbols.

    Returns the most frequent element that is **not** O or H.
    Returns ``None`` if no such element is found.

    Parameters
    ----------
    symbols : list[str]
        Chemical symbols of all atoms (e.g. ``ase.Atoms.get_chemical_symbols()``).

    Returns
    -------
    str or None
    """
    counts: dict[str, int] = {}
    for s in symbols:
        if s not in ("O", "H"):
            counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


# ---------------------------------------------------------------------------
# random doping
# ---------------------------------------------------------------------------


def random_doping(
    base: ase.Atoms,
    fracs: dict[str, float],
    base_element: str,
    rng: random.Random,
) -> ase.Atoms:
    """Randomly replace *base_element* atoms in *base* according to *fracs*.

    *fracs* keys are the dopant elements; values are their fractions over the
    base-element sublattice.  Any base-element site not assigned a dopant
    remains *base_element*.  Dopants with a fraction that rounds to 0 atoms
    are skipped gracefully.

    Parameters
    ----------
    base : ase.Atoms
        Template structure.
    fracs : dict[str, float]
        Element → fraction mapping (substitution sublattice only).
    base_element : str
        Chemical symbol of the host element to substitute.
    rng : random.Random
        Seeded random instance for reproducibility.

    Returns
    -------
    ase.Atoms
        New ``Atoms`` object with doped chemical symbols.  Coordinates and
        cell are copied from *base*.
    """
    from ase import Atoms as AseAtoms

    symbols = list(base.get_chemical_symbols())
    indices = [i for i, s in enumerate(symbols) if s == base_element]
    n_sites = len(indices)

    if n_sites == 0:
        raise ValueError(
            f"base_element {base_element!r} not found in template POSCAR. "
            f"Available symbols: {sorted(set(symbols))}"
        )

    # Compute per-element atom counts; handle round-to-zero gracefully.
    counts: dict[str, int] = {}
    for elem, frac in fracs.items():
        if elem in ("O", "H"):
            continue  # fixed lattice — not part of substitution
        n = int(round(frac * n_sites))
        if n > 0:
            counts[elem] = n

    assigned = sum(counts.values())
    if assigned > n_sites:
        # Scale down proportionally to fit available sites.
        scale = n_sites / assigned
        counts = {e: max(1, int(round(c * scale))) for e, c in counts.items()}
        assigned = sum(counts.values())

    # Build the new symbol list for doping sites.
    dopant_list: list[str] = []
    for elem, n in counts.items():
        dopant_list.extend([elem] * n)
    # Remaining sites stay as base_element.
    remainder = n_sites - assigned
    if remainder > 0:
        dopant_list.extend([base_element] * remainder)

    rng.shuffle(indices)
    rng.shuffle(dopant_list)

    new_symbols = list(symbols)
    for idx, new_elem in zip(indices, dopant_list):
        new_symbols[idx] = new_elem

    doped = AseAtoms(
        symbols=new_symbols,
        positions=base.get_positions(),
        cell=base.get_cell(),
        pbc=base.get_pbc(),
    )
    return doped


# ---------------------------------------------------------------------------
# main conversion entry point
# ---------------------------------------------------------------------------


def formula_to_npy(
    csv_path: str,
    output_dir: str,
    poscar: str,
    formula_col: str = "formula",
    property_col: str = "Property",
    property_name: str = "Property",
    base_element: str | None = None,
    sets: int = 1,
    seed: int = 42,
) -> list[str]:
    """Convert a formula CSV + template POSCAR to ``deepmd/npy`` systems.

    CSV format: two or more named columns. The formula column holds composition
    strings (e.g. ``Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1``); the property
    column holds the scalar target value.

    For each CSV row, *sets* random doped structures are generated.  Each
    structure is written as a ``deepmd/npy`` system under
    ``output_dir/sys_{i:04d}/`` (zero-padded index across all rows × sets).

    Parameters
    ----------
    csv_path : str
        Path to the formula CSV file.
    output_dir : str
        Destination directory for ``deepmd/npy`` output.
    poscar : str
        Path to template POSCAR (VASP format).
    formula_col : str
        Column name for the formula.  Default: ``"formula"``.
    property_col : str
        Column name for the property value.  Default: ``"Property"``.
    property_name : str
        Label key written into each system (``set.000/{property_name}.npy``).
        Default: ``"Property"``.
    base_element : str | None
        Host element for random substitution.  Auto-inferred from the template
        POSCAR when ``None``.
    sets : int
        Number of random realisations per formula row.  Default: 1.
    seed : int
        Random seed for reproducibility.  Default: 42.

    Returns
    -------
    list[str]
        Resolved paths of the created ``deepmd/npy`` system directories.
    """
    import dpdata
    from ase.io import read as ase_read

    # Load template.
    template = ase_read(poscar, format="vasp")
    if base_element is None:
        base_element = infer_base_element(list(template.get_chemical_symbols()))
    if base_element is None:
        raise ValueError(
            "Could not infer base_element from template POSCAR. "
            "Pass base_element= explicitly."
        )

    # Parse CSV/TXT — headered delimited files, headerless delimited files when
    # columns are integer indices, or headerless whitespace files.
    rows: list[tuple[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        # Sniff delimiter from first non-empty line.
        first_line = ""
        for line in fh:
            if line.strip():
                first_line = line
                break
        fh.seek(0)
        delimiter = _sniff_table_delimiter(first_line)
        if delimiter is not None and _is_int_like(formula_col) and _is_int_like(
            property_col
        ):
            formula_idx = _resolve_col_index(formula_col)
            property_idx = _resolve_col_index(property_col)
            reader = csv.reader(fh, delimiter=delimiter)
            for line_no, fields in enumerate(reader, start=1):
                if not fields or all(v.strip() == "" for v in fields):
                    continue
                try:
                    formula_str = fields[formula_idx].strip()
                    prop_str = fields[property_idx].strip()
                except IndexError:
                    raise ValueError(
                        f"Line {line_no} in {csv_path!r} has {len(fields)} "
                        f"field(s), cannot read columns {formula_idx} and "
                        f"{property_idx}."
                    ) from None
                rows.append((formula_str, _parse_property_value(prop_str, line_no)))
        elif delimiter is None:
            formula_idx = _resolve_col_index(formula_col)
            property_idx = _resolve_col_index(property_col)
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                fields = line.split()
                try:
                    formula_str = fields[formula_idx].strip()
                    prop_str = fields[property_idx].strip()
                except IndexError:
                    raise ValueError(
                        f"Line {line_no} in {csv_path!r} has {len(fields)} "
                        f"field(s), cannot read columns {formula_idx} and "
                        f"{property_idx}."
                    ) from None
                rows.append((formula_str, _parse_property_value(prop_str, line_no)))
        else:
            raw_rows = [
                fields
                for fields in csv.reader(fh, delimiter=delimiter)
                if fields and any(v.strip() for v in fields)
            ]
            if not raw_rows:
                raise ValueError(f"No data rows found in formula CSV: {csv_path!r}")

            fieldnames = raw_rows[0]
            try:
                formula_header = _resolve_col(formula_col, fieldnames)
                try:
                    property_header = _resolve_col(property_col, fieldnames)
                except KeyError:
                    if property_col == "Property" and property_name != property_col:
                        property_header = _resolve_col(property_name, fieldnames)
                    else:
                        raise
            except KeyError:
                if not _looks_like_headerless_row(fieldnames):
                    raise
                for line_no, fields in enumerate(raw_rows, start=1):
                    if len(fields) < 2:
                        raise ValueError(
                            f"Line {line_no} in {csv_path!r} has {len(fields)} "
                            "field(s), cannot read default columns 0 and 1."
                        )
                    rows.append(
                        (
                            fields[0].strip(),
                            _parse_property_value(fields[1].strip(), line_no),
                        )
                    )
            else:
                reader = csv.DictReader(
                    [delimiter.join(row) for row in raw_rows[1:]],
                    fieldnames=fieldnames,
                    delimiter=delimiter,
                )
                for raw_row in reader:
                    if all((v or "").strip() == "" for v in raw_row.values()):
                        continue
                    formula_str = (raw_row.get(formula_header) or "").strip()
                    prop_str = (raw_row.get(property_header) or "").strip()
                    if not formula_str:
                        raise ValueError(
                            f"Empty formula value in column {formula_header!r}"
                        )
                    rows.append((formula_str, _parse_property_value(prop_str)))

    if not rows:
        raise ValueError(
            f"No data rows found in {csv_path!r}. "
            "Check that the file is a CSV with formula and property columns."
        )

    # Generate doped structures.
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    output_paths: list[str] = []
    sys_idx = 0

    for formula_str, prop_val in rows:
        fracs = parse_formula(formula_str, base_element=base_element)
        # Extract only substitution-sublattice fractions for doping.
        sub_fracs = {k: v for k, v in fracs.items() if k not in ("O", "H")}
        for _ in range(sets):
            doped = random_doping(template, sub_fracs, base_element, rng)
            sys_dir = out_root / f"sys_{sys_idx:04d}"
            sys_dir_str = str(sys_dir)

            # Convert ASE Atoms → dpdata System → deepmd/npy.
            symbols = list(doped.symbols)
            unique_symbols = sorted(set(symbols))
            symbol_to_idx = {s: i for i, s in enumerate(unique_symbols)}
            atom_types = np.array([symbol_to_idx[s] for s in symbols], dtype=int)
            atom_names = unique_symbols
            atom_numbs = [symbols.count(s) for s in unique_symbols]
            system = dpdata.System(
                data={
                    "atom_types": atom_types,
                    "atom_names": atom_names,
                    "atom_numbs": atom_numbs,
                    "coords": doped.positions[np.newaxis, :, :].astype(np.float64),
                    "cells": doped.cell.array[np.newaxis, :, :].astype(np.float64),
                    "orig": np.zeros(3, dtype=np.float64),
                }
            )
            # Attach label directly via attach_labels, then write out.
            # dpdata's to("deepmd/npy") only writes standard keys, so we
            # write the property label manually afterward.
            label_val = np.array([prop_val], dtype=np.float64)
            system.data[property_name] = label_val
            system.to("deepmd/npy", sys_dir_str)
            # Write the property label file manually into set.000/.
            set_dir = Path(sys_dir_str) / "set.000"
            set_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(set_dir / f"{property_name}.npy"), label_val)

            output_paths.append(sys_dir_str)
            sys_idx += 1

    return output_paths


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _resolve_col(
    spec: str,
    fieldnames: list[str],
) -> str:
    """Resolve a case-insensitive column name to the exact CSV header."""
    lower_map = {name.lower(): name for name in fieldnames if name is not None}
    key = str(spec).lower()
    if key in lower_map:
        return lower_map[key]
    raise KeyError(f"Column {spec!r} not found in CSV header {fieldnames}")


def _looks_like_headerless_row(fields: list[str]) -> bool:
    """Return True if a delimited row looks like ``formula,value`` data."""
    if len(fields) < 2:
        return False
    try:
        parse_formula(fields[0])
        float(fields[1])
    except ValueError:
        return False
    return True


def _sniff_table_delimiter(first_line: str) -> str | None:
    """Detect common one-character table delimiters."""
    for delimiter in ("\t", ",", ";", "|"):
        if delimiter in first_line:
            return delimiter
    return None


def _is_int_like(spec: int | str) -> bool:
    """Return True when *spec* can be used as a 0-based column index."""
    try:
        int(spec)
    except (TypeError, ValueError):
        return False
    return True


def _resolve_col_index(spec: int | str) -> int:
    """Resolve an integer-like column spec for headerless files."""
    try:
        idx = int(spec)
    except (TypeError, ValueError):
        raise ValueError(
            "Headerless formula files require integer column "
            f"indices, got {spec!r}."
        ) from None
    if idx < 0:
        raise ValueError(f"Column index must be non-negative, got {idx}.")
    return idx


def _parse_property_value(prop_str: str, line_no: int | None = None) -> float:
    """Parse a property value with a useful error message."""
    try:
        return float(prop_str)
    except ValueError:
        location = f" on line {line_no}" if line_no is not None else ""
        raise ValueError(f"Could not parse property value {prop_str!r}{location}") from None
