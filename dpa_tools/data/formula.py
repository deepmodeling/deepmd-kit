# SPDX-License-Identifier: LGPL-3.0-or-later
"""Formula CSV + template POSCAR → deepmd/npy conversion.

Converts a CSV of elemental composition formulas (e.g.
``Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1``) and property values, paired with a
template POSCAR, into ``deepmd/npy`` systems via random atomic substitution
on the template's base-element sublattice.
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

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
    if (
        base_element is not None
        and base_element not in sub_fracs
        and total_sub < 1.0
    ):
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
    base: "ase.Atoms",
    fracs: dict[str, float],
    base_element: str,
    rng: random.Random,
) -> "ase.Atoms":
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
    formula_col: int | str = 0,
    property_col: int | str = 1,
    property_name: str = "property",
    base_element: str | None = None,
    sets: int = 1,
    seed: int = 42,
) -> list[str]:
    """Convert a formula CSV + template POSCAR to ``deepmd/npy`` systems.

    CSV format: two or more columns.  The formula column holds composition
    strings (e.g. ``Ni0.65Gd0.15Fe0.10Co0.05Yb0.05O2H1``); the property
    column holds the scalar target value.  Header auto-detected: if the first
    data row's property column cannot be parsed as ``float``, that row is
    skipped as a header.

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
    formula_col : int | str
        Column index (0-based) or column name for the formula.  Default: 0.
    property_col : int | str
        Column index (0-based) or column name for the property value.  Default: 1.
    property_name : str
        Label key written into each system (``set.000/{property_name}.npy``).
        Default: ``"property"``.
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

    # Parse CSV — auto-detect delimiter (tab or comma).
    rows: list[tuple[str, float]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        # Sniff delimiter from first non-empty line.
        first_line = ""
        for line in fh:
            if line.strip():
                first_line = line
                break
        delimiter = "\t" if "\t" in first_line else ","
        fh.seek(0)
        reader = csv.reader(fh, delimiter=delimiter)
        for raw_row in reader:
            if not raw_row or all(c.strip() == "" for c in raw_row):
                continue
            row_values = [c.strip() for c in raw_row]
            # Resolve column indices from names if needed.
            fidx = _resolve_col(formula_col, row_values, allow_name=True)
            pidx = _resolve_col(property_col, row_values, allow_name=True)
            formula_str = row_values[fidx]
            prop_str = row_values[pidx]
            try:
                prop_val = float(prop_str)
            except ValueError:
                # Likely a header row — skip.
                continue
            rows.append((formula_str, prop_val))

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
    spec: int | str,
    row_values: list[str],
    allow_name: bool = False,
) -> int:
    """Resolve a column specifier to an integer index.

    - *int* → used directly.
    - *str* + ``allow_name=True`` → looks up the column name in *row_values*
      (case-insensitive), falling back to ``int(spec)``.
    """
    if isinstance(spec, int):
        return spec
    if allow_name:
        lower_map = {v.lower(): i for i, v in enumerate(row_values)}
        key = spec.lower()
        if key in lower_map:
            return lower_map[key]
    return int(spec)
