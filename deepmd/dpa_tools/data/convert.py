# SPDX-License-Identifier: LGPL-3.0-or-later
"""Format-agnostic data conversion.

Public entry point: ``auto_convert()`` — sniffs the input and routes to the
appropriate pipeline (SMILES→npy via ``smiles_to_npy``, or structure→npy via
``dpdata``).  CLI callers should use this instead of calling ``convert()``
or ``smiles_to_npy()`` directly.
"""

from __future__ import annotations

import csv
import glob as _glob
import json
import logging
from pathlib import Path
from typing import Union

import numpy as np

from deepmd.dpa_tools.data.validate import check_data

_LOG = logging.getLogger("dpa_tools")

# Recognised SMILES / molecule column names (case-insensitive).
_SMILES_COLUMNS = frozenset({"smiles", "smi", "mol"})


def _sniff_csv(path: str) -> set[str] | None:
    """Return the set of column names from a CSV file, or ``None`` if
    the file does not look like a table."""
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return None

            columns = []
            for header in reader.fieldnames:
                if header is None:
                    return None
                header = header.strip()
                if not header:
                    return None
                # Reject binary/malformed files that csv.DictReader otherwise
                # treats as a one-column header, e.g. b"\x00\x01\x02".
                if any(ord(ch) < 32 for ch in header):
                    return None
                columns.append(header.lower())
            return set(columns)
    except Exception:
        return None


def _sniff_xlsx(path: str) -> set[str]:
    """Return the set of column names from the first sheet of an Excel file,
    or ``None`` if pandas / openpyxl is not available."""
    try:
        import pandas as pd
    except ImportError:
        return None
    try:
        df = pd.read_excel(path, nrows=0, engine="openpyxl")
        return {str(h).lower() for h in df.columns}
    except Exception:
        return None


def _is_smiles_input(path: str) -> bool:
    """Return True if *path* looks like a CSV / Excel file whose columns
    contain at least one recognised SMILES / molecule identifier."""
    suffix = Path(path).suffix.lower()
    columns: set[str] | None = None
    if suffix == ".csv":
        columns = _sniff_csv(path)
    elif suffix in (".xlsx", ".xls"):
        columns = _sniff_xlsx(path)
    if columns is None:
        return False
    return bool(columns & _SMILES_COLUMNS)


# ---------------------------------------------------------------------------
# auto_convert — the single public entry point
# ---------------------------------------------------------------------------


def auto_convert(
    input_path: str,
    output_dir: str,
    *,
    fmt: str | None = None,
    type_map: list[str] | None = None,
    property_name: str = "Property",
    property_col: str = "Property",
    train_ratio: float = 0.9,
    smiles_col: str = "SMILES",
    mol_dir: str | None = None,
    seed: int = 42,
    poscar: str | None = None,
    formula_col: int | str = 0,
    base_element: str | None = None,
    sets: int = 1,
    overwrite: bool = False,
    validate: bool = True,
    strict: bool = False,
    verbose: bool = True,
) -> dict:
    """Convert any supported input to ``deepmd/npy``, auto-detecting the format.

    *If ``fmt="formula"``* the call delegates to
    :func:`~deepmd.dpa_tools.data.formula.formula_to_npy`, which reads a
    CSV of elemental composition formulas + property values, and generates
    doped structures from a template POSCAR via random substitution.

    *If the input is a CSV / Excel file with SMILES columns* the call
    delegates to :func:`~deepmd.dpa_tools.data.smiles.smiles_to_npy`, which
    generates 3D conformers (via RDKit), splits into train/valid, and writes
    the standard ``deepmd/npy`` layout.

    *Otherwise* the call delegates to ``dpdata`` with ``fmt="auto"`` (or the
    explicit *fmt* if provided), converting a single structure file (POSCAR,
    extxyz, cif, …) into ``deepmd/npy``.

    Returns a dict with keys ``"method"`` (``"formula"``, ``"smiles"``, or
    ``"dpdata"``) and any additional metadata the chosen backend provides.
    """
    # --- explicit SMILES hint, or auto-sniff ---
    is_smiles_fmt = isinstance(fmt, str) and fmt.lower() == "smiles"
    if is_smiles_fmt or (fmt is None and _is_smiles_input(input_path)):
        from deepmd.dpa_tools.data.smiles import smiles_to_npy

        result = smiles_to_npy(
            data={"dataset": input_path, "mol_dir": mol_dir},
            output_dir=output_dir,
            property_name=property_name,
            property_col=property_col,
            train_ratio=train_ratio,
            smiles_col=smiles_col,
            seed=seed,
            overwrite=overwrite,
        )
        converted = {
            "method": "smiles",
            "train_systems": result.train_systems,
            "valid_systems": result.valid_systems,
            "type_map": result.type_map,
            "samples_used": result.samples_used,
            "failed_rows": result.failed_rows,
            "skipped_zero": result.skipped_zero,
            "skipped_overlap": result.skipped_overlap,
        }
        if verbose:
            print(f"RDKit converted samples: {converted['samples_used']}")
            print(f"RDKit failed rows     : {len(converted['failed_rows'])}")
        return converted

    # --- explicit formula hint ---
    if fmt == "formula":
        from .formula import formula_to_npy

        out = formula_to_npy(
            csv_path=input_path,
            output_dir=output_dir,
            poscar=poscar,
            formula_col=formula_col,
            property_col=property_col,
            property_name=property_name,
            base_element=base_element,
            sets=sets,
            seed=seed,
        )
        if verbose:
            print(f"Formula conversion: {len(out)} systems written.")
        return {"method": "formula", "output_systems": out}

    # --- structure file → dpdata ---
    out = convert(
        input_path=input_path,
        output_dir=output_dir,
        fmt=fmt,
        type_map=type_map,
        validate=validate,
        strict=strict,
    )
    return {"method": "dpdata", "output_dir": out}


# ---------------------------------------------------------------------------
# convert() — thin dpdata wrapper (kept for programmatic use)
# ---------------------------------------------------------------------------

def convert(
    input_path: str,
    output_dir: str,
    fmt: str | None = None,
    type_map: list[str] = None,
    validate: bool = True,
    strict: bool = False,
) -> str:
    """Convert one or more structure files to ``deepmd/npy`` format.

    Thin wrapper over ``dpdata``.  When *fmt* is ``None`` (or ``"auto"``),
    dpdata auto-detects the format from the file extension or content.
    Explicit *fmt* values (``"extxyz"``, ``"vasp/poscar"``, ``"cif"``, …)
    are passed through to ``dpdata`` unchanged.

    Parameters
    ----------
    input_path : str
        Path or glob pattern to the input file(s) (e.g. ``"calcs/**/OUTCAR"``,
        ``"raw/*.sdf"``).  Wildcards (``*``, ``?``, ``[``) are expanded via
        :func:`glob.glob` with ``recursive=True``:

        - **No wildcards** — treated as a literal path; output goes directly
          into *output_dir*.
        - **Glob matches 1 file** — same as literal path (output → *output_dir*).
        - **Glob matches N > 1 files** — each match is converted into a numbered
          subdirectory ``{output_dir}/sys_{i:04d}/`` (zero-indexed, sorted).
        - **Glob matches nothing** — raises ``FileNotFoundError``.

    output_dir : str
        Destination directory for the deepmd/npy output.
    fmt : str, optional
        Format hint (e.g. ``"extxyz"``, ``"vasp/poscar"``).  Auto-detected
        when ``None``.
    type_map : list[str], optional
        Ordered element symbol list.
    validate : bool
        Run ``check_data()`` on the output after conversion.
    strict : bool
        Fail on the first validation issue instead of warning.

    Returns
    -------
    str
        Resolved path to the output directory.
    """
    # --- glob expansion ---
    input_str = str(input_path)
    if any(ch in input_str for ch in "*?["):
        matches = sorted(_glob.glob(input_str, recursive=True))
        if not matches:
            raise FileNotFoundError(f"No files matched pattern: {input_str}")
        if len(matches) == 1:
            # Single match — behave identically to literal path.
            input_files = [(matches[0], str(Path(output_dir).resolve()))]
        else:
            output_root = str(Path(output_dir).resolve())
            input_files = [
                (m, str(Path(output_root) / f"sys_{i:04d}"))
                for i, m in enumerate(matches)
            ]
    else:
        input_files = [(input_str, str(Path(output_dir).resolve()))]

    for _in_path, _out_dir in input_files:
        _convert_one(
            input_path=_in_path,
            output_dir=_out_dir,
            fmt=fmt,
            type_map=type_map,
            validate=validate,
            strict=strict,
        )

    return str(Path(output_dir).resolve())


# ---------------------------------------------------------------------------
# _convert_one() — single-file dpdata conversion (internal helper)
# ---------------------------------------------------------------------------


def _convert_one(
    input_path: str,
    output_dir: str,
    fmt: str | None = None,
    type_map: list[str] = None,
    validate: bool = True,
    strict: bool = False,
) -> str:
    """Convert a single structure file to ``deepmd/npy`` format.

    Internal helper called by :func:`convert` — do not use directly.
    """
    try:
        import dpdata
    except ImportError as e:
        raise ImportError(
            "dpdata is required for format conversion. "
            "Install it with: pip install dpdata"
        ) from e

    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    to_kwargs: dict = {}
    if type_map:
        to_kwargs["type_map"] = type_map

    # Try labeled first; dpdata auto-detects when fmt is None.
    load_kwargs = {"fmt": fmt} if fmt and fmt != "auto" else {}
    try:
        sys = dpdata.LabeledSystem(str(input_path), **load_kwargs)
    except Exception:
        sys = dpdata.System(str(input_path), **load_kwargs)

    sys.to("deepmd/npy", output_dir, **to_kwargs)

    if validate:
        # Re-load the newly-written directory to validate via dpdata API.
        try:
            loaded = dpdata.LabeledSystem(output_dir, fmt="deepmd/npy")
        except Exception:
            loaded = dpdata.System(output_dir, fmt="deepmd/npy")
        for issue in check_data(loaded, strict=strict):
            _LOG.warning("[Validation] %s", issue.description)

    return output_dir


# ---------------------------------------------------------------------------
# batch_convert() — glob many inputs into a mirrored deepmd/npy tree
# ---------------------------------------------------------------------------

def _glob_base(pattern: str) -> Path:
    """The fixed (non-wildcard) directory prefix of a glob pattern.

    Used to compute each match's path relative to the part of the pattern the
    user actually typed, so the output tree mirrors the input tree. For
    ``./calcs/**/OUTCAR`` the base is ``./calcs``.
    """
    base_parts: list[str] = []
    for part in Path(pattern).parts:
        if any(ch in part for ch in "*?["):
            break
        base_parts.append(part)
    base = Path(*base_parts) if base_parts else Path(".")
    # A pattern with no wildcard at all resolves to a file; mirror from its
    # parent so the single match still lands in its own subdirectory.
    if base.is_file():
        base = base.parent
    return base


def batch_convert(
    glob_pattern: str,
    output_dir: str,
    fmt: str,
    type_map: list[str] = None,
    validate: bool = True,
    strict: bool = False,
    recursive: bool = True,
) -> list[str]:
    """
    Convert every file matching a glob pattern to deepmd/npy in one call.

    The input directory tree is mirrored under ``output_dir``: a match at
    ``<base>/sub/run/OUTCAR`` (where ``<base>`` is the non-wildcard prefix of
    ``glob_pattern``) is written to ``<output_dir>/sub/run/OUTCAR/``. Using
    the file stem as the leaf directory keeps the layout collision-free even
    when one input directory holds several convertible files.

    A ``manifest.json`` recording inputs, outputs, and skipped files is
    written into ``output_dir``.

    Parameters
    ----------
    glob_pattern : str
        Glob pattern for the input files, e.g. ``"./calcs/**/OUTCAR"``.
    output_dir : str
        Root directory for the mirrored deepmd/npy output tree.
    fmt : str
        dpdata format string, applied to every match (see ``convert()``).
    type_map : list[str], optional
        Ordered element symbol list, passed through to ``convert()``.
    validate : bool
        Passed through to ``convert()`` — validate each converted system.
    strict : bool
        If True, the first failure (a conversion error or, when ``validate``
        is on, a validation issue) raises instead of being skipped. If False
        (default), failures are logged and skipped, and conversion continues.
    recursive : bool
        If True (default), ``**`` in the pattern matches across directories.

    Returns
    -------
    list[str]
        Resolved paths of the successfully created deepmd/npy directories,
        in sorted input order. Feeds directly into ``load_data()``.
    """
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    base = _glob_base(glob_pattern)
    matches = sorted(_glob.glob(glob_pattern, recursive=recursive))

    converted: list[dict] = []
    skipped: list[dict] = []

    for input_path in matches:
        in_path = Path(input_path)
        if not in_path.is_file():
            continue
        try:
            rel = in_path.relative_to(base)
        except ValueError:
            rel = Path(in_path.name)
        # Mirror the input tree; the file stem is the leaf system directory.
        out_sub = output_root / rel.parent / in_path.stem
        try:
            out = convert(
                input_path=str(in_path),
                output_dir=str(out_sub),
                fmt=fmt,
                type_map=type_map,
                validate=validate,
                strict=strict,
            )
            converted.append({"input": str(in_path), "output": out})
        except Exception as e:
            if strict:
                raise
            # Drop the output subdir if convert() created it but wrote
            # nothing — an empty dir would just make load_data() and the
            # split_* helpers choke later, and keeps the return value in
            # sync with what's actually on disk. A half-written dir (dpdata
            # crashed mid-write) is kept for debugging.
            if out_sub.exists() and not any(out_sub.iterdir()):
                try:
                    out_sub.rmdir()
                except OSError:
                    pass  # races / permissions — don't block the batch
            _LOG.warning("[batch_convert] skipping %s: %s", in_path, e)
            skipped.append({"input": str(in_path), "error": str(e)})

    manifest = {
        "glob_pattern": glob_pattern,
        "fmt": fmt,
        "type_map": type_map,
        "converted": converted,
        "skipped": skipped,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _LOG.info(
        "[batch_convert] %d converted, %d skipped — manifest: %s",
        len(converted), len(skipped), manifest_path,
    )

    return [c["output"] for c in converted]


# ---------------------------------------------------------------------------
# attach_labels() — property label injection using fit()'s head language
# ---------------------------------------------------------------------------

# Dict head types we know how to map to a DeePMD-kit data key.
# Anything outside this set is likely a typo; users should pass a plain string
# (e.g. head="force") for ad-hoc keys not listed here.
_KNOWN_DICT_HEAD_TYPES = frozenset({"property", "dos", "dipole", "polar"})


def _key_from_head(head: Union[str, dict]) -> str:
    """Derive the deepmd/npy filename key from a head specification.

    DeePMD-kit stores label ``key`` as ``set.*/key.npy``.  This function maps
    the same ``head`` vocabulary used by ``DPAFineTuner.fit()`` to that key.

    Rules
    -----
    - ``str``  → key is the string itself (``"energy"`` → ``energy.npy``)
    - ``dict`` with ``"property_name"``
      → key is ``head["property_name"]``
      (used with ``"type": "property"`` heads; confirmed by DeePMD-kit
      ``PropertyFittingNet`` docstring: "If the data file is named
      ``humo.npy``, this parameter should be ``'humo'``.")
    - ``{"type": "dos", ...}``    → ``dos.npy``
    - ``{"type": "dipole", ...}`` → ``dipole.npy``
    - ``{"type": "polar", ...}``  → ``polar.npy``

    Unknown dict ``type`` values raise ``ValueError`` with the supported list,
    rather than silently writing a file DeePMD-kit will never find.
    """
    if isinstance(head, str):
        return head

    if isinstance(head, dict):
        # property_name present → that IS the data key (overrides type check)
        if "property_name" in head:
            return head["property_name"]

        htype = head.get("type")
        if htype is None:
            raise ValueError(
                "head dict must contain 'property_name' or 'type'. "
                f"Got keys: {sorted(head.keys())}"
            )

        if htype not in _KNOWN_DICT_HEAD_TYPES:
            raise ValueError(
                f"Unknown dict head type {htype!r}. "
                f"Supported types: {sorted(_KNOWN_DICT_HEAD_TYPES)}. "
                f"For ad-hoc keys, pass a plain string instead: head={htype!r}"
            )

        if htype == "property":
            # "property" is a meta-type: the real key comes from property_name.
            # We already handled property_name above, so if we're here it's missing.
            raise ValueError(
                "head type 'property' requires a 'property_name' key "
                "(DeePMD-kit will read '{property_name}.npy'). "
                "Example: head={'type': 'property', 'property_name': 'bandgap', 'task_dim': 1}"
            )

        # dos / dipole / polar: key == type name
        return htype

    raise TypeError(
        f"head must be str or dict, got {type(head).__name__!r}"
    )


def attach_labels(
    system,
    head: Union[str, dict],
    values: np.ndarray,
) -> None:
    """
    Attach per-frame property labels to a dpdata system.

    Uses the same ``head`` specification language as ``DPAFineTuner.fit()``,
    so users only need to learn one vocabulary for describing properties.

    Labels are stored directly in the system's ``data`` dict under the
    resolved key.

    Parameters
    ----------
    system : dpdata.System or dpdata.LabeledSystem
        The target system (modified in-place).
    head : str | dict
        Property head specification — same as ``DPAFineTuner(head=...)``:

        - ``"energy"``
          → stores as ``system.data["energies"]``, shape ``(n_frames,)``
        - ``"bandgap"`` (any plain string)
          → stores as ``system.data["bandgap"]``, shape ``(n_frames,)`` or ``(n_frames, N)``
        - ``{"type": "property", "property_name": "bandgap", "task_dim": 1}``
          → stores as ``system.data["bandgap"]``, shape ``(n_frames, 1)``
        - ``{"type": "dos", "numb_dos": 250}``
          → stores as ``system.data["dos"]``, shape ``(n_frames, 250)``
    values : np.ndarray
        Per-frame label array. First axis must equal total number of frames
        in the system.

    Notes
    -----
    **Idempotency**: calling ``attach_labels`` twice with the *same* head on
    the same system overwrites the existing data. Calling with *different*
    heads writes separate keys.

    Examples
    --------
    >>> attach_labels(system, head="energy",
    ...               values=np.array([-12.3, -11.8, -13.1]))
    >>> attach_labels(system,
    ...               head={"type": "dos", "numb_dos": 250},
    ...               values=dos_array)   # shape (n_frames, 250)
    """
    key = _key_from_head(head)
    values = np.asarray(values, dtype=np.float64)

    coords = np.asarray(system.data["coords"])
    n_frames = coords.shape[0]

    if values.shape[0] != n_frames:
        raise ValueError(
            f"values has {values.shape[0]} frames but system "
            f"contains {n_frames} frames."
        )

    system.data[key] = values
