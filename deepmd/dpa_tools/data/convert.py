# data/convert.py

from __future__ import annotations

import glob as _glob
import json
import logging
from pathlib import Path
from typing import Union

import numpy as np

from deepmd.dpa_tools.data.validate import check_data

_LOG = logging.getLogger("dpa_tools")


# ---------------------------------------------------------------------------
# convert() — format conversion only, no label semantics
# ---------------------------------------------------------------------------

def convert(
    input_path: str,
    output_dir: str,
    fmt: str,
    type_map: list[str] = None,
    validate: bool = True,
    strict: bool = False,
) -> str:
    """
    Convert a structure/trajectory file to deepmd/npy format.

    This is a thin convenience wrapper over dpdata. For complex conversions
    (unit changes, selective atoms, multi-system merging) use dpdata directly.

    Labeled formats (extxyz, vasp/outcar, etc.) produce a complete deepmd/npy
    directory including ``energy.npy`` and ``force.npy``.
    Structure-only formats (vasp/poscar, cif) produce a directory with
    ``coord.npy`` and ``box.npy`` only. Use ``attach_labels()`` afterwards
    to add property labels before calling ``fit()``.

    Parameters
    ----------
    input_path : str
        Path to the input file or directory.
    output_dir : str
        Destination directory for the deepmd/npy output.
    fmt : str
        Input format string as accepted by dpdata, e.g. ``"extxyz"``,
        ``"vasp/outcar"``, ``"vasp/poscar"``, ``"cif"``.
        Must be provided explicitly — dpa_tools does not auto-detect formats.
    type_map : list[str], optional
        Ordered element symbol list (e.g. ``["Cu", "O"]``). Controls the
        integer encoding in ``type.raw`` and must match the target checkpoint's
        type_map. Strongly recommended — omitting it lets dpdata infer the
        order, which may not agree with the checkpoint.
    validate : bool
        If True (default), run ``check_data()`` on the output and emit any
        findings via ``logging.warning``. Set False to skip the check.
    strict : bool
        If True, ``check_data()`` raises ``DPADataError`` on the first issue
        instead of warning. Ignored when ``validate`` is False.

    Returns
    -------
    str
        Resolved path to the output deepmd/npy directory.

    Examples
    --------
    >>> from deepmd.dpa_tools.data import convert, load_data, attach_labels
    # Labeled format (energy + forces included):
    >>> convert("train.xyz", "./data/train", fmt="extxyz", type_map=["Cu", "O"])
    # Structure-only format, attach labels separately:
    >>> convert("POSCAR", "./data/single", fmt="vasp/poscar", type_map=["Cu", "O"])
    >>> system = load_data("./data/single")[0]
    >>> attach_labels(system, head="bandgap", values=np.array([1.23]))
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

    # Try labeled first; if the format carries no labels dpdata will just
    # produce a system with empty energy/force arrays, which is harmless.
    try:
        sys = dpdata.LabeledSystem(str(input_path), fmt=fmt)
    except Exception:
        sys = dpdata.System(str(input_path), fmt=fmt)

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
