# data/loader.py
#
# Polymorphic entry point: normalises str / Path / glob / dpdata objects
# into a flat list[dpdata.System].  Disk I/O and format detection are
# delegated to dpdata.

from __future__ import annotations

import glob as _glob
from pathlib import Path
from typing import List, Optional, Union

import dpdata

from dpa_adapt.data.errors import DPADataError

_SOURCE_ATTR = "_dpa_source"

# Backward-compat key aliases: old code used "energy"/"force" but dpdata
# stores them as "energies"/"forces".  Single source of truth — all other
# modules import from here.
_LABEL_KEY_ALIASES = {
    "energy": "energies",
    "force": "forces",
}


def _resolve_label_key(key: str) -> str:
    """Map legacy label keys to dpdata's canonical names."""
    return _LABEL_KEY_ALIASES.get(key, key)


# Type alias covering every form the public API accepts.
_SystemLike = Union[str, Path, dpdata.System, dpdata.LabeledSystem]
_DataInput = Union[_SystemLike, List[_SystemLike]]


def _get_source(system) -> Optional[str]:
    """Return the source path stored on a system, or None."""
    return getattr(system, _SOURCE_ATTR, None)


def load_data(
    data: _DataInput,
    fmt: Optional[str] = None,
) -> List[dpdata.System]:
    """
    Normalise arbitrary data input into a flat list of ``dpdata.System``.

    This is the single polymorphic entry point for all data in dpa_adapt.
    Every internal consumer receives its data through this function so that
    disk-access logic lives in exactly one place.

    Parameters
    ----------
    data : str | Path | dpdata.System | dpdata.LabeledSystem | list
        - **str / Path** — a deepmd/npy system directory (or any path that
          dpdata can open).  If the string contains glob wildcards (``*``,
          ``?``) it is expanded and every match is loaded.
        - **dpdata.System / dpdata.LabeledSystem** — passed through as-is
          (no deep copy).
        - **list** — each element is processed recursively and the results
          are flattened into a single list.
    fmt : str, optional
        dpdata format string.  Defaults to ``"deepmd/npy"`` for paths;
        ignored when *data* is already a dpdata object.

    Returns
    -------
    list[dpdata.System]
        One ``dpdata.System`` (or ``LabeledSystem``) per resolved input.
    """
    # 1. List → recurse and flatten
    if isinstance(data, list):
        result: List[dpdata.System] = []
        for item in data:
            result.extend(load_data(item, fmt=fmt))
        return result

    # 2. Glob string → expand, then recurse
    if isinstance(data, str) and _glob.has_magic(data):
        matches = sorted(Path(p) for p in _glob.glob(data))
        if not matches:
            raise DPADataError(
                f"Glob pattern {data!r} matched no files or directories."
            )

        # Fail-fast: deepmd/npy (the default) only works on directories.
        load_fmt = fmt if fmt is not None else "deepmd/npy"
        if load_fmt == "deepmd/npy":
            non_dirs = [str(m) for m in matches if not m.is_dir()]
            if non_dirs:
                raise DPADataError(
                    f"Glob pattern {data!r} matched non-directory paths "
                    f"incompatible with fmt={load_fmt!r}: {non_dirs}. "
                    "Pass fmt= explicitly or load these separately."
                )

        result: List[dpdata.System] = []
        for match in matches:
            result.extend(load_data(match, fmt=fmt))
        return result

    # 3. dpdata object → pass through (no copy)
    if isinstance(data, (dpdata.System, dpdata.LabeledSystem)):
        return [data]

    # 4. str / Path → delegate to dpdata
    path = str(data)
    if not Path(path).exists():
        raise DPADataError(f"Path does not exist: {path!r}")

    load_fmt = fmt if fmt is not None else "deepmd/npy"

    # Try labeled first so that training labels are preserved when present.
    try:
        system: dpdata.System = dpdata.LabeledSystem(path, fmt=load_fmt)
    except Exception:
        try:
            system = dpdata.System(path, fmt=load_fmt)
        except Exception as exc:
            raise DPADataError(
                f"Failed to load {path!r} via dpdata (fmt={load_fmt!r}): {exc}"
            ) from exc

    # Stamp source path so downstream consumers (e.g. cv formula extraction)
    # can recover the original filesystem location.
    setattr(system, _SOURCE_ATTR, str(Path(path).resolve()))

    return [system]
