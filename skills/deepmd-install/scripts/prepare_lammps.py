#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Add one managed DeePMD built-in include to a LAMMPS CMake project."""

from __future__ import (
    annotations,
)

import argparse
import os
import re
import tempfile
from pathlib import (
    Path,
)

BEGIN_MARKER = "# >>> DeePMD-kit built-in module >>>"
END_MARKER = "# <<< DeePMD-kit built-in module <<<"
LEGACY_PATTERN = re.compile(
    r"^\s*include\([^\n]*source/lmp/builtin\.cmake[^\n]*\)\s*$", re.MULTILINE
)


def _managed_block(builtin: Path) -> str:
    """Return the canonical managed CMake block."""
    return "\n".join(
        (
            BEGIN_MARKER,
            f"include([=[{builtin}]=])",
            END_MARKER,
        )
    )


def _replace_managed_block(text: str, block: str) -> tuple[str, bool]:
    """Replace a valid managed block or reject malformed markers."""
    begin_count = text.count(BEGIN_MARKER)
    end_count = text.count(END_MARKER)
    if begin_count == 0 and end_count == 0:
        return text, False
    if begin_count != 1 or end_count != 1:
        raise ValueError("LAMMPS CMakeLists.txt contains malformed DeePMD markers")
    begin = text.index(BEGIN_MARKER)
    end = text.index(END_MARKER, begin) + len(END_MARKER)
    return text[:begin] + block + text[end:], True


def render_updated_text(text: str, builtin: Path) -> str:
    """Return an idempotently patched CMakeLists.txt.

    Parameters
    ----------
    text : str
        Existing LAMMPS top-level CMake text.
    builtin : Path
        Absolute DeePMD-kit `source/lmp/builtin.cmake` path.

    Returns
    -------
    str
        Text containing exactly one managed include.

    Raises
    ------
    ValueError
        If multiple unmanaged includes or malformed markers are present.
    """
    block = _managed_block(builtin)
    updated, replaced = _replace_managed_block(text, block)
    if replaced:
        outside_block = updated.replace(block, "", 1)
        if LEGACY_PATTERN.search(outside_block) is not None:
            raise ValueError(
                "LAMMPS CMakeLists.txt contains an unmanaged DeePMD include"
            )
        return updated
    legacy_matches = list(LEGACY_PATTERN.finditer(text))
    if len(legacy_matches) > 1:
        raise ValueError("LAMMPS CMakeLists.txt contains multiple DeePMD includes")
    if legacy_matches:
        match = legacy_matches[0]
        return updated[: match.start()] + block + updated[match.end() :]
    separator = "" if updated.endswith("\n") else "\n"
    return f"{updated}{separator}\n{block}\n"


def _write_atomic(path: Path, text: str) -> None:
    """Atomically replace a text file while preserving its mode."""
    mode = path.stat().st_mode
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.chmod(temporary, mode & 0o7777)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare(lammps_source: Path, deepmd_source: Path, *, check: bool) -> bool:
    """Prepare or check the managed LAMMPS include.

    Parameters
    ----------
    lammps_source : Path
        Absolute LAMMPS source directory.
    deepmd_source : Path
        Absolute DeePMD-kit source directory.
    check : bool
        If true, compare without writing.

    Returns
    -------
    bool
        True when the file already had the canonical content.
    """
    lammps_source = lammps_source.resolve(strict=True)
    deepmd_source = deepmd_source.resolve(strict=True)
    cmake_file = lammps_source / "cmake" / "CMakeLists.txt"
    builtin = deepmd_source / "source" / "lmp" / "builtin.cmake"
    if not cmake_file.is_file():
        raise FileNotFoundError(f"LAMMPS CMake file not found: {cmake_file}")
    if not builtin.is_file():
        raise FileNotFoundError(f"DeePMD built-in module not found: {builtin}")
    original = cmake_file.read_text(encoding="utf-8")
    updated = render_updated_text(original, builtin)
    unchanged = updated == original
    if not check and not unchanged:
        _write_atomic(cmake_file, updated)
    return unchanged


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and prepare the LAMMPS source tree.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to the process arguments.

    Returns
    -------
    int
        Zero when prepared or already correct, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-source", required=True, type=Path)
    parser.add_argument("--deepmd-source", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        unchanged = prepare(args.lammps_source, args.deepmd_source, check=args.check)
    except (OSError, ValueError) as exc:
        print(f"FAIL  prepare_lammps: {exc}")
        return 1
    if args.check and not unchanged:
        print("FAIL  prepare_lammps: managed include differs from the plan")
        return 1
    status = "UNCHANGED" if unchanged else "UPDATED"
    print(f"PASS  prepare_lammps: {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
