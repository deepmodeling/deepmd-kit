#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify dynamic-library resolution for native binaries and libraries."""

from __future__ import (
    annotations,
)

import argparse
import platform
import shutil
import subprocess
from dataclasses import (
    dataclass,
)
from pathlib import (
    Path,
)


@dataclass(frozen=True)
class LinkResult:
    """Represent one dynamic-link inspection result."""

    path: Path
    passed: bool
    detail: str


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a bounded native-link inspection command."""
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def check_dynamic_links(path: Path) -> LinkResult:
    """Check one native file for unresolved dynamic dependencies.

    Parameters
    ----------
    path : Path
        Native executable or library to inspect.

    Returns
    -------
    LinkResult
        Inspection status and actionable detail.
    """
    path = path.resolve(strict=False)
    if not path.is_file():
        return LinkResult(path, False, "file not found")
    system = platform.system()
    if system == "Linux":
        tool = shutil.which("ldd")
        command = [tool, str(path)] if tool is not None else None
    elif system == "Darwin":
        tool = shutil.which("otool")
        command = [tool, "-L", str(path)] if tool is not None else None
    else:
        return LinkResult(path, True, f"not applicable on {system}")
    if command is None:
        return LinkResult(path, False, "link inspection tool unavailable")
    try:
        completed = _run(command)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LinkResult(path, False, f"{type(exc).__name__}: {exc}")
    output = "\n".join((completed.stdout, completed.stderr))
    unresolved = [line.strip() for line in output.splitlines() if "not found" in line]
    if unresolved:
        return LinkResult(path, False, "; ".join(unresolved))
    if completed.returncode != 0:
        return LinkResult(path, False, f"returncode={completed.returncode}")
    return LinkResult(path, True, "all dynamic dependencies resolved")


def _collect_paths(
    explicit: list[Path], directory: Path | None, pattern: str
) -> list[Path]:
    """Collect unique native paths from CLI selectors."""
    paths = [item.resolve(strict=False) for item in explicit]
    if directory is not None:
        paths.extend(item.resolve(strict=False) for item in directory.glob(pattern))
    return list(dict.fromkeys(paths))


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and inspect native files.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to the process arguments.

    Returns
    -------
    int
        Zero when every selected file resolves, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", default=[], type=Path)
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--pattern", default="libdeepmd*.so")
    args = parser.parse_args(argv)
    paths = _collect_paths(args.path, args.directory, args.pattern)
    if not paths:
        print("FAIL  dynamic_links: no native files matched")
        return 1
    results = [check_dynamic_links(path) for path in paths]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status:<5} dynamic_links:{result.path}: {result.detail}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
