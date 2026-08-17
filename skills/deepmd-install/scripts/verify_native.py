#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify dynamic-library resolution for native libraries."""

from __future__ import (
    annotations,
)

import argparse
import platform
import shutil
import subprocess
import sys
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


def _system_name() -> str:
    """Return the host operating-system name."""
    return platform.system()


def _find_executable(name: str) -> str | None:
    """Return the absolute path of an executable available on PATH."""
    return shutil.which(name)


def _darwin_load_command(path: Path) -> list[str]:
    """Build an isolated dyld probe for a loadable Mach-O library."""
    loader = (
        "import ctypes, os, sys; "
        "mode = getattr(os, 'RTLD_LOCAL', 0) | getattr(os, 'RTLD_NOW', 0); "
        "ctypes.CDLL(sys.argv[1], mode=mode)"
    )
    return [sys.executable, "-c", loader, str(path)]


def _default_pattern(system_name: str | None = None) -> str:
    """Return the platform-specific DeePMD native-library glob."""
    selected_system = _system_name() if system_name is None else system_name
    return "libdeepmd*.dylib" if selected_system == "Darwin" else "libdeepmd*.so"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a bounded native-link inspection command."""
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def check_dynamic_links(
    path: Path, *, darwin_probe: list[str] | None = None
) -> LinkResult:
    """Check one native file for unresolved dynamic dependencies.

    Parameters
    ----------
    path : Path
        Native executable or library to inspect.
    darwin_probe : list of str, optional
        Command that loads a Mach-O executable. Libraries use an isolated
        ``ctypes.CDLL`` probe when this argument is omitted.

    Returns
    -------
    LinkResult
        Inspection status and actionable detail.
    """
    path = path.resolve(strict=False)
    if not path.is_file():
        return LinkResult(path, False, "file not found")
    system = _system_name()
    if system == "Linux":
        tool = _find_executable("ldd")
        command = [tool, str(path)] if tool is not None else None
    elif system == "Darwin":
        command = darwin_probe or _darwin_load_command(path)
    else:
        return LinkResult(path, False, f"unsupported platform: {system}")
    if command is None:
        return LinkResult(path, False, "link inspection tool unavailable")
    try:
        completed = _run(command)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return LinkResult(path, False, f"{type(exc).__name__}: {exc}")
    output = "\n".join((completed.stdout, completed.stderr))
    unresolved = [
        line.strip() for line in output.splitlines() if "not found" in line.lower()
    ]
    if unresolved:
        return LinkResult(path, False, "; ".join(unresolved))
    if completed.returncode != 0:
        details = [line.strip() for line in output.splitlines() if line.strip()]
        suffix = f": {'; '.join(details[-4:])}" if details else ""
        return LinkResult(path, False, f"returncode={completed.returncode}{suffix}")
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
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        type=Path,
        help="Native library path.",
    )
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--pattern", default=_default_pattern())
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
