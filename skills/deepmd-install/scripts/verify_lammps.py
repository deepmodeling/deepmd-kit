#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Verify a DeePMD-enabled LAMMPS binary and its required pair styles."""

from __future__ import (
    annotations,
)

import argparse
import json
import platform
import re
import shutil
import subprocess
from dataclasses import (
    asdict,
    dataclass,
)
from pathlib import (
    Path,
)
from typing import (
    Any,
)


@dataclass(frozen=True)
class CheckResult:
    """Represent one LAMMPS verification result."""

    name: str
    passed: bool
    detail: str


def required_styles(model_family: str, flavor: str) -> list[str]:
    """Return exact pair styles required by a runtime selection.

    Parameters
    ----------
    model_family : str
        `conventional`, `dpa4`, or `dpa4c`.
    flavor : str
        `host` or `kokkos-cuda`.

    Returns
    -------
    list of str
        Required pair-style tokens.
    """
    base = "dpa4spin" if model_family == "dpa4c" else "deepmd"
    result = [base]
    if flavor == "kokkos-cuda":
        result.append(f"{base}/kk")
    return result


def _resolve_binary(value: str) -> Path | None:
    """Resolve an absolute path or executable name."""
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    resolved = shutil.which(value)
    return Path(resolved) if resolved is not None else None


def _run(
    command: list[str], *, timeout: float = 30.0
) -> subprocess.CompletedProcess[str]:
    """Run a bounded LAMMPS verification command."""
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _check_help(binary: Path, styles: list[str]) -> list[CheckResult]:
    """Run LAMMPS help and check exact style tokens."""
    try:
        completed = _run([str(binary), "-h"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [CheckResult("lammps_help", False, f"{type(exc).__name__}: {exc}")]
    output = "\n".join((completed.stdout, completed.stderr))
    results = [
        CheckResult(
            "lammps_help",
            completed.returncode == 0,
            f"binary={binary} returncode={completed.returncode}",
        )
    ]
    tokens = set(re.findall(r"[A-Za-z0-9_.+/-]+", output))
    for style in styles:
        results.append(CheckResult(f"pair_style:{style}", style in tokens, style))
    return results


def _check_links(binary: Path) -> CheckResult:
    """Check native library resolution on Linux or macOS."""
    system = platform.system()
    if system == "Linux":
        tool = shutil.which("ldd")
        command = [tool, str(binary)] if tool is not None else None
    elif system == "Darwin":
        tool = shutil.which("otool")
        command = [tool, "-L", str(binary)] if tool is not None else None
    else:
        return CheckResult("dynamic_links", True, f"not applicable on {system}")
    if command is None:
        return CheckResult("dynamic_links", False, "link inspection tool unavailable")
    try:
        completed = _run(command)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("dynamic_links", False, f"{type(exc).__name__}: {exc}")
    output = "\n".join((completed.stdout, completed.stderr))
    unresolved = [line.strip() for line in output.splitlines() if "not found" in line]
    passed = completed.returncode == 0 and not unresolved
    detail = (
        "; ".join(unresolved) if unresolved else f"returncode={completed.returncode}"
    )
    return CheckResult("dynamic_links", passed, detail)


def run_checks(
    *, binary: str, model_family: str, flavor: str, check_links: bool
) -> list[CheckResult]:
    """Verify a LAMMPS executable.

    Parameters
    ----------
    binary : str
        Absolute path or executable name.
    model_family : str
        Model family used to derive pair styles.
    flavor : str
        Host or Kokkos CUDA build.
    check_links : bool
        Whether to inspect dynamic-library resolution.

    Returns
    -------
    list of CheckResult
        Ordered verification results.
    """
    resolved = _resolve_binary(binary)
    if resolved is None or not resolved.is_file():
        return [CheckResult("lammps_binary", False, f"not found: {binary}")]
    if not resolved.stat().st_mode & 0o111:
        return [CheckResult("lammps_binary", False, f"not executable: {resolved}")]
    results = [CheckResult("lammps_binary", True, str(resolved))]
    results.extend(_check_help(resolved, required_styles(model_family, flavor)))
    if check_links:
        results.append(_check_links(resolved))
    return results


def _print_results(checks: list[CheckResult], *, as_json: bool) -> None:
    """Print verification results."""
    if as_json:
        payload: dict[str, Any] = {
            "passed": all(item.passed for item in checks),
            "checks": [asdict(item) for item in checks],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for item in checks:
        status = "PASS" if item.passed else "FAIL"
        print(f"{status:<5} {item.name}: {item.detail}")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and verify LAMMPS.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments. Defaults to the process arguments.

    Returns
    -------
    int
        Zero when every requested check passes, otherwise one.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True)
    parser.add_argument(
        "--model-family",
        required=True,
        choices=("conventional", "dpa4", "dpa4c"),
    )
    parser.add_argument("--flavor", required=True, choices=("host", "kokkos-cuda"))
    parser.add_argument("--check-links", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    checks = run_checks(
        binary=args.binary,
        model_family=args.model_family,
        flavor=args.flavor,
        check_links=args.check_links,
    )
    _print_results(checks, as_json=args.json)
    return 0 if all(item.passed for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
