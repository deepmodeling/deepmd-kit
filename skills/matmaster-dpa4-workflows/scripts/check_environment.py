#!/usr/bin/env python3
"""Inventory a MatMaster/Bohrium job environment without exposing secrets."""

# ruff: noqa: T201 -- stdout is this command-line tool's result interface.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


COMMANDS = {
    "bohr": ["bohr", "version"],
    "dp": ["dp", "--version"],
    "lmp": ["lmp", "-h"],
    "lmp_mpi": ["lmp_mpi", "-h"],
    "nvidia_smi": [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(command: list[str]) -> dict:
    executable = shutil.which(command[0])
    result = {"path": executable, "available": executable is not None}
    if executable is None:
        return result
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=20, check=False
    )
    output = (completed.stdout or completed.stderr).strip().splitlines()
    result.update(
        {
            "returncode": completed.returncode,
            "first_line": output[0][:500] if output else "",
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument(
        "--require-file",
        action="append",
        default=[],
        help="Case-relative required file",
    )
    parser.add_argument("--probe", action="store_true", help="Run safe version probes")
    parser.add_argument("--require-bohr", action="store_true")
    parser.add_argument("--require-deepmd", action="store_true")
    parser.add_argument("--require-lammps", action="store_true")
    parser.add_argument(
        "--require-runtime", action="store_true", help="Require both DeePMD and LAMMPS"
    )
    args = parser.parse_args()

    commands = {}
    for name, command in COMMANDS.items():
        path = shutil.which(command[0])
        commands[name] = (
            probe(command)
            if args.probe
            else {"path": path, "available": path is not None}
        )
    report = {
        "commands": commands,
        "environment_present": {
            "BOHR_ACCESS_KEY": bool(os.environ.get("BOHR_ACCESS_KEY")),
            "ACCESS_KEY": bool(os.environ.get("ACCESS_KEY")),
            "PROJECT_ID": bool(os.environ.get("PROJECT_ID")),
            "OPENAPI_HOST": bool(os.environ.get("OPENAPI_HOST")),
            "TIEFBLUE_HOST": bool(os.environ.get("TIEFBLUE_HOST")),
        },
        "persistent_paths": {
            "/personal": {
                "exists": Path("/personal").exists(),
                "writable": os.access("/personal", os.W_OK),
            },
            "/share": {
                "exists": Path("/share").exists(),
                "writable": os.access("/share", os.W_OK),
            },
        },
    }
    if args.model:
        model = args.model.resolve()
        report["model"] = {
            "path": str(model),
            "exists": model.is_file(),
            "size": model.stat().st_size if model.is_file() else None,
            "sha256": sha256(model) if model.is_file() else None,
        }
    if args.case_dir:
        case = args.case_dir.resolve()
        required = ["run.sh", *args.require_file]
        report["case"] = {
            "path": str(case),
            "exists": case.is_dir(),
            "required": {name: (case / name).is_file() for name in required},
        }
    failures = []
    if args.require_bohr and not commands["bohr"]["available"]:
        failures.append("bohr_cli_missing")
    if (args.require_deepmd or args.require_runtime) and not commands["dp"][
        "available"
    ]:
        failures.append("dp_missing")
    if (args.require_lammps or args.require_runtime) and not (
        commands["lmp"]["available"] or commands["lmp_mpi"]["available"]
    ):
        failures.append("lammps_missing")
    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
