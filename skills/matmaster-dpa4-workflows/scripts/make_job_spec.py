#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate and validate a Bohrium container job specification."""

# ruff: noqa: T201 -- stdout is this command-line tool's result interface.

from __future__ import (
    annotations,
)

import argparse
import json
import shlex
from pathlib import (
    Path,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--command", default="bash run.sh")
    parser.add_argument("--log-file", default="run.log")
    parser.add_argument("--backward", action="append", default=[])
    parser.add_argument("--result-path")
    parser.add_argument("--max-run-time", type=int)
    parser.add_argument("--max-reschedule-times", type=int, default=0)
    parser.add_argument("--nnode", type=int, default=1)
    parser.add_argument("--disk-size", type=int)
    args = parser.parse_args()

    case_dir = args.case_dir.resolve()
    errors = []
    if not case_dir.is_dir():
        errors.append(f"case directory does not exist: {case_dir}")
    if "/" not in args.image or ":" not in args.image:
        errors.append("image must be a full registry path with a tag")
    if "/root/input" in args.command:
        errors.append("command must not assume /root/input; use relative paths")
    try:
        tokens = shlex.split(args.command)
    except ValueError as exc:
        errors.append(f"invalid command quoting: {exc}")
        tokens = []
    if (
        tokens[:2] == ["bash", "run.sh"]
        and case_dir.is_dir()
        and not (case_dir / "run.sh").is_file()
    ):
        errors.append("default command requires case_dir/run.sh")
    if args.project_id <= 0:
        errors.append("project ID must be positive")
    if args.max_run_time is not None and args.max_run_time <= 0:
        errors.append("max run time must be positive minutes")
    if args.max_reschedule_times < 0 or args.nnode <= 0:
        errors.append("reschedule count must be nonnegative and nnode positive")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2

    spec = {
        "job_name": args.name,
        "command": args.command,
        "log_file": args.log_file,
        "backward_files": args.backward,
        "project_id": args.project_id,
        "machine_type": args.machine,
        "image_address": args.image,
        "job_type": "container",
        "max_reschedule_times": args.max_reschedule_times,
        "nnode": args.nnode,
    }
    if args.result_path:
        spec["result_path"] = args.result_path
    if args.max_run_time is not None:
        spec["max_run_time"] = args.max_run_time
    if args.disk_size is not None:
        spec["disk_size"] = args.disk_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
