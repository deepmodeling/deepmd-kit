#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Safe adapter for common Bohrium job and group operations."""

# ruff: noqa: T201 -- stdout is this command-line tool's result interface.

from __future__ import (
    annotations,
)

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import (
    Path,
)


def executable(required: bool = True) -> str:
    found = shutil.which("bohr")
    fallback = Path.home() / ".bohrium" / "bohr"
    if found:
        return found
    if fallback.is_file():
        return str(fallback)
    if required:
        raise SystemExit("bohr CLI not found; install the current official Bohrium CLI")
    return "bohr"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("BOHR_ACCESS_KEY") and not env.get("ACCESS_KEY"):
        env["ACCESS_KEY"] = env["BOHR_ACCESS_KEY"]
    env.setdefault("OPENAPI_HOST", "https://open.bohrium.com")
    env.setdefault("TIEFBLUE_HOST", "https://tiefblue.dp.tech")
    return env


def run(command: list[str], execute: bool = True) -> int:
    print(
        json.dumps(
            {"command": shlex.join(command), "execute": execute}, ensure_ascii=False
        ),
        flush=True,
    )
    if not execute:
        return 0
    completed = subprocess.run(command, env=environment(), check=False)
    return completed.returncode


def mutate(command: list[str], execute: bool) -> int:
    """Print a copy-safe native dry-run unless actual execution was requested."""
    return run(command if execute else [*command, "--dry-run"], execute)


def contains_placeholder(value: object) -> bool:
    if isinstance(value, str):
        return len(value) > 4 and value.startswith("__") and value.endswith("__")
    if isinstance(value, dict):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_placeholder(item) for item in value)
    return False


def validate_submit_inputs(spec_path: Path, input_dir: Path) -> None:
    if not spec_path.is_file():
        raise SystemExit(f"job spec not found: {spec_path}")
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")
    try:
        raw = spec_path.read_text()
        spec = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"invalid job spec: {exc}") from exc
    if contains_placeholder(spec):
        raise SystemExit("job spec contains unresolved placeholders")
    required = (
        "job_name",
        "command",
        "project_id",
        "machine_type",
        "image_address",
        "job_type",
    )
    missing = [key for key in required if key not in spec or spec[key] in (None, "")]
    if missing:
        raise SystemExit(f"job spec missing required fields: {', '.join(missing)}")
    if spec["job_type"] != "container":
        raise SystemExit("job_type must be container")
    if not isinstance(spec["project_id"], int) or spec["project_id"] <= 0:
        raise SystemExit("project_id must be a positive integer")
    if "/" not in spec["image_address"] or ":" not in spec["image_address"]:
        raise SystemExit("image_address must be a full registry path with a tag")
    if "/root/input" in spec["command"]:
        raise SystemExit("command must use relative paths, not /root/input")
    if spec["command"].strip() == "bash run.sh":
        run_script = input_dir / "run.sh"
        if not run_script.is_file():
            raise SystemExit("command requires input directory run.sh")
        script_text = run_script.read_text()
        if re.search(r"__[A-Z0-9_]+__", script_text):
            raise SystemExit("run.sh contains unresolved placeholders")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("doctor")

    p = sub.add_parser("project-list")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("image-list")
    p.add_argument("--type")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("machine-list")
    p.add_argument("--kind", choices=("cpu", "gpu", "all"), default="gpu")
    p.add_argument("--scene", choices=("job", "node", "notebook"), default="job")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("node-stop")
    p.add_argument("node_id")
    p.add_argument("--execute", action="store_true")

    p = sub.add_parser("job-list")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--group-id")
    p.add_argument("--state", choices=("running", "failed", "finished", "pending"))
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("job-describe")
    p.add_argument("job_id")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("job-log")
    p.add_argument("job_id")
    p.add_argument("--output", type=Path)

    p = sub.add_parser("job-download")
    p.add_argument("job_id")
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("group-list")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("group-download")
    p.add_argument("group_id")
    p.add_argument("--output", type=Path, required=True)

    p = sub.add_parser("group-create")
    p.add_argument("--name", required=True)
    p.add_argument("--project-id", type=int, required=True)
    p.add_argument("--execute", action="store_true")

    p = sub.add_parser("job-submit")
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--name")
    p.add_argument("--group-id")
    p.add_argument("--result-path")
    submit_mode = p.add_mutually_exclusive_group()
    submit_mode.add_argument(
        "--validate", action="store_true", help="Run the CLI-native dry-run"
    )
    submit_mode.add_argument("--execute", action="store_true")

    for name in ("job-terminate", "job-kill"):
        p = sub.add_parser(name)
        p.add_argument("job_ids", nargs="+")
        p.add_argument("--execute", action="store_true")

    args = parser.parse_args()
    mutating = args.action in (
        "node-stop",
        "group-create",
        "job-submit",
        "job-terminate",
        "job-kill",
    )
    preview_only = (
        mutating and not args.execute and not getattr(args, "validate", False)
    )
    bohr = executable(required=not preview_only)
    if args.action == "doctor":
        return run([bohr, "version"])
    if args.action == "project-list":
        command = [bohr, "project", "list"]
        if args.json:
            command.extend(["--output", "json"])
        return run(command)
    if args.action == "image-list":
        command = [bohr, "image", "list"]
        if args.type:
            command.extend(["-t", args.type])
        if args.json:
            command.extend(["--output", "json"])
        return run(command)
    if args.action == "machine-list":
        command = [bohr, "machine", "list", "-c", args.kind, "-s", args.scene]
        if args.json:
            command.extend(["--output", "json"])
        return run(command)
    if args.action == "node-stop":
        return mutate([bohr, "node", "stop", args.node_id], args.execute)
    if args.action == "job-list":
        command = [bohr, "job", "list", "-n", str(args.limit)]
        if args.json:
            command.extend(["--output", "json"])
        if args.group_id:
            command.extend(["-j", args.group_id])
        flags = {
            "running": "-r",
            "failed": "-f",
            "finished": "-i",
            "pending": "--pending",
        }
        if args.state:
            command.append(flags[args.state])
        return run(command)
    if args.action == "job-describe":
        command = [bohr, "job", "describe", "-i", args.job_id]
        if args.json:
            command.extend(["--output", "json"])
        return run(command)
    if args.action == "job-log":
        command = [bohr, "job", "log", "-i", args.job_id]
        if args.output:
            command.extend(["--out", str(args.output)])
        return run(command)
    if args.action == "job-download":
        return run(
            [bohr, "job", "download", "-i", args.job_id, "--out", str(args.output)]
        )
    if args.action == "group-list":
        command = [bohr, "job_group", "list", "-n", str(args.limit)]
        if args.json:
            command.extend(["--output", "json"])
        return run(command)
    if args.action == "group-download":
        return run(
            [
                bohr,
                "job_group",
                "download",
                "-j",
                args.group_id,
                "--out",
                str(args.output),
            ]
        )
    if args.action == "group-create":
        return mutate(
            [
                bohr,
                "job_group",
                "create",
                "-n",
                args.name,
                "--project_id",
                str(args.project_id),
            ],
            args.execute,
        )
    if args.action == "job-submit":
        validate_submit_inputs(args.spec, args.input)
        command = [
            bohr,
            "job",
            "submit",
            "-i",
            str(args.spec),
            "--input_directory",
            str(args.input),
        ]
        if args.name:
            command.extend(["-n", args.name])
        if args.group_id:
            command.extend(["--job_group_id", args.group_id])
        if args.result_path:
            command.extend(["-r", args.result_path])
        if args.validate:
            return run([*command, "--dry-run"], execute=True)
        return mutate(command, args.execute)
    if args.action in ("job-terminate", "job-kill"):
        verb = "terminate" if args.action.endswith("terminate") else "kill"
        return mutate([bohr, "job", verb, *args.job_ids], args.execute)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
