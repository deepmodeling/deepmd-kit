#!/usr/bin/env python3
"""Run a non-destructive first-pass audit of DPA4 workflow case folders."""

# ruff: noqa: T201 -- stdout is this command-line tool's result interface.

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


FATAL_PATTERNS = {
    "unknown_dpa4": re.compile(r"Unknown model type:\s*dpa4", re.I),
    "missing_mass": re.compile(
        r"Not all per-type masses are set|Type\s+\d+\s+is missing", re.I
    ),
    "cuda_error": re.compile(r"CUDA Runtime|CUDA error|operation not supported", re.I),
    "oom": re.compile(r"out of memory|\bOOM\b|std::bad_alloc", re.I),
    "lost_atoms": re.compile(r"Lost atoms|Bond atoms missing|Out of range atoms", re.I),
    "non_numeric": re.compile(
        r"Non-numeric (?:atom coords|box dimensions|pressure|energy)|\bnan\b|\binf\b",
        re.I,
    ),
    "segfault": re.compile(r"Segmentation fault|SIGSEGV", re.I),
    "lammps_error": re.compile(r"^ERROR(?: on proc \d+)?:", re.I | re.M),
    "python_exception": re.compile(
        r"Traceback \(most recent call last\):|FileNotFoundError|RuntimeError:", re.I
    ),
    "distributed_error": re.compile(
        r"NCCL (?:error|failure)|ProcessGroupNCCL|ChildFailedError", re.I
    ),
}

COMPLETE_PATTERNS = [
    re.compile(r"Total wall time:", re.I),
    re.compile(r"Loop time of", re.I),
    re.compile(r"Minimization stats:", re.I),
]

ENERGY_RE = re.compile(
    r"(?:PotEng|\bpe\b|potential[_ ]?energy|final[_ ]?energy)\s*[=:, ]+"
    r"([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
    re.I,
)


def read_text(path: Path, limit: int = 50_000_000) -> str:
    try:
        with path.open("r", errors="replace") as handle:
            return handle.read(limit)
    except OSError:
        return ""


def find_case_dirs(root: Path) -> list[Path]:
    markers = (
        "in.lammps",
        "log.lammps",
        "run.log",
        "train.log",
        "lcurve.out",
        "input.json",
        "structure.data",
        "input.data",
    )
    cases: set[Path] = set()
    for marker in markers:
        for path in root.rglob(marker):
            if path.is_file():
                cases.add(path.parent)
    return sorted(cases)


def infer_mode(case: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    inputs = [path for path in (case / "in.lammps", case / "in.lmp") if path.is_file()]
    text = "\n".join(read_text(path) for path in inputs)
    has_run = bool(re.search(r"^\s*run\s+\S+", text, re.M))
    has_minimize = bool(re.search(r"^\s*minimize\s+", text, re.M))
    if has_run:
        return "md"
    if has_minimize:
        return "minimize"
    training_input = case / "input.json"
    if training_input.is_file() and re.search(
        r'"training"\s*:', read_text(training_input)
    ):
        return "train"
    return "unknown"


def extract_energy(case: Path, log_text: str) -> float | None:
    for name in ("energy.dat", "final_energy.dat"):
        path = case / name
        if path.is_file():
            values = re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", read_text(path))
            if values:
                value = float(values[-1])
                return value if math.isfinite(value) else None
    matches = ENERGY_RE.findall(log_text)
    if matches:
        value = float(matches[-1])
        return value if math.isfinite(value) else None
    return None


def match_required(case: Path, patterns: list[str]) -> dict[str, list[str]]:
    matched: dict[str, list[str]] = {}
    for pattern in patterns:
        paths = sorted(
            str(path.relative_to(case)) for path in case.glob(pattern) if path.exists()
        )
        matched[pattern] = paths
    return matched


def audit_case(case: Path, root: Path, mode: str, required: list[str]) -> dict:
    logs = [
        path
        for path in (
            case / "log.lammps",
            case / "run.log",
            case / "train.log",
            case / "stderr.log",
            case / "log",
        )
        if path.is_file()
    ]
    log_text = "\n".join(read_text(path) for path in logs)
    fatal = [
        name for name, pattern in FATAL_PATTERNS.items() if pattern.search(log_text)
    ]
    inferred_mode = infer_mode(case, mode)
    complete = any(pattern.search(log_text) for pattern in COMPLETE_PATTERNS)
    required_matches = match_required(case, required)
    missing_required = [
        pattern for pattern, matches in required_matches.items() if not matches
    ]
    reasons: list[str] = []
    if not logs:
        reasons.append("missing_log")
    if fatal:
        reasons.extend(fatal)
    if logs and inferred_mode in ("md", "minimize") and not complete:
        reasons.append("missing_completion_marker")
    reasons.extend(f"missing_required:{pattern}" for pattern in missing_required)
    verdict = "passes_file_audit" if not reasons else "needs_review"
    artifacts = {
        "freeze_log": str(case / "freeze.log")
        if (case / "freeze.log").is_file()
        else None,
        "final_structure": next(
            (
                str(case / name)
                for name in ("final.data", "relaxed.data")
                if (case / name).is_file()
            ),
            None,
        ),
        "restart_files": sorted(str(path) for path in case.glob("restart*")),
        "trajectory_candidates": sorted(
            str(path)
            for pattern in ("*.lammpstrj", "dump*", "trajectory/*")
            for path in case.glob(pattern)
            if path.is_file()
        ),
        "checkpoint_candidates": sorted(
            str(path)
            for pattern in ("*.pt", "checkpoint*", "ckpt/*")
            for path in case.glob(pattern)
            if path.is_file()
        ),
        "metric_candidates": sorted(
            str(path)
            for pattern in ("lcurve.out", "metrics*", "results/*")
            for path in case.glob(pattern)
            if path.is_file()
        ),
    }
    return {
        "case_id": str(case.relative_to(root)) or ".",
        "case_dir": str(case),
        "mode": inferred_mode,
        "verdict": verdict,
        "reasons": reasons,
        "completion_marker": complete,
        "fatal_signatures": fatal,
        "last_detected_energy": extract_energy(case, log_text),
        "required_matches": required_matches,
        "artifacts": artifacts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="Root directory containing case directories"
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "train", "inference", "minimize", "md"),
        default="auto",
    )
    parser.add_argument(
        "--require",
        action="append",
        default=[],
        metavar="GLOB",
        help="Require a case-relative file glob; repeat for multiple artifacts",
    )
    parser.add_argument(
        "--json", type=Path, help="Write the full JSON report to this path"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    cases = [
        audit_case(case, root, args.mode, args.require) for case in find_case_dirs(root)
    ]
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["verdict"]] = counts.get(case["verdict"], 0) + 1
    report = {
        "root": str(root),
        "case_count": len(cases),
        "mode_request": args.mode,
        "required_patterns": args.require,
        "verdict_counts": counts,
        "warning": "A file/log audit pass is not training, inference, MD-quality, or scientific acceptance.",
        "cases": cases,
    }
    if args.json:
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("root", "case_count", "verdict_counts", "warning")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for case in cases:
        if case["verdict"] != "passes_file_audit":
            print(f"REVIEW\t{case['case_id']}\t{','.join(case['reasons'])}")
    return 0 if cases and counts.get("needs_review", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
