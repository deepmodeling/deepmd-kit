"""Render three-column accuracy, repeat-band gates and first-peak performance."""

# ruff: noqa: T201, ANN001, ANN201, RUF001 -- manual measurement report
from collections import Counter
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
P = argparse.ArgumentParser()
P.add_argument("--root", type=Path, default=SCRIPT_DIR)
HERE = P.parse_args().root.resolve()
subprocess.run(
    [sys.executable, str(SCRIPT_DIR / "analyze.py"), "--root", str(HERE)], check=True
)
data = json.loads((HERE / "summary.json").read_text())
states = data["statuses"]
groups = data["comparisons"]


def select(model, n, left, right):
    return [
        g
        for g in groups
        if (g["model"], g["n"], g["left"], g["right"]) == (model, n, left, right)
    ]


def band(model, n, left, right, field):
    gs = select(model, n, left, right)
    if not gs:
        return None
    return max(g["metrics"][field]["max_abs_range"][1] for g in gs)


def strict_status(model, n, left, right):
    gs = select(model, n, left, right)
    if not gs:
        return "N/A"
    if (
        min(
            min(g["left_completed_processes"], g["right_completed_processes"])
            for g in gs
        )
        < 3
    ):
        return "INCOMPLETE"
    return "PASS" if all(g["all_passed"] for g in gs) else "FAIL"


def band_status(model, n, mode):
    ratios = []
    for field in ("energy", "forces", "virial", "stress"):
        cross = band(model, n, mode, "TORCH", field)
        own = band(model, n, "TORCH", "TORCH", field)
        if cross is None or own is None:
            return "N/A"
        ratios.append(cross / own if own else (0 if cross == 0 else float("inf")))
    return ("PASS" if max(ratios) <= 3 else "FAIL") + f" ({max(ratios):.3f}x)"


def summary_cell(model, n, mode, field):
    ss = [s for s in states if (s["model"], s["n"], s["mode"]) == (model, n, mode)]
    good = sum(s["status"] == "completed" for s in ss)
    if not good:
        return "OOM" if any(s["status"] == "oom" for s in ss) else "pending/failed"
    own = band(model, n, mode, mode, field)
    text = f"self max_abs={own:.4g}" if own is not None else "self pending"
    if mode != "TORCH":
        cross = band(model, n, mode, "TORCH", field)
        pure = band(model, n, "TORCH", "TORCH", field)
        text += (
            f"; vs Torch={cross:.4g}"
            if cross is not None
            else "; Torch ref unavailable"
        )
        if cross is not None and pure is not None:
            ratio = cross / pure if pure else (0 if cross == 0 else float("inf"))
            text += f"; band×{ratio:.3g}"
    return text + (f" ({good}/3 processes)" if good < 3 else "")


lines = [
    "# ECS current-master comparison — measured 2026-10-25, finalized 2026-10-26",
    "",
    "BASE: 2ba617bc3db67adee2b8ba00c71e9fa1c2d2bb8c, including merged residual-inplace #6058. Old668-base numbers are not substituted. Candidate is the isolated port of PR#2; its AOT branch-specialization issue remains unresolved.",
    "",
    f"Recorded {len(states)}/135 process results: {dict(Counter(s['status'] for s in states))}. Missing cells are NOT passes. Three processes ×3 frames ×3 accuracy repeats per arm; 8 extra warmups/30 wall+Event samples, separate steady replay. Same physical GPU within each case. TF32/AMP off.",
    "",
    "## Three accuracy columns",
    "",
    "Each field: worst pairwise max-absolute difference over same-frame outputs; self includes within/across-process repeats. Scientific band multiplier = worst deviation from compiled TORCH / TORCH own worst repeat max_abs, separately per field/model/size/system. <=3 is the meeting criterion, not a replacement for the fixed strict elementwise gate. Zero repeat band requires zero cross difference. Energy/virial eV, force eV/Angstrom, stress eV/Angstrom^3.",
    "",
    "| Model/system | Atoms | Output | Compiled Torch | Current master | Candidate |",
    "| --- | ---: | --- | --- | --- | --- |",
]
cases = [
    (m, n) for m in ("Neo", "NeoCu", "Nano", "Mini", "Pro") for n in (96, 512, 4096)
]
gate_lines = [
    "## Three-column PASS/FAIL overview",
    "",
    "Strict columns cover all four fields: Torch self-repeat; master self-repeat and versus Torch; candidate self-repeat, versus Torch and versus master. A missing Torch reference is N/A, never PASS. The meeting band is a separately reported empirical diagnostic, not a rigorous error bound or a waiver of strict failure.",
    "",
    "| Model/system | Atoms | Compiled Torch | Current master | Candidate |",
    "| --- | ---: | --- | --- | --- |",
]
for model, n in cases:
    torch_gate = strict_status(model, n, "TORCH", "TORCH")
    master_gate = f"strict self={strict_status(model, n, 'BASE', 'BASE')}; vs Torch={strict_status(model, n, 'BASE', 'TORCH')}; band={band_status(model, n, 'BASE')}"
    candidate_gate = f"strict self={strict_status(model, n, 'ECS', 'ECS')}; vs Torch={strict_status(model, n, 'ECS', 'TORCH')}; vs master={strict_status(model, n, 'ECS', 'BASE')}; band={band_status(model, n, 'ECS')}"
    gate_lines.append(
        f"| {model} | {n} | strict self={torch_gate} | {master_gate} | {candidate_gate} |"
    )
first_table = lines.index("## Three accuracy columns")
lines[first_table:first_table] = [*gate_lines, ""]
for model, n in cases:
    for field in ("energy", "forces", "virial", "stress"):
        cells = [
            summary_cell(model, n, mode, field) for mode in ("TORCH", "BASE", "ECS")
        ]
        lines.append(f"| {model} | {n} | {field} | " + " | ".join(cells) + " |")
lines += [
    "",
    "Strict per-element atol1e-5/rtol1e-4 and finite results, mean_abs/max_relative and within/across detail remain in ACCURACY_TABLE.md and summary.json. Pure-reference OOM prevents band acceptance. Band PASS does not silently erase strict FAIL.",
    "",
    "## Timing and memory — process medians and worst capacity",
    "",
    "Compilation excluded from timing, included in first/full memory. This independent-process protocol does not freeze all generated autotune choices: no stable isolated speedup is inferred solely from these percentages. GB=1e9B. Both allocated/reserved raw bytes are retained in result.json. Numbers are preliminary until 3 processes per arm complete.",
    "",
    "| Model/system | Atoms | Mode | Processes | Wall P50 ms | Wall P90 ms | Event P50 ms | First alloc GB | Full peak incl first GB | Steady alloc GB |",
    "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for model, n in cases:
    for mode in ("TORCH", "BASE", "ECS"):
        ss = [
            s
            for s in states
            if (s["model"], s["n"], s["mode"], s["status"])
            == (model, n, mode, "completed")
        ]
        if not ss:
            continue
        raw = [
            json.loads((HERE / "results" / s["run"] / "result.json").read_text())
            for s in ss
        ]
        vals = [
            np.median([r["steady_wall_ms"][k] for r in raw]) for k in ("median", "p90")
        ]
        vals += [np.median([r["steady_event_ms"]["median"] for r in raw])]
        vals += [
            max(r[k]["allocated"] for r in raw) / 1e9
            for k in ("first_peak", "complete_peak", "steady_peak")
        ]
        lines.append(
            f"| {model} | {n} | {mode} | {len(raw)}/3 | "
            + " | ".join(f"{v:.6f}" for v in vals)
            + " |"
        )
lines += [
    "",
    "## Route and acceptance limits",
    "",
    "Nano/Mini/Pro candidate is native fallback, not demonstrated ECS acceleration. Neo dispatch must be checked against generated wrappers; current single-cohort specialization must not be described as 80 runtime batches. Pro-MPtrj is not OMat24-Pro; original checkpoint unmodified, reader-only legacy metadata alias applies equally to all arms. H20/PT2/C++/training and unmeasured larger models are not accepted by this matrix.",
    "",
    "Upstream review also requires focused guard/alias tests, no new public fine-grained development switch for a merge-ready path, freeze-time policy/selection where applicable, optional-Triton CPU safety and real CI approval. Current port keeps the old default-off experimental switch pending policy resolution. Do not submit as ready-to-merge.",
]
lines += [
    "",
    "## Candidate vs current-master changes",
    "",
    "Positive percentage means reduction/improvement; negative means increase. Medians of independent process quantiles, not a controlled paired significance estimate. Fallback timing variation is not attributed to ECS.",
    "",
    "| Model/system | Atoms | Time reduction % | First-inclusive full alloc reduction % | Steady alloc reduction % | Candidate route |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
for model, n in cases:
    by_mode = {}
    for mode in ("BASE", "ECS"):
        rows = [
            json.loads((HERE / "results" / s["run"] / "result.json").read_text())
            for s in states
            if (s["model"], s["n"], s["mode"], s["status"])
            == (model, n, mode, "completed")
        ]
        if len(rows) == 3:
            by_mode[mode] = (
                float(np.median([r["steady_wall_ms"]["median"] for r in rows])),
                max(r["complete_peak"]["allocated"] for r in rows),
                max(r["steady_peak"]["allocated"] for r in rows),
            )
    if len(by_mode) != 2:
        lines.append(f"| {model} | {n} | N/A | N/A | N/A | incomplete/OOM |")
        continue
    changes = [
        100 * (b - c) / b for b, c in zip(by_mode["BASE"], by_mode["ECS"], strict=True)
    ]
    route = (
        "single-cohort specialized, not bounded runtime batching"
        if model.startswith("Neo")
        else "native fallback"
    )
    lines.append(
        f"| {model} | {n} | "
        + " | ".join(f"{v:.4f}" for v in changes)
        + f" | {route} |"
    )
(HERE / "REPORT.md").write_text("\n".join(lines) + "\n")
# Generated public evidence only; never copy advisor or private handoff notes.
public_dir = HERE / "candidate/doc/development"
if public_dir.is_dir():
    (public_dir / "ecs-current-master-validation.md").write_text(
        "\n".join(lines) + "\n"
    )
    (public_dir / "ecs-current-master-validation.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
print("REPORT updated", len(states))
