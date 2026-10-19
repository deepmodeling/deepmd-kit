# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Matched-seed independent-process analysis of serialized normal API runs."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
root = a.directory
assert json.loads((root / "controller.json").read_text())["status"] == "completed"
rows = json.loads((root / "rounds.json").read_text())
formal = [r for r in rows if r["command"]["phase"] == "formal"]
ready = json.loads((root / "ready.json").read_text())
seeds = sorted({r["seed"] for r in formal})
pairs = []
for seed in seeds:
    stats = {}
    for arm in ("BASE", "ECS"):
        group = [r for r in formal if r["seed"] == seed and r["variant"] == arm]
        values = [v for r in group for v in r["wall_ms"]["samples"]]
        stats[arm] = {
            "p50_ms": float(np.median(values)),
            "p90_ms": float(np.percentile(values, 90)),
            "round_logmean_ms": float(
                np.exp(np.mean(np.log([r["wall_ms"]["median"] for r in group])))
            ),
            "round_medians_ms": [r["wall_ms"]["median"] for r in group],
        }
    change = float(
        np.log(stats["ECS"]["round_logmean_ms"])
        - np.log(stats["BASE"]["round_logmean_ms"])
    )
    pairs.append(
        {
            "seed": seed,
            "arms": stats,
            "logratio": change,
            "percent": float(100 * np.expm1(change)),
        }
    )
logs = np.array([p["logratio"] for p in pairs])
se = logs.std(ddof=1) / np.sqrt(len(logs)) if len(logs) > 1 else None
ci = (
    (
        100
        * np.expm1(logs.mean() + np.array([-1, 1]) * t.ppf(0.975, len(logs) - 1) * se)
    ).tolist()
    if se is not None
    else None
)
monitor = json.loads((root / "monitor.json").read_text())
telemetry = []
for r in formal:
    vals = [
        m
        for m in monitor
        if r["start_monotonic_ns"] <= m["t_ns"] <= r["end_monotonic_ns"]
    ]
    telemetry.append(
        {
            "run": r["run"],
            "seed": r["seed"],
            "arm": r["variant"],
            "block": r["command"]["block"],
            "mean_sm_mhz": float(np.mean([m["sm_mhz"] for m in vals])),
            "mem_mhz": sorted({m["mem_mhz"] for m in vals}),
            "temperature_range": [
                min(m["temperature"] for m in vals),
                max(m["temperature"] for m in vals),
            ],
            "clock_reason_masks": sorted({m["clock_reasons"] for m in vals}),
        }
    )
artifacts = []
for entry in ready:
    before = json.loads((root / f"cache_{entry['run']}_before.json").read_text())
    after = json.loads((root / f"cache_{entry['run']}_after.json").read_text())
    assert before == after
    artifacts.append({"run": entry["run"], "unchanged": True, "count": len(before)})
stats = {}
for arm in ("BASE", "ECS"):
    values = [v for r in formal if r["variant"] == arm for v in r["wall_ms"]["samples"]]
    stats[arm] = {
        "samples": len(values),
        "p50_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }
summary = {
    "scope": "normal API; one CUDA process at a time; matched Python hash seeds; see controller arguments",
    "pairs": pairs,
    "mean_change_percent": float(100 * np.expm1(logs.mean())),
    "process_pair_t_95_percent": ci,
    "stats": stats,
    "interval_telemetry": telemetry,
    "compiler_artifacts": artifacts,
    "interpretation": "finite independent process pairs; a confidence interval is not proof of zero cost",
    "cold_peak": "separate existing cold-process measurement, not replaced",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(
    json.dumps(
        {k: v for k, v in summary.items() if k != "interval_telemetry"}, indent=2
    )
)
