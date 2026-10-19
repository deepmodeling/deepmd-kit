# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Analyze ABBA blocks using process/temporal clusters, not iid frame counts."""

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
root = a.directory
records = json.loads((root / "rounds.json").read_text())
control = json.loads((root / "controller.json").read_text())
assert control["status"] == "completed", (
    "incomplete/failed trials cannot be final evidence"
)
arms = ast.literal_eval(control["args"]["variants"])
blocks = defaultdict(list)
for row in records:
    blocks[(row["pair"], row["command"]["block"])].append(row)
effects, absolute, by_pair = [], [], defaultdict(list)
for (pair, block), rows in sorted(blocks.items()):
    assert len(rows) == 4
    per_arm = {
        arm: [r["wall_ms"]["median"] for r in rows if r["variant"] == arm]
        for arm in arms
    }
    assert all(len(v) == 2 for v in per_arm.values())
    logratio = float(
        np.mean(np.log(per_arm[arms[1]])) - np.mean(np.log(per_arm[arms[0]]))
    )
    difference = float(np.mean(per_arm[arms[1]]) - np.mean(per_arm[arms[0]]))
    effects.append(
        {
            "pair": pair,
            "block": block,
            "logratio": logratio,
            "percent": 100 * np.expm1(logratio),
            "difference_ms": difference,
        }
    )
    by_pair[pair].append(logratio)
    absolute.append(difference)
rng = np.random.default_rng(20261019)
matrix = np.asarray(list(by_pair.values()))
draws = []
for _ in range(20000):
    pair_indices = rng.integers(0, len(matrix), size=len(matrix))
    resampled = [
        rng.choice(matrix[i], size=len(matrix[i]), replace=True).mean()
        for i in pair_indices
    ]
    draws.append(np.mean(resampled))
ci = 100 * np.expm1(np.percentile(draws, [2.5, 97.5]))
pair_means = matrix.mean(axis=1)
process_t_ci = None
upper95 = None
if len(pair_means) > 1:
    se = pair_means.std(ddof=1) / np.sqrt(len(pair_means))
    radius = student_t.ppf(0.975, len(pair_means) - 1) * se
    process_t_ci = (
        100 * np.expm1([pair_means.mean() - radius, pair_means.mean() + radius])
    ).tolist()
    upper95 = float(
        100
        * np.expm1(pair_means.mean() + student_t.ppf(0.95, len(pair_means) - 1) * se)
    )
stats = {}
for arm in arms:
    rows = [r for r in records if r["variant"] == arm]
    samples = [v for r in rows for v in r["wall_ms"]["samples"]]
    stats[arm] = {
        "rounds": len(rows),
        "samples": len(samples),
        "median_of_round_medians_ms": float(
            np.median([r["wall_ms"]["median"] for r in rows])
        ),
        "pooled_p50_ms": float(np.median(samples)),
        "pooled_p90_ms": float(np.percentile(samples, 90)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "event_p50_ms": float(
            np.median([v for r in rows for v in r["event_ms"]["samples"]])
        ),
    }
monitor = json.loads((root / "monitor.json").read_text())
telemetry = []
for row in records:
    values = [
        m
        for m in monitor
        if row["start_monotonic_ns"] <= m["t_ns"] <= row["end_monotonic_ns"]
    ]
    telemetry.append(
        {
            "pair": row["pair"],
            "block": row["command"]["block"],
            "position": row["command"]["position"],
            "arm": row["variant"],
            "count": len(values),
            "sm_mhz": {
                str(k): v for k, v in Counter(m["sm_mhz"] for m in values).items()
            },
            "mem_mhz": {
                str(k): v for k, v in Counter(m["mem_mhz"] for m in values).items()
            },
            "temperature_range": [
                min(m["temperature"] for m in values),
                max(m["temperature"] for m in values),
            ]
            if values
            else None,
            "mean_power_w": float(np.mean([m["power_mw"] / 1000 for m in values]))
            if values
            else None,
        }
    )
ready = json.loads((root / "ready.json").read_text())
kernel_audits = []
for pair in by_pair:
    arm_kernels = []
    for arm in arms:
        entry = next(r for r in ready if r["pair"] == pair and r["variant"] == arm)
        grouped = defaultdict(list)
        for kernel in entry.get("kernels", []):
            grouped[kernel["semantic_sha256"]].extend(kernel["launchers"])
        arm_kernels.append(grouped)
    shared = set(arm_kernels[0]) & set(arm_kernels[1])
    different = []
    identical_binaries = 0
    for identity in shared:
        configs = [{v["config"] for v in group[identity]} for group in arm_kernels]
        if configs[0] != configs[1]:
            different.append(
                {"semantic_sha256": identity, "configs": [sorted(v) for v in configs]}
            )
        bins = [
            {v["cubin_sha256"] for v in group[identity] if v["cubin_sha256"]}
            for group in arm_kernels
        ]
        if bins[0] and bins[0] == bins[1]:
            identical_binaries += 1
    before = json.loads((root / f"cache_pair{pair}_before.json").read_text())
    after = json.loads((root / f"cache_pair{pair}_after.json").read_text())
    kernel_audits.append(
        {
            "pair": pair,
            "shared_semantic_programs": len(shared),
            "common_programs_identical_cubin": identical_binaries,
            "different_configs": different,
            "all_compiler_artifact_hashes_unchanged": before == after,
            "fingerprinted_artifacts": len(before),
        }
    )
profiles = []
for path in sorted(root.glob("pair*_*/*.json")):
    if path.name != "trace.json":
        continue
    events = json.loads(path.read_text())["traceEvents"]
    frames = [
        v
        for v in events
        if v.get("name", "").startswith("CONTROLLED_FRAME_") and v.get("ph") == "X"
    ]
    for frame in frames:
        kernels = [
            v
            for v in events
            if v.get("cat") == "kernel"
            and v.get("ph") == "X"
            and frame["ts"] <= v["ts"]
            and v["ts"] + v["dur"] <= frame["ts"] + frame["dur"]
        ]
        mixed = [v for v in kernels if v["name"].startswith("_stack_")]
        profiles.append(
            {
                "run": path.parent.name,
                "frame": frame["name"],
                "kernel_count": len(kernels),
                "kernel_sum_ms": sum(v["dur"] for v in kernels) / 1000,
                "mixing_kernel_count": len(mixed),
                "mixing_kernel_sum_ms": sum(v["dur"] for v in mixed) / 1000,
                "mixing_names": dict(Counter(v["name"] for v in mixed)),
            }
        )
summary = {
    "arms": arms,
    "process_pairs": len(by_pair),
    "blocks": len(effects),
    "paired_logmean_change_percent": float(100 * np.expm1(matrix.mean())),
    "mean_block_difference_ms": float(np.mean(absolute)),
    "hierarchical_bootstrap_95_percent": ci.tolist(),
    "process_pair_t_95_percent": process_t_ci,
    "process_pair_t_one_sided_upper95_percent": upper95,
    "t_interval_caveat": "t interval over independent process-pair means; n=2 is calibration only, not a precise population estimate",
    "bootstrap_draws": 20000,
    "bootstrap_seed": 20261019,
    "interval_method": "resample independent process pairs, then ABBA blocks within each selected pair; frame samples are not iid replicates",
    "per_pair_change_percent": [float(100 * np.expm1(row.mean())) for row in matrix],
    "interpretation": "positive interval: repeatable cost in this controlled setup"
    if ci[0] > 0
    else "negative interval: repeatable improvement in this controlled setup"
    if ci[1] < 0
    else "interval crosses zero: unresolved sign, not proof of zero cost",
    "stats": stats,
    "block_effects": effects,
    "interval_telemetry": telemetry,
    "kernel_audits": kernel_audits,
    "profiles": profiles,
    "monitor_errors": control["errors"],
    "memory_scope": "resident timing only; retain independent cold-first/full-window capacity measurements separately",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
print(
    json.dumps(
        {
            k: v
            for k, v in summary.items()
            if k not in ("block_effects", "interval_telemetry", "profiles")
        },
        indent=2,
    )
)
