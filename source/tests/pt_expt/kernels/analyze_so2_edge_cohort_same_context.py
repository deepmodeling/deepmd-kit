# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Cluster normal same-context API contrasts by independent process."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

p = argparse.ArgumentParser()
p.add_argument("directories", type=Path, nargs="+")
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()
rng = np.random.default_rng(20261019)
processes = []
all_rows = []
for root in a.directories:
    status = json.loads((root / "same_context_status.json").read_text())
    assert status["status"] == "completed"
    rows = json.loads((root / "same_context_rounds.json").read_text())
    all_rows.extend(rows)
    effects = []
    for block in sorted({r["command"]["block"] for r in rows}):
        group = [r for r in rows if r["command"]["block"] == block]
        m = {
            arm: np.mean(
                np.log([r["wall_ms"]["median"] for r in group if r["variant"] == arm])
            )
            for arm in ("BASE", "ECS")
        }
        effects.append(float(m["ECS"] - m["BASE"]))
    draws = np.mean(rng.choice(effects, (20000, len(effects))), axis=1)
    meta = json.loads((root / "same_context_manifest.json").read_text())
    processes.append(
        {
            "run": root.name,
            "first": meta["first"],
            "blocks": len(effects),
            "logmean": float(np.mean(effects)),
            "change_percent": float(100 * np.expm1(np.mean(effects))),
            "conditional_block_ci95_percent": (
                100 * np.expm1(np.percentile(draws, [2.5, 97.5]))
            ).tolist(),
            "worst_error_ratio": max(
                v["max_ratio"]
                for case in meta["accuracy"].values()
                for v in case.values()
            ),
            "wrappers": {
                arm: entry["sha256"] for arm, entry in meta["wrappers"].items()
            },
        }
    )
logs = np.array([p["logmean"] for p in processes])
se = logs.std(ddof=1) / np.sqrt(len(logs)) if len(logs) > 1 else None
interval = (
    (
        100
        * np.expm1(logs.mean() + np.array([-1, 1]) * t.ppf(0.975, len(logs) - 1) * se)
    ).tolist()
    if se is not None
    else None
)
summary = {
    "processes": processes,
    "process_count": len(processes),
    "mean_change_percent": float(100 * np.expm1(logs.mean())),
    "process_t95_percent": interval,
    "confidence_note": "processes, not frames, are independent replicates; conditional block intervals do not establish population equivalence",
    "stats": {
        arm: {
            "samples": sum(
                len(r["wall_ms"]["samples"]) for r in all_rows if r["variant"] == arm
            ),
            "p50_ms": float(
                np.median(
                    [
                        v
                        for r in all_rows
                        if r["variant"] == arm
                        for v in r["wall_ms"]["samples"]
                    ]
                )
            ),
            "p90_ms": float(
                np.percentile(
                    [
                        v
                        for r in all_rows
                        if r["variant"] == arm
                        for v in r["wall_ms"]["samples"]
                    ],
                    90,
                )
            ),
        }
        for arm in ("BASE", "ECS")
    },
    "scope": "normal full ASE API, both implementations in one CUDA context; two calculators, actual master and exact PR definitions; no Dynamo inference frontend",
}
a.out.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
