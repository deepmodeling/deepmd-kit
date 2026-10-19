# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Supplement only: match observed clock/temperature bins, never replace raw trials."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("directories", type=Path, nargs="+")
p.add_argument("--out", type=Path, required=True)
p.add_argument("--memory-clock", type=int, default=13801)
a = p.parse_args()
result = []
for root in a.directories:
    rows = json.loads((root / "same_context_rounds.json").read_text())
    monitor = json.loads((root / "same_context_monitor.json").read_text())
    kept = []
    for block in sorted({r["command"]["block"] for r in rows}):
        group = [r for r in rows if r["command"]["block"] == block]
        clocks = {}
        temperatures = {}
        memory = {}
        effects = {}
        for arm in ("BASE", "ECS"):
            rounds = [r for r in group if r["variant"] == arm]
            samples = [
                m
                for r in rounds
                for m in monitor
                if r["start_monotonic_ns"] <= m["t_ns"] <= r["end_monotonic_ns"]
            ]
            clocks[arm] = np.mean([m["sm_mhz"] for m in samples])
            temperatures[arm] = np.mean([m["temperature"] for m in samples])
            memory[arm] = {m["mem_mhz"] for m in samples}
            effects[arm] = np.mean(np.log([r["wall_ms"]["median"] for r in rounds]))
        if (
            abs(clocks["ECS"] / clocks["BASE"] - 1) <= 0.005
            and memory["BASE"] == memory["ECS"] == {a.memory_clock}
            and abs(temperatures["ECS"] - temperatures["BASE"]) <= 2
        ):
            kept.append(
                {
                    "block": block,
                    "logratio": float(effects["ECS"] - effects["BASE"]),
                    "sm_mhz": {k: float(v) for k, v in clocks.items()},
                }
            )
    result.append(
        {
            "run": root.name,
            "retained_blocks": len(kept),
            "blocks": kept,
            "mean_logratio": float(np.mean([r["logratio"] for r in kept]))
            if kept
            else None,
        }
    )
logs = np.array([r["mean_logratio"] for r in result if r["mean_logratio"] is not None])
radius = t.ppf(0.975, len(logs) - 1) * logs.std(ddof=1) / np.sqrt(len(logs))
summary = {
    "criteria": f"block mean SM clock difference <=0.5%, both memory clocks {a.memory_clock}MHz, mean temperature difference <=2C",
    "primary": False,
    "caveat": "post-stratified sensitivity analysis; conditioning can change composition and does not prove constant clocks or replace full results",
    "processes": result,
    "mean_change_percent": float(100 * np.expm1(logs.mean())),
    "process_t95_percent": (
        100 * np.expm1(logs.mean() + np.array([-1, 1]) * radius)
    ).tolist(),
}
a.out.write_text(json.dumps(summary, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "processes"}, indent=2))
print([(r["run"], r["retained_blocks"]) for r in result])
