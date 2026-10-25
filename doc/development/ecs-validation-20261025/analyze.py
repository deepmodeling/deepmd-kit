"""All-pairs scientific-accuracy ranges; never relax thresholds."""
# ruff: noqa: T201, ANN001, ANN201, RUF001 -- standalone manual experiment, not imported package code

import argparse
from itertools import combinations, product
import json
from pathlib import Path

import numpy as np

P = argparse.ArgumentParser()
P.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
a = P.parse_args()
root = a.root
all_results = []
cases = {}
for d in sorted((root / "results").glob("*")):
    if not (d / "result.json").exists():
        continue
    result = json.loads((d / "result.json").read_text())
    manifest = (
        json.loads((d / "manifest.json").read_text())
        if (d / "manifest.json").exists()
        else {}
    )
    pieces = d.name.split("_")
    model, n, mode, proc = pieces[0], pieces[1], "_".join(pieces[2:-1]), pieces[-1]
    entry = {
        "run": d.name,
        "model": model,
        "n": int(n[1:]),
        "mode": mode,
        "gpu": manifest.get("args", {}).get("gpu"),
        "process": proc,
        "status": "oom"
        if "deepmd.utils.errors.OutOfMemoryError:" in result.get("exception", "")
        else result["status"],
        "raw_status": result["status"],
        "model_sha256": manifest.get("model_sha256"),
        "routes": result.get("routes"),
        "runtime_cohort_calls": result.get("runtime_cohort_calls", []),
        "finite": result.get("finite"),
        "parameter_count": result.get("parameter_count"),
        "first_peak": result.get("first_peak"),
        "complete_peak": result.get("complete_peak"),
    }
    all_results.append(entry)
    if result["status"] != "completed":
        continue
    with np.load(d / "outputs.npz") as z:
        arrays = {k: z[k] for k in z.files}
    with np.load(d / "inputs.npz") as z:
        inputs = {k: z[k] for k in z.files}
    key = (model, int(n[1:]))
    group = cases.setdefault(key, {"inputs": inputs, "modes": {}})
    for k, v in inputs.items():
        if not np.array_equal(group["inputs"][k], v):
            raise ValueError(f"inputs differ: {d}/{k}")
    group["modes"].setdefault(mode, []).append((proc, arrays))


for model in {r["model"] for r in all_results}:
    hashes = {
        r["model_sha256"]
        for r in all_results
        if r["model"] == model and r["model_sha256"]
    }
    assert len(hashes) == 1, f"checkpoint changed inside the matrix: {model}"
for model, n in {(r["model"], r["n"]) for r in all_results}:
    devices = {
        r["gpu"]
        for r in all_results
        if (r["model"], r["n"]) == (model, n) and r["gpu"] is not None
    }
    assert len(devices) == 1, f"cross-GPU case mixing: {model}/{n}"


def error(got, ref):
    finite = bool(np.isfinite(got).all() and np.isfinite(ref).all())
    if not finite:
        return {"finite": False, "passed": False}
    diff = np.abs(got - ref)
    ratio = diff / (1e-5 + 1e-4 * np.abs(ref))
    worst = int(np.argmax(ratio))
    return {
        "finite": True,
        "passed": bool((ratio <= 1).all()),
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_relative": float((diff / np.maximum(np.abs(ref), 1e-8)).max()),
        "max_ratio": float(ratio.max()),
        "bitwise": bool(np.array_equal(got, ref)),
        "worst_ratio_element": {
            "index": [int(v) for v in np.unravel_index(worst, ratio.shape)],
            "got": float(got.flat[worst]),
            "reference": float(ref.flat[worst]),
            "abs_error": float(diff.flat[worst]),
            "tolerance": float(1e-5 + 1e-4 * abs(ref.flat[worst])),
        },
    }


groups = []
details = []
for (model, n), case in sorted(cases.items()):
    modes = case["modes"]
    for left, right in [
        ("TORCH", "TORCH"),
        ("BASE", "BASE"),
        ("ECS", "ECS"),
        ("BASE", "TORCH"),
        ("ECS", "TORCH"),
        ("ECS", "BASE"),
        ("ECS_RUNTIME", "ECS_RUNTIME"),
        ("ECS_RUNTIME", "BASE"),
        ("ECS_RUNTIME", "TORCH"),
        ("ECS_RUNTIME", "ECS"),
    ]:
        if left not in modes or right not in modes:
            continue
        for scope in (
            ["within_process", "across_process"]
            if left == right
            else ["cross_implementation"]
        ):
            raw = []
            for frame in range(3):
                la = [(p, r, arr) for p, arr in modes[left] for r in range(3)]
                ra = [(p, r, arr) for p, arr in modes[right] for r in range(3)]
                pairs = combinations(la, 2) if left == right else product(la, ra)
                for (lp, lr, lv), (rp, rr, rv) in pairs:
                    if scope == "within_process" and lp != rp:
                        continue
                    if scope == "across_process" and lp == rp:
                        continue
                    for out in ("energy", "forces", "virial", "stress"):
                        raw.append(
                            {
                                "frame": frame,
                                "left_process": lp,
                                "left_repeat": lr,
                                "right_process": rp,
                                "right_repeat": rr,
                                "output": out,
                                **error(
                                    lv[f"{lr}_{frame}_{out}"], rv[f"{rr}_{frame}_{out}"]
                                ),
                            }
                        )
            if not raw:
                continue
            record = {
                "model": model,
                "n": n,
                "left": left,
                "right": right,
                "scope": scope,
                "left_completed_processes": len(modes[left]),
                "right_completed_processes": len(modes[right]),
                "all_passed": all(v["passed"] for v in raw),
                "metrics": {},
            }
            for out in ("energy", "forces", "virial", "stress"):
                rows = [r for r in raw if r["output"] == out]
                finite = [r for r in rows if r["finite"]]
                record["metrics"][out] = {
                    "comparisons": len(rows),
                    "failures": sum(not r["passed"] for r in rows),
                    "all_finite": len(finite) == len(rows),
                    "all_bitwise": all(r.get("bitwise", False) for r in rows),
                    "max_abs_range": [
                        min(r["max_abs"] for r in finite),
                        max(r["max_abs"] for r in finite),
                    ]
                    if finite
                    else None,
                    "worst_mean_abs": max(
                        (r["mean_abs"] for r in finite), default=None
                    ),
                    "worst_max_relative": max(
                        (r["max_relative"] for r in finite), default=None
                    ),
                    "worst_tolerance_ratio": max(
                        (r["max_ratio"] for r in finite), default=None
                    ),
                }
            groups.append(record)
            details.append(
                {
                    "model": model,
                    "n": n,
                    "left": left,
                    "right": right,
                    "scope": scope,
                    "pairs": raw,
                }
            )
summary = {
    "threshold": {"atol": 1e-5, "rtol": 1e-4, "relative_floor": 1e-8},
    "range_definition": "min/max of max_abs over same-frame comparisons; NOT a confidence interval",
    "statuses": all_results,
    "comparisons": groups,
}
(root / "summary.json").write_text(json.dumps(summary, indent=2))
(root / "pair_details.json").write_text(json.dumps(details, indent=2))
lines = [
    "# ECS 多模型多尺寸精度测试",
    "",
    "> 精度优先；未放宽阈值。误差范围为每对输出max_abs的min～max，不是置信区间。基线SHA与模型SHA见JSON。",
    "",
    "模式：TORCH=纯PyTorch编译；ECS=当前PR候选（非Neo为回退）；ECS_RUNTIME=强制真实8192分批。单位：能量/位力eV，力eV/Å，stress eV/Å³。所有结果均不是DFT预测误差。",
    "",
    "| 模型 | 原子数 | 对照 | 范围 | 完成进程数 | 能量max_abs范围 | 力max_abs范围 | 位力max_abs范围 | stress max_abs范围 | 最差误差/容差 | 状态 |",
    "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
]


def fmt(r):
    return "～".join(f"{v:.3e}" for v in r) if r else "nonfinite"


for g in groups:
    m = g["metrics"]
    worst = max(v["worst_tolerance_ratio"] or 0 for v in m.values())
    lines.append(
        f"| {g['model']} | {g['n']} | {g['left']} vs {g['right']} | {g['scope']} | {g['left_completed_processes']}/{g['right_completed_processes']} | {fmt(m['energy']['max_abs_range'])} | {fmt(m['forces']['max_abs_range'])} | {fmt(m['virial']['max_abs_range'])} | {fmt(m['stress']['max_abs_range'])} | {worst:.4f} | {'PASS' if g['all_passed'] else 'FAIL'}{'（样本未齐）' if min(g['left_completed_processes'], g['right_completed_processes']) < 3 else ''} |"
    )
lines += [
    "",
    "## 所有进程状态（含OOM/失败，不能略去）",
    "",
    "| 运行 | 状态 | 参数量 | opaque ECS调用 | single路径标记 |",
    "| --- | --- | ---: | ---: | --- |",
]
for r in all_results:
    routes = r["routes"] or []
    lines.append(
        f"| {r['run']} | {r['status']} | {r['parameter_count']} | {sum(x['opaque_cohort_calls'] for x in routes)} | {any(x['single_cohort_marker'] for x in routes)} |"
    )
(root / "ACCURACY_TABLE.md").write_text("\n".join(lines) + "\n")
print(
    "runs",
    len(all_results),
    "completed",
    sum(r["status"] == "completed" for r in all_results),
    "comparisons",
    len(groups),
    "failed_groups",
    sum(not g["all_passed"] for g in groups),
)
