# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Resident worker preserving the public benchmark's pre-validation entry."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
import _so2_edge_cohort_benchmark_common as common

p = argparse.ArgumentParser()
p.add_argument("--baseline-root", type=Path, required=True)
p.add_argument("--candidate-root", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument(
    "--variant", choices=("BASE", "BASE2", "NATIVE", "ECS", "ECS2"), required=True
)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--gpu", default="0")
p.add_argument("--cpu", type=int, default=8)
a = p.parse_args()
os.sched_setaffinity(0, {a.cpu})
init = json.loads(sys.stdin.readline())
allowed = set(init["pids"])
original_check = subprocess.check_output


def checked(cmd, *args, **kwargs):
    value = original_check(cmd, *args, **kwargs)
    if isinstance(cmd, list) and "--query-compute-apps=pid" in cmd:
        foreign = {int(x) for x in value.split()} - allowed
        if foreign:
            raise RuntimeError(f"foreign compute processes: {foreign}")
        return ""
    return value


common.ROOT = (
    a.baseline_root.resolve()
    if a.variant in ("BASE", "BASE2")
    else a.candidate_root.resolve()
)
original_setup = common.setup


def setup(*args, **kwargs):
    subprocess.check_output = checked
    try:
        original_setup(*args, **kwargs)
    finally:
        subprocess.check_output = original_check
    os.environ.update(
        DP_SO2_EDGE_COHORT_INFER="1" if a.variant in ("ECS", "ECS2") else "0",
        TORCHINDUCTOR_CACHE_DIR=str(a.cache / "inductor"),
        TRITON_CACHE_DIR=str(a.cache / "triton"),
    )


common.setup = setup


def emit(value):
    print("@CONTROL@" + json.dumps(value), flush=True)


def callback(ns):
    import torch
    from torch._dynamo.utils import counters
    from torch._inductor.codecache import PyCodeCache

    wrappers, kernels = [], []
    for module in PyCodeCache.modules:
        path = Path(module.__file__)
        if "def call(" in path.read_text():
            wrappers.append({"path": str(path), "sha256": common.sha(path)})
            for name, kernel in vars(module).items():
                if not hasattr(kernel, "launchers") or not hasattr(
                    kernel, "triton_meta"
                ):
                    continue
                source = ast.parse(kernel.fn.src)
                for node in ast.walk(source):
                    if isinstance(node, ast.FunctionDef):
                        node.name = "kernel"
                identity = {
                    "body": ast.dump(source, include_attributes=False),
                    "signature": {
                        k: str(v)
                        for k, v in kernel.triton_meta.get("signature", {}).items()
                    },
                    "constants": kernel.triton_meta.get("constants", {}),
                }
                launches = []
                for launch in kernel.launchers:
                    compiled = next(
                        (
                            r.kernel
                            for r in kernel.compile_results
                            if str(r.config) == str(launch.config)
                        ),
                        None,
                    )
                    binary = (
                        getattr(compiled, "asm", {}).get("cubin")
                        if compiled is not None
                        else None
                    )
                    launches.append(
                        {
                            "config": str(launch.config),
                            "cubin_sha256": hashlib.sha256(binary).hexdigest()
                            if binary is not None
                            else None,
                            "n_regs": getattr(launch, "n_regs", None),
                            "n_spills": getattr(launch, "n_spills", None),
                            "shared": getattr(launch, "shared", None),
                        }
                    )
                kernels.append(
                    {
                        "name": name,
                        "semantic_sha256": hashlib.sha256(
                            json.dumps(identity, sort_keys=True, default=str).encode()
                        ).hexdigest(),
                        "launchers": launches,
                    }
                )
    first = {
        "variant": a.variant,
        "pid": os.getpid(),
        "entry_wall_ms": common.stats(ns["wall"]),
        "entry_event_ms": common.stats(ns["gpu"]),
        "first_ms": ns["first_ms"],
        "first_peak": ns["first"],
        "wrappers": wrappers,
        "kernels": kernels,
        "source_sha256": {
            str(path.relative_to(common.ROOT)): common.sha(path)
            for path in (common.ROOT / "deepmd/pt_expt/kernels/triton/sezm").glob(
                "so2_*cohort.py"
            )
        },
        "counters": {k: dict(v) for k, v in counters.items()},
    }
    common.write(a.out / "ready.json", first)
    emit(dict(status="ready", **first))
    rounds = []
    evaluate, frames = ns["evaluate"], ns["frames"]
    for line in sys.stdin:
        command = json.loads(line)
        if command["action"] == "stop":
            break
        if command["action"] == "profile":
            from torch.profiler import ProfilerActivity, profile, record_function

            for pos in frames[:8]:
                evaluate(pos)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for i in range(5):
                    torch.cuda.synchronize()
                    with record_function(f"CONTROLLED_FRAME_{i}"):
                        evaluate(frames[8 + i])
                        torch.cuda.synchronize()
            trace = a.out / "trace.json"
            prof.export_chrome_trace(str(trace))
            emit({"status": "profiled", "variant": a.variant, "trace": str(trace)})
            continue
        if command.get("cpu") is not None:
            os.sched_setaffinity(0, {command["cpu"]})
        for i in range(command.get("warmup", 8)):
            evaluate(frames[i % len(frames)])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        pre_counters = {k: dict(v) for k, v in counters.items()}
        start_ns = time.monotonic_ns()
        wall, events = [], []
        for i in range(command.get("samples", 50)):
            torch.cuda.synchronize()
            s, e = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start = time.perf_counter()
            s.record()
            evaluate(frames[8 + i % 30])
            e.record()
            torch.cuda.synchronize()
            wall.append(1000 * (time.perf_counter() - start))
            events.append(s.elapsed_time(e))
        record = {
            "variant": a.variant,
            "pid": os.getpid(),
            "command": command,
            "start_monotonic_ns": start_ns,
            "end_monotonic_ns": time.monotonic_ns(),
            "cpu": sorted(os.sched_getaffinity(0)),
            "wall_ms": common.stats(wall),
            "event_ms": common.stats(events),
            "peak": {
                "allocated": torch.cuda.max_memory_allocated(),
                "reserved": torch.cuda.max_memory_reserved(),
            },
            "counters_before": pre_counters,
            "counters_after": {k: dict(v) for k, v in counters.items()},
        }
        rounds.append(record)
        common.write(a.out / "rounds.json", rounds)
        emit(record)


sys.argv = [
    str(HERE / "_so2_edge_cohort_benchmark_impl.py"),
    "--gpu",
    a.gpu,
    "--out",
    str(a.out),
    "--model",
    str(a.model),
    "--nrep",
    "2",
    "--reference",
    str(a.reference),
]
ns = runpy.run_path(
    str(HERE / "_so2_edge_cohort_benchmark_impl.py"),
    init_globals={"_controlled_rounds": callback},
    run_name="__main__",
)
result = ns["result"]
if not result["precision_pass"] or not result["finite"]:
    raise SystemExit("accuracy failure")
emit({"status": "completed", "variant": a.variant, "precision_pass": True})
