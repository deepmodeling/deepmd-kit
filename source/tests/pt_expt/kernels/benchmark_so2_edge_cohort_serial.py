# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""One CUDA process at a time: remove resident-context initialization bias."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pynvml as nv

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--baseline-root", type=Path, required=True)
p.add_argument("--candidate-root", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--gpu", default="0")
p.add_argument("--cpu", type=int, default=8)
p.add_argument("--monitor-cpu", type=int, default=16)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
p.add_argument("--rounds", type=int, default=8)
p.add_argument("--samples", type=int, default=50)
p.add_argument("--conditioning-samples", type=int, default=200)
a = p.parse_args()
assert a.samples >= 10 and a.rounds >= 1
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
cache = a.cache.resolve()
os.sched_setaffinity(0, {a.monitor_cpu})
nv.nvmlInit()
handle = nv.nvmlDeviceGetHandleByIndex(int(a.gpu))
cases = [
    (arm, seed)
    for i, seed in enumerate(a.seeds)
    for arm in (("BASE", "ECS") if i % 2 == 0 else ("ECS", "BASE"))
]
worker = None
stop = threading.Event()
monitor, errors, ready, records = [], [], [], []


def write(name, value):
    (out / name).write_text(json.dumps(value, indent=2))


def watch():
    while not stop.wait(0.05):
        pids = {v.pid for v in nv.nvmlDeviceGetComputeRunningProcesses(handle)}
        allowed = {worker.pid} if worker is not None else set()
        if pids - allowed:
            errors.append(f"foreign GPU0 pids: {pids - allowed}")
            if worker is not None and worker.poll() is None:
                worker.terminate()
            return
        monitor.append(
            {
                "t_ns": time.monotonic_ns(),
                "pids": sorted(pids),
                "sm_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_SM),
                "mem_mhz": nv.nvmlDeviceGetClockInfo(handle, nv.NVML_CLOCK_MEM),
                "temperature": nv.nvmlDeviceGetTemperature(
                    handle, nv.NVML_TEMPERATURE_GPU
                ),
                "power_mw": nv.nvmlDeviceGetPowerUsage(handle),
                "clock_reasons": nv.nvmlDeviceGetCurrentClocksThrottleReasons(handle),
            }
        )


def fingerprint():
    return {
        str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
        for pattern in ("*.best_config", "*.cubin", "*.ptx")
        for p in cache.rglob(pattern)
    }


def message(log):
    while True:
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"worker ended {worker.poll()}")
        log.write(line)
        log.flush()
        if line.startswith("@CONTROL@"):
            return json.loads(line[len("@CONTROL@") :])


assert not nv.nvmlDeviceGetComputeRunningProcesses(handle)
thread = threading.Thread(target=watch, daemon=True)
thread.start()
try:
    for run, (variant, seed) in enumerate(cases):
        assert not nv.nvmlDeviceGetComputeRunningProcesses(handle)
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        with (out / f"{run}_{variant}.log").open("x") as log:
            worker = subprocess.Popen(
                [
                    sys.executable,
                    str(HERE / "_so2_edge_cohort_controlled_worker.py"),
                    "--baseline-root",
                    str(a.baseline_root.resolve()),
                    "--candidate-root",
                    str(a.candidate_root.resolve()),
                    "--model",
                    str(a.model.resolve()),
                    "--reference",
                    str(a.reference.resolve()),
                    "--variant",
                    variant,
                    "--out",
                    str(out / f"{run}_{variant}"),
                    "--cache",
                    str(cache),
                    "--gpu",
                    a.gpu,
                    "--cpu",
                    str(a.cpu),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
                env=env,
            )
            worker.stdin.write(json.dumps({"pids": [worker.pid]}) + "\n")
            worker.stdin.flush()
            r = message(log)
            assert r["status"] == "ready"
            r.update(run=run, seed=seed)
            ready.append(r)
            write("ready.json", ready)
            print("READY", run, variant, seed, r["entry_wall_ms"]["median"], flush=True)
            frozen = fingerprint()
            write(f"cache_{run}_before.json", frozen)
            for block in range(a.rounds + 1):
                command = {
                    "action": "measure",
                    "pair": seed,
                    "block": block,
                    "position": run % 2,
                    "samples": a.conditioning_samples if block == 0 else a.samples,
                    "warmup": 8,
                    "cpu": a.cpu,
                    "phase": "conditioning" if block == 0 else "formal",
                }
                worker.stdin.write(json.dumps(command) + "\n")
                worker.stdin.flush()
                r = message(log)
                r.update(run=run, seed=seed, pair=seed)
                records.append(r)
                write("rounds.json", records)
                assert fingerprint() == frozen, "autotune/cache changed in timed runs"
                print("ROUND", run, variant, block, r["wall_ms"]["median"], flush=True)
            write(f"cache_{run}_after.json", fingerprint())
            worker.stdin.write('{"action":"stop"}\n')
            worker.stdin.flush()
            r = message(log)
            assert r["status"] == "completed" and r["precision_pass"]
            worker.wait(timeout=30)
            assert worker.returncode == 0
            worker = None
finally:
    if worker is not None:
        if worker.poll() is None:
            worker.terminate()
        worker.wait(timeout=30)
    stop.set()
    thread.join(timeout=5)
    write("monitor.json", monitor)
    write(
        "controller.json",
        {
            "errors": errors,
            "status": "completed"
            if len(records) == len(cases) * (a.rounds + 1) and not errors
            else "failed",
            "args": {k: str(v) for k, v in vars(a).items()},
            "protocol": "one CUDA process at a time; matched hash seeds; conditioning separated from formal rounds",
        },
    )
    nv.nvmlShutdown()
