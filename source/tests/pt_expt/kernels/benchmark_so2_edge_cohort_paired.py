# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Sequential resident-worker ABBA trials with fixed caches and interval NVML."""

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
p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--baseline-root", type=Path, required=True)
p.add_argument("--candidate-root", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--variants", nargs=2, default=["BASE", "ECS"])
p.add_argument("--pairs", type=int, default=3)
p.add_argument("--blocks", type=int, default=12)
p.add_argument("--samples", type=int, default=50)
p.add_argument("--cpu", type=int, default=8)
p.add_argument("--gpu", default="0")
p.add_argument("--monitor-cpu", type=int, default=16)
p.add_argument("--profile", action="store_true")
a = p.parse_args()
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
a.cache = a.cache.resolve()
a.cache.mkdir(parents=True, exist_ok=True)
os.sched_setaffinity(0, {a.monitor_cpu})
nv.nvmlInit()
h = nv.nvmlDeviceGetHandleByIndex(int(a.gpu))
assert not nv.nvmlDeviceGetComputeRunningProcesses(h), f"GPU{a.gpu} not idle"
workers = {}
stop = threading.Event()
monitor, errors, records, ready = [], [], [], []


def watch():
    while not stop.wait(0.05):
        try:
            pids = {v.pid for v in nv.nvmlDeviceGetComputeRunningProcesses(h)}
            foreign = pids - {w.pid for w in workers.values()}
            if foreign:
                errors.append(f"foreign compute pids: {sorted(foreign)}")
                for w in list(workers.values()):
                    if w.poll() is None:
                        w.terminate()
                return
            monitor.append(
                {
                    "t_ns": time.monotonic_ns(),
                    "pids": sorted(pids),
                    "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                    "temperature": nv.nvmlDeviceGetTemperature(
                        h, nv.NVML_TEMPERATURE_GPU
                    ),
                    "power_mw": nv.nvmlDeviceGetPowerUsage(h),
                    "utilization": nv.nvmlDeviceGetUtilizationRates(h).gpu,
                    "pstate": nv.nvmlDeviceGetPerformanceState(h),
                }
            )
        except Exception as exc:
            errors.append(repr(exc))
            return


def write(name, value):
    (out / name).write_text(json.dumps(value, indent=2))


def fingerprint(include_binaries=False):
    patterns = ["*.best_config"] + (["*.cubin", "*.ptx"] if include_binaries else [])
    return {
        str(path.relative_to(a.cache)): hashlib.sha256(path.read_bytes()).hexdigest()
        for pattern in patterns
        for path in sorted(a.cache.rglob(pattern))
    }


def message(worker, log):
    while True:
        line = worker.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited: {worker.poll()} {errors}")
        log.write(line)
        log.flush()
        if line.startswith("@CONTROL@"):
            return json.loads(line[len("@CONTROL@") :])


thread = threading.Thread(target=watch, daemon=True)
thread.start()
logs = []
try:
    for pair in range(a.pairs):
        pair_ready = []
        init_order = a.variants if pair % 2 == 0 else list(reversed(a.variants))
        for variant in init_order:
            log = (out / f"pair{pair}_{variant}.log").open("x")
            logs.append(log)
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
                    str(out / f"pair{pair}_{variant}"),
                    "--cache",
                    str(a.cache),
                    "--cpu",
                    str(a.cpu),
                    "--gpu",
                    a.gpu,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                bufsize=1,
            )
            workers[variant] = worker
            worker.stdin.write(
                json.dumps({"pids": [w.pid for w in workers.values()]}) + "\n"
            )
            worker.stdin.flush()
            r = message(worker, log)
            assert r["status"] == "ready"
            r["pair"] = pair
            ready.append(r)
            pair_ready.append(r)
            write("ready.json", ready)
            print("READY", pair, variant, r["entry_wall_ms"]["median"], flush=True)
        frozen = fingerprint()
        binary_before = fingerprint(True)
        write(f"cache_pair{pair}_before.json", binary_before)
        for block in range(a.blocks):
            left, right = a.variants if block % 2 == 0 else list(reversed(a.variants))
            for position, variant in enumerate((left, right, right, left)):
                worker = workers[variant]
                command = {
                    "action": "measure",
                    "pair": pair,
                    "block": block,
                    "position": position,
                    "warmup": 8,
                    "samples": a.samples,
                    "cpu": a.cpu,
                }
                assert fingerprint() == frozen, "autotune cache changed before timing"
                worker.stdin.write(json.dumps(command) + "\n")
                worker.stdin.flush()
                log = logs[-2 + init_order.index(variant)]
                r = message(worker, log)
                assert fingerprint() == frozen, "autotune cache changed during timing"
                r["pair"] = pair
                records.append(r)
                write("rounds.json", records)
                print(
                    "ROUND",
                    pair,
                    block,
                    position,
                    variant,
                    r["wall_ms"]["median"],
                    flush=True,
                )
        if a.profile:
            for variant in init_order:
                worker = workers[variant]
                worker.stdin.write('{"action":"profile"}\n')
                worker.stdin.flush()
                profiled = message(worker, logs[-2 + init_order.index(variant)])
                assert profiled["status"] == "profiled"
                write(f"profile_pair{pair}_{variant}.json", profiled)
        after = fingerprint(True)
        write(f"cache_pair{pair}_after.json", after)
        assert after == binary_before, (
            "compiled binaries changed during measured blocks"
        )
        for variant in init_order:
            worker = workers[variant]
            worker.stdin.write('{"action":"stop"}\n')
            worker.stdin.flush()
            r = message(worker, logs[-2 + init_order.index(variant)])
            assert r["status"] == "completed" and r["precision_pass"]
            worker.wait(timeout=30)
            assert worker.returncode == 0
        workers.clear()
finally:
    for worker in workers.values():
        if worker.poll() is None:
            worker.terminate()
        worker.wait(timeout=30)
    stop.set()
    thread.join(timeout=5)
    write("monitor.json", monitor)
    write(
        "controller.json",
        {
            "args": {k: str(v) for k, v in vars(a).items()},
            "errors": errors,
            "status": "completed"
            if not errors and len(records) == a.pairs * a.blocks * 4
            else "failed",
        },
    )
    for log in logs:
        log.close()
    nv.nvmlShutdown()
