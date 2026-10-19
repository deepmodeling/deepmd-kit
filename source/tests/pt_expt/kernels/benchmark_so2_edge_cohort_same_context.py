# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Normal ASE API A/B in one CUDA context, preserving the original baseline.

Experimental setup copies exactly the three reviewed kernel definitions and
registered backward into memory *after* compiling the baseline. No production
file, kernel arithmetic, or checkpoint is edited. Each model keeps its own
compiled cache. Inference bypasses Dynamo; stale module-level cache reuse is
explicitly prohibited and compiled graph provenance is checked.
"""

import argparse
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import runpy
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
import _so2_edge_cohort_benchmark_common as common

p = argparse.ArgumentParser()
p.add_argument("--baseline-root", type=Path, required=True)
p.add_argument("--candidate-root", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--reference", type=Path, required=True)
p.add_argument("--cache", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--first", choices=("BASE", "ECS"), default="BASE")
p.add_argument("--gpu", default="0")
p.add_argument("--cpu", type=int, default=8)
p.add_argument("--monitor-cpu", type=int, default=16)
p.add_argument("--blocks", type=int, default=24)
a = p.parse_args()
os.sched_setaffinity(0, {a.cpu})
PUBLIC = a.candidate_root.resolve()
common.ROOT = a.baseline_root.resolve()
original_setup = common.setup
state = {}


def install_candidate():
    from deepmd.pt_expt.kernels.triton.sezm import so2_value_path as sv

    if not state:
        state.update(
            sv=sv,
            backward=sv._mixing_stack_backward,
            kernels={
                name: getattr(sv, name)
                for name in (
                    "_stack_gemm_bwd_kernel",
                    "_stack_point_bwd_kernel",
                    "_stack_recompute_kernel",
                )
            },
        )
        source = PUBLIC / "deepmd/pt_expt/kernels/triton/sezm/so2_value_path.py"
        tree = ast.parse(source.read_text())
        names = set(state["kernels"]) | {"_mixing_stack_backward"}
        definitions = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        assert len(definitions) == 4
        scope = dict(vars(sv), _EDGE_COHORT_INFER=True)
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"),
            scope,
        )
        state["candidate_definitions"] = {name: scope[name] for name in names}
        name = "deepmd.pt_expt.kernels.triton.sezm.so2_edge_cohort"
        path = PUBLIC / "deepmd/pt_expt/kernels/triton/sezm/so2_edge_cohort.py"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    for name in state["kernels"]:
        setattr(sv, name, state["candidate_definitions"][name])
    sv._mixing_stack_op.register_autograd(
        state["candidate_definitions"]["_mixing_stack_backward"],
        setup_context=sv._mixing_stack_setup_context,
    )


def restore_baseline():
    sv = state["sv"]
    for name, kernel in state["kernels"].items():
        setattr(sv, name, kernel)
    sv._mixing_stack_op.register_autograd(
        state["backward"], setup_context=sv._mixing_stack_setup_context
    )


def setup(*args, **kwargs):
    original_setup(*args, **kwargs)
    os.environ.update(
        TORCHINDUCTOR_CACHE_DIR=str(a.cache.resolve() / "inductor"),
        TRITON_CACHE_DIR=str(a.cache.resolve() / "triton"),
    )
    if a.first == "ECS":
        install_candidate()


common.setup = setup


def callback(ns):
    import numpy as np
    import torch
    import pynvml as nv
    from deepmd.calculator import DP
    from deepmd.pt.model.model import sezm_model
    from torch._inductor.codecache import PyCodeCache

    first = ns["calc"]
    model = ns["inner"]
    assert model.compiled_core_compute_cache
    first_cache = dict(model.compiled_core_compute_cache)
    old_modules = {
        str(m.__file__)
        for m in PyCodeCache.modules
        if "def call(" in Path(m.__file__).read_text()
    }
    assert len(old_modules) == 1
    if a.first == "BASE":
        install_candidate()
    else:
        restore_baseline()
    sezm_model._SEZM_COMPILE_CACHE.clear()
    sezm_model._SEZM_TASK_BUF_ORDER.clear()
    second = DP(model=str(a.model))
    assert not second.dp.get_model().compiled_core_compute_cache
    calculators = {a.first: first, ("ECS" if a.first == "BASE" else "BASE"): second}
    base, frames = ns["base"], ns["frames"]

    def evaluate(calc, pos):
        calc.reset()
        atoms = base.copy()
        atoms.positions = pos
        atoms.calc = calc
        forces = atoms.get_forces().copy()
        return {
            "energy": np.asarray(atoms.get_potential_energy()),
            "forces": forces,
            "virial": np.asarray(calc.results["virial"]).copy(),
            "stress": atoms.get_stress().copy(),
        }

    started = time.perf_counter()
    for pos in frames[:8]:
        evaluate(second, pos)
    torch.cuda.synchronize()
    assert model.compiled_core_compute_cache == first_cache, "first graph cache changed"
    modules = {
        str(m.__file__)
        for m in PyCodeCache.modules
        if "def call(" in Path(m.__file__).read_text()
    }
    new_modules = modules - old_modules
    assert len(new_modules) == 1, (
        "second mode silently reused a graph or compiled multiple entries"
    )
    wrappers = {
        a.first: next(iter(old_modules)),
        ("ECS" if a.first == "BASE" else "BASE"): next(iter(new_modules)),
    }
    assert "so2_single_cohort" in Path(wrappers["ECS"]).read_text()
    assert "so2_single_cohort" not in Path(wrappers["BASE"]).read_text()
    with np.load(a.reference) as ref:
        accuracy = {}
        outputs = {}
        for arm, calc in calculators.items():
            values = {
                f"{i}_{k}": v
                for i, pos in enumerate(frames[:3])
                for k, v in evaluate(calc, pos).items()
            }
            outputs[arm] = values
            accuracy[arm] = ns["compare"](values, ref)
            assert all(v["passed"] and v["finite"] for v in accuracy[arm].values())
            np.savez_compressed(a.out / f"{arm}_outputs.npz", **values)
        accuracy["paired"] = ns["compare"](outputs["ECS"], outputs["BASE"])
        assert all(v["passed"] for v in accuracy["paired"].values())
    cache = a.cache.resolve()

    def artifacts():
        return {
            str(p.relative_to(cache)): hashlib.sha256(p.read_bytes()).hexdigest()
            for pattern in ("*.best_config", "*.cubin", "*.ptx")
            for p in cache.rglob(pattern)
        }

    frozen = artifacts()
    common.write(
        a.out / "same_context_manifest.json",
        {
            "first": a.first,
            "second_compile_s": time.perf_counter() - started,
            "wrappers": {
                k: {"path": v, "sha256": common.sha(Path(v))}
                for k, v in wrappers.items()
            },
            "accuracy": accuracy,
            "original_entry_wall_ms": ns["wall"],
            "seed": os.environ.get("PYTHONHASHSEED"),
            "method": "two normal ASE calculators; distinct AOT compiled caches; same process/context; exact candidate kernel/backward definitions transplanted only for tracing",
        },
    )
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(int(a.gpu))
    stop = threading.Event()
    monitor = []
    monitor_errors = []

    def watch():
        os.sched_setaffinity(0, {a.monitor_cpu})
        while not stop.wait(0.05):
            pids = {p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h)}
            if pids != {os.getpid()}:
                monitor_errors.append(f"foreign CUDA pid {pids}")
                return
            monitor.append(
                {
                    "t_ns": time.monotonic_ns(),
                    "sm_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                    "mem_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                    "temperature": nv.nvmlDeviceGetTemperature(
                        h, nv.NVML_TEMPERATURE_GPU
                    ),
                    "clock_reasons": nv.nvmlDeviceGetCurrentClocksThrottleReasons(h),
                }
            )

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    records = []
    try:
        for block in range(a.blocks):
            left, right = ("BASE", "ECS") if block % 2 == 0 else ("ECS", "BASE")
            for position, arm in enumerate((left, right, right, left)):
                calc = calculators[arm]
                for pos in frames[:8]:
                    evaluate(calc, pos)
                torch.cuda.synchronize()
                wall = []
                events = []
                start_ns = time.monotonic_ns()
                for i in range(50):
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    start = time.perf_counter()
                    s.record()
                    evaluate(calc, frames[8 + i % 30])
                    e.record()
                    torch.cuda.synchronize()
                    wall.append(1000 * (time.perf_counter() - start))
                    events.append(s.elapsed_time(e))
                records.append(
                    {
                        "pair": 0,
                        "variant": arm,
                        "command": {"block": block, "position": position},
                        "wall_ms": common.stats(wall),
                        "event_ms": common.stats(events),
                        "start_monotonic_ns": start_ns,
                        "end_monotonic_ns": time.monotonic_ns(),
                    }
                )
                common.write(a.out / "same_context_rounds.json", records)
                assert not monitor_errors, str(monitor_errors)
                print(
                    "CONTEXT_API",
                    block,
                    position,
                    arm,
                    records[-1]["wall_ms"]["median"],
                    flush=True,
                )
    finally:
        stop.set()
        thread.join(timeout=5)
        nv.nvmlShutdown()
        common.write(a.out / "same_context_monitor.json", monitor)
    assert not monitor_errors, str(monitor_errors)
    assert artifacts() == frozen
    assert model.compiled_core_compute_cache == first_cache
    common.write(
        a.out / "same_context_status.json",
        {
            "status": "completed",
            "accuracy_passed": True,
            "cache_unchanged": True,
            "first": a.first,
            "blocks": a.blocks,
            "samples_per_arm": a.blocks * 2 * 50,
        },
    )
    restore_baseline()


sys.argv = [
    str(HERE / "_so2_edge_cohort_benchmark_impl.py"),
    "--out",
    str(a.out),
    "--gpu",
    a.gpu,
    "--nrep",
    "2",
    "--model",
    str(a.model),
    "--reference",
    str(a.reference),
]
ns = runpy.run_path(
    str(HERE / "_so2_edge_cohort_benchmark_impl.py"),
    init_globals={"_controlled_rounds": callback},
    run_name="__main__",
)
assert ns["result"]["precision_pass"] and ns["result"]["finite"]
