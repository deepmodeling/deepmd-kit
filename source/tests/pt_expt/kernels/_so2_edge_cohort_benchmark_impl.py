# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Measurement body executed in an isolated module namespace by the CLI."""

import argparse
from pathlib import Path
import time
import traceback

from _so2_edge_cohort_benchmark_common import environment, setup, sha, stats, write

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--gpu", default="0")
p.add_argument("--variant", choices=("BASE", "TORCH"), default="BASE")
p.add_argument("--nrep", type=int, default=5)
p.add_argument(
    "--model",
    type=Path,
    required=True,
)
p.add_argument("--reference", type=Path)
p.add_argument("--warmup", type=int, default=8)
p.add_argument("--samples", type=int, default=30)
p.add_argument("--system", choices=("diamond", "water"), default="diamond")
a = p.parse_args()
setup(a.gpu, a.out, "0" if a.variant == "TORCH" else "2")

import numpy as np
import torch
from ase.build import bulk
from deepmd.calculator import DP
from deepmd.pt_expt.kernels.triton.sezm import so2_value_path as sv

assert a.warmup >= 3 and a.samples >= 10
manifest = environment()
manifest.update(model_sha256=sha(a.model), args={k: str(v) for k, v in vars(a).items()})
manifest["neo_stack_fp32_configs"] = sv.stack_fp32_configs(32, 3)
write(a.out / "manifest.json", manifest)

if a.system == "diamond":
    base = bulk("C", "diamond", a=3.567, cubic=True).repeat((a.nrep,) * 3)
else:
    from ase.build import molecule

    base = molecule("H2O")
    base.set_cell([4.0, 4.0, 4.0])
    base.center()
    base.pbc = True
    base = base.repeat((a.nrep,) * 3)
rng = np.random.default_rng(0)
frames = [
    base.positions + 0.02 * rng.standard_normal(base.positions.shape)
    for _ in range(max(3, a.warmup + a.samples))
]
calc = DP(model=str(a.model))
inner = calc.dp.get_model()
assert not inner.training, "Inference must not enable model AMP through training mode"
write(a.out / "model_definition.json", calc.dp.get_model_def_script())


def evaluate(pos):
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


def compare(got, ref):
    result = {}
    for key, value in got.items():
        expected = ref[key]
        assert value.shape == expected.shape
        finite = bool(np.isfinite(value).all() and np.isfinite(expected).all())
        if finite:
            d = np.abs(value - expected)
            ratio = d / (1e-5 + 1e-4 * np.abs(expected))
            result[key] = {
                "passed": bool((ratio <= 1).all()),
                "finite": True,
                "max_abs": float(d.max()),
                "mean_abs": float(d.mean()),
                "max_relative": float((d / np.maximum(np.abs(expected), 1e-8)).max()),
                "max_ratio": float(ratio.max()),
                "bitwise_equal": bool(np.array_equal(value, expected)),
            }
        else:
            result[key] = {"passed": False, "finite": False}
    return result


result = {"variant": a.variant, "n_atoms": len(base), "status": "running"}
try:
    torch.cuda.synchronize()
    before = {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
    }
    start = time.perf_counter()
    evaluate(frames[0])
    torch.cuda.synchronize()
    first_ms = 1000 * (time.perf_counter() - start)
    first = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    print("first", first_ms, first, flush=True)
    write(a.out / "first.json", {"first_ms": first_ms, "memory": first})
    for pos in frames[1 : a.warmup]:
        evaluate(pos)
    wall, gpu = [], []
    for pos in frames[a.warmup : a.warmup + a.samples]:
        torch.cuda.synchronize()
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start = time.perf_counter()
        s.record()
        evaluate(pos)
        e.record()
        torch.cuda.synchronize()
        wall.append(1000 * (time.perf_counter() - start))
        gpu.append(s.elapsed_time(e))
    if globals().get("_controlled_rounds") is not None:
        globals()["_controlled_rounds"](globals())
    complete = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    torch.cuda.reset_peak_memory_stats()
    for pos in frames[a.warmup : a.warmup + a.samples]:
        evaluate(pos)
    torch.cuda.synchronize()
    steady = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    arrays = {
        f"{i}_{k}": v
        for i, pos in enumerate(frames[:3])
        for k, v in evaluate(pos).items()
    }
    np.savez_compressed(a.out / "outputs.npz", **arrays)
    accuracy = {}
    if a.reference:
        with np.load(a.reference) as ref:
            accuracy = compare(arrays, ref)
    repeats = []
    for _ in range(3):
        repeated = {
            f"{i}_{k}": v
            for i, pos in enumerate(frames[:3])
            for k, v in evaluate(pos).items()
        }
        repeats.append(compare(repeated, arrays))
    from torch._inductor.codecache import PyCodeCache

    wrappers = [
        str(m.__file__)
        for m in PyCodeCache.modules
        if "def call(" in Path(m.__file__).read_text()
    ]
    assert wrappers, "No compiled wrapper; eager must not be reported as compiled"
    result.update(
        first_ms=first_ms,
        wall_ms=stats(wall),
        event_elapsed_ms=stats(gpu),
        event_scope="entire API call on default stream; includes host gaps, not kernel-sum",
        throughput_atoms_per_s=len(base) / (np.median(wall) / 1000),
        memory={
            "before": before,
            "first": first,
            "complete": complete,
            "steady": steady,
        },
        accuracy=accuracy,
        precision_pass=all(v["passed"] for v in accuracy.values())
        if accuracy
        else None,
        finite=all(bool(np.isfinite(v).all()) for v in arrays.values()),
        repeat_noise=repeats,
        wrappers=wrappers,
        gpu_end=environment()["gpu"],
        status="completed",
    )
except Exception:
    result["status"] = "failed"
    result["exception"] = traceback.format_exc()
    print(result["exception"], flush=True)
finally:
    write(a.out / "result.json", result)
    print(
        {
            k: v
            for k, v in result.items()
            if k not in ("repeat_noise", "accuracy", "wrappers")
        },
        flush=True,
    )
if result["status"] != "completed":
    raise SystemExit(1)
