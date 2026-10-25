"""Accuracy-first full API matrix. No production source mutation."""
# ruff: noqa: T201, TID253, ANN001, ANN002, ANN003, ANN201 -- standalone manual experiment, not imported package code

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

P = argparse.ArgumentParser()
P.add_argument("--root", type=Path, required=True)
P.add_argument("--model", type=Path, required=True)
P.add_argument("--out", type=Path, required=True)
P.add_argument("--cache", type=Path, required=True)
P.add_argument("--gpu", type=int, required=True)
P.add_argument(
    "--mode",
    choices=["TORCH", "BASE", "ECS", "ECS_RUNTIME", "TORCH_EAGER"],
    required=True,
)
P.add_argument("--nrep", type=int, required=True)
P.add_argument("--seed", type=int, default=0)
P.add_argument("--repeat-cell", type=int, nargs=3)
P.add_argument("--system", choices=["diamond", "bcccu"], default="diamond")
P.add_argument("--samples", type=int, default=30)
P.add_argument("--warmup", type=int, default=8)
a = P.parse_args()
a.out.mkdir(parents=True, exist_ok=False)


def write(name, obj):
    (a.out / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    )


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


busy = subprocess.check_output(
    [
        "nvidia-smi",
        "-i",
        str(a.gpu),
        "--query-compute-apps=pid",
        "--format=csv,noheader",
    ],
    text=True,
).strip()
if busy:
    write("result.json", {"status": "blocked_busy", "pids": busy})
    raise SystemExit(3)
for k in list(os.environ):
    if k.startswith("DP_"):
        del os.environ[k]
os.environ.update(
    CUDA_VISIBLE_DEVICES=str(a.gpu),
    DP_COMPILE_INFER="0" if a.mode == "TORCH_EAGER" else "1",
    DP_COMPILE="0",
    DP_TRITON_INFER="0" if a.mode.startswith("TORCH") else "2",
    DP_CUDA_INFER="0",
    DP_TF32_INFER="0",
    DP_AMP_INFER="0",
    DP_SO2_EDGE_COHORT_INFER="1" if a.mode.startswith("ECS") else "0",
    OMP_NUM_THREADS="1",
    DP_INTRA_OP_PARALLELISM_THREADS="1",
    DP_INTER_OP_PARALLELISM_THREADS="1",
    PYTORCH_ALLOC_CONF="expandable_segments:True",
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
    TORCHINDUCTOR_COMPILE_THREADS="1",
    TORCHINDUCTOR_CACHE_DIR=str(a.cache / "inductor"),
    TRITON_CACHE_DIR=str(a.cache / "triton"),
)
sys.path.insert(0, str(a.root))
sys.meta_path[:] = [
    f
    for f in sys.meta_path
    if not type(f).__module__.startswith("_editable_skbc_deepmd_kit")
]
import site
import deepmd

for installed in site.getsitepackages():
    p = Path(installed) / "deepmd"
    if (p / "lib").is_dir():
        deepmd.__path__.append(str(p))
assert Path(deepmd.__file__).resolve().parent.parent == a.root.resolve()
import numpy as np
import torch
import triton

torch.set_num_threads(1)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.reset_peak_memory_stats()
manifest = {
    "args": {k: str(v) for k, v in vars(a).items()},
    "model_sha256": sha(a.model),
    "torch": torch.__version__,
    "triton": triton.__version__,
    "cuda": torch.version.cuda,
    "baseline_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=a.root, text=True
    ).strip(),
    "source_hashes": {
        str(p.relative_to(a.root)): sha(p)
        for p in [
            a.root / "deepmd/pt_expt/kernels/triton/sezm/so2_value_path.py",
            a.root / "deepmd/pt_expt/kernels/triton/sezm/so2_edge_cohort.py",
        ]
        if p.exists()
    },
    "flags": {
        k: v
        for k, v in os.environ.items()
        if k.startswith(
            ("DP_", "TORCHINDUCTOR", "TRITON_CACHE", "PYTORCH_ALLOC", "PYTHONHASH")
        )
    },
}
write("manifest.json", manifest)
result = {
    "status": "running",
    "mode": a.mode,
    "n_atoms": (8 if a.system == "diamond" else 2)
    * int(np.prod(a.repeat_cell if a.repeat_cell else [a.nrep] * 3)),
    "started": time.time(),
    "threshold": {"atol": 1e-5, "rtol": 1e-4, "relative_denominator_floor": 1e-8},
}
arrays = {}
runtime_calls = []
try:
    from ase.build import bulk
    from deepmd.calculator import DP

    repeat_cell = tuple(a.repeat_cell) if a.repeat_cell else (a.nrep,) * 3
    atoms = (
        bulk("C", "diamond", a=3.567, cubic=True)
        if a.system == "diamond"
        else bulk("Cu", "bcc", a=2.87, cubic=True)
    ).repeat(repeat_cell)
    result["n_atoms"] = len(atoms)
    result["system"] = a.system
    rng = np.random.default_rng(a.seed)
    frames = np.stack(
        [
            atoms.positions + 0.02 * rng.standard_normal(atoms.positions.shape)
            for _ in range(3)
        ]
    )
    np.savez_compressed(
        a.out / "inputs.npz",
        positions=frames,
        cell=np.asarray(atoms.cell),
        numbers=atoms.numbers,
    )
    result["input_sha256"] = sha(a.out / "inputs.npz")
    if a.mode == "ECS_RUNTIME":
        from deepmd.pt_expt.kernels.triton.sezm import so2_edge_cohort as ec

        ec.SINGLE_COHORT_MAX_EDGES = 0
        original_runtime = ec.cohort_input_backward_impl

        def traced_runtime(*args, **kwargs):
            edges = args[0].shape[0]
            cap = args[7] if len(args) > 7 else kwargs["cohort_edges"]
            runtime_calls.append(
                {"edges": edges, "cohort": cap, "chunks": (edges + cap - 1) // cap}
            )
            return original_runtime(*args, **kwargs)

        ec.cohort_input_backward_impl = traced_runtime
    original_load = torch.load
    if (
        manifest["model_sha256"]
        == "446a6e893101fc52190cfccabb9d2328863d651aca36981fd8d3401c71a61221"
    ):
        assert (
            manifest["model_sha256"]
            == "446a6e893101fc52190cfccabb9d2328863d651aca36981fd8d3401c71a61221"
        )
        proof = json.loads((a.model.parent / "cpu_load_check.json").read_text())
        assert proof["strict_missing"] == proof["strict_unexpected"] == []

        def legacy_load(file, *args, **kwargs):
            obj = original_load(file, *args, **kwargs)
            if (
                isinstance(file, (str, Path))
                and Path(file).resolve() == a.model.resolve()
            ):
                state = obj["model"] if "model" in obj else obj
                state["_extra_state"] = copy.deepcopy(state["_extra_state"])
                fit = state["_extra_state"]["model_params"]["fitting_net"]
                assert fit["type"] == "ener"
                fit["type"] = "dpa4_ener"
            return obj

        torch.load = legacy_load
        result["legacy_compatibility"] = (
            "Pro-only in-memory fitting alias ener -> dpa4_ener; strict CPU state check passed; original checkpoint unchanged"
        )
    try:
        calc = DP(model=str(a.model))
    finally:
        torch.load = original_load
    inner = calc.dp.get_model()
    assert not inner.training
    definition = calc.dp.get_model_def_script()
    write("model_definition.json", definition)
    result["parameter_count"] = sum(p.numel() for p in inner.parameters())

    def evaluate(pos):
        calc.reset()
        sample = atoms.copy()
        sample.positions = pos
        sample.calc = calc
        forces = sample.get_forces().copy()
        return {
            "energy": np.asarray(sample.get_potential_energy()),
            "forces": forces,
            "virial": np.asarray(calc.results["virial"]).copy(),
            "stress": sample.get_stress().copy(),
        }

    wall = []
    events = []
    for repeat in range(3):
        for frame, pos in enumerate(frames):
            torch.cuda.synchronize()
            s, e = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start = time.perf_counter()
            s.record()
            out = evaluate(pos)
            e.record()
            torch.cuda.synchronize()
            elapsed = 1000 * (time.perf_counter() - start)
            wall.append(elapsed)
            events.append(s.elapsed_time(e))
            for k, v in out.items():
                arrays[f"{repeat}_{frame}_{k}"] = v
            np.savez_compressed(a.out / "outputs.npz", **arrays)
            if repeat == 0 and frame == 0:
                result["first_peak"] = {
                    "allocated": torch.cuda.max_memory_allocated(),
                    "reserved": torch.cuda.max_memory_reserved(),
                }
            write(
                "progress.json",
                {
                    "repeat": repeat,
                    "frame": frame,
                    "elapsed_ms": elapsed,
                    "finite": all(bool(np.isfinite(v).all()) for v in out.values()),
                },
            )
            print(a.mode, len(atoms), repeat, frame, elapsed, flush=True)
    # Accuracy collection is separate from warmed timing and steady memory.
    assert a.warmup >= 3 and a.samples >= 10
    for i in range(a.warmup):
        evaluate(frames[i % len(frames)])
    timed_wall, timed_event = [], []
    for i in range(a.samples):
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        tick = time.perf_counter()
        begin.record()
        measured = evaluate(frames[i % len(frames)])
        end.record()
        torch.cuda.synchronize()
        assert all(bool(np.isfinite(v).all()) for v in measured.values())
        timed_wall.append(1000 * (time.perf_counter() - tick))
        timed_event.append(begin.elapsed_time(end))
    full_peak = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    torch.cuda.reset_peak_memory_stats()
    for i in range(a.samples):
        evaluate(frames[i % len(frames)])
    torch.cuda.synchronize()
    result["steady_peak"] = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
    result["full_window_peak"] = full_peak
    result["steady_wall_ms"] = {
        "median": float(np.median(timed_wall)),
        "p90": float(np.percentile(timed_wall, 90)),
        "min": min(timed_wall),
        "max": max(timed_wall),
        "samples": timed_wall,
    }
    result["steady_event_ms"] = {
        "median": float(np.median(timed_event)),
        "p90": float(np.percentile(timed_event, 90)),
        "samples": timed_event,
    }
    from torch._inductor.codecache import PyCodeCache

    wrappers = []
    route = []
    for m in PyCodeCache.modules:
        path = Path(m.__file__)
        text = path.read_text()
        if "def call(" not in text:
            continue
        wrappers.append(str(path))
        route.append(
            {
                "sha256": sha(path),
                "opaque_cohort_calls": text.count("so2_cohort_input_bwd.default("),
                "single_cohort_named_calls": text.count("so2_single_cohort_input_bwd"),
                "stack_kernel_definitions": text.count("def _stack_"),
                "single_cohort_marker": "so2_single_cohort_input_bwd" in text,
            }
        )
    if a.mode != "TORCH_EAGER":
        assert wrappers, "compiled path missing"
    assert torch.get_float32_matmul_precision() == "highest"
    assert not torch.backends.cuda.matmul.allow_tf32
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.is_autocast_enabled("cuda")
    result.update(
        status="completed",
        finite=all(bool(np.isfinite(v).all()) for v in arrays.values()),
        evaluations=9,
        runtime_cohort_calls=runtime_calls,
        precision_runtime={
            "matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "outer_autocast": torch.is_autocast_enabled("cuda"),
        },
        wall_ms_including_first=wall,
        event_ms_including_first=events,
        wrappers=wrappers,
        routes=route,
        complete_peak={
            k: max(full_peak[k], result["steady_peak"][k]) for k in full_peak
        },
    )
except Exception as exc:
    result.update(
        status="oom"
        if isinstance(exc, torch.OutOfMemoryError)
        or type(exc).__name__ == "OutOfMemoryError"
        else "failed",
        exception=traceback.format_exc(),
        partial_arrays=len(arrays),
        complete_peak={
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        },
    )
    print(result["exception"], flush=True)
finally:
    result["elapsed_s"] = time.time() - result["started"]
    write("result.json", result)
raise SystemExit(0 if result["status"] == "completed" else 1)
