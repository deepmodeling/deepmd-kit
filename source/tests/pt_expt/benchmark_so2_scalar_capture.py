# SPDX-License-Identifier: LGPL-3.0-or-later
"""Manual full-API CSC5 benchmark; run one route in each fresh process.

Example: python source/tests/pt_expt/benchmark_so2_scalar_capture.py \
    --model model.pt --route CSC5 --cell 8 8 8 --out results/csc5
"""
# ruff: noqa: T201 -- standalone benchmark CLI

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import time
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
p.add_argument("--route", choices=("TORCH", "BASE", "CSC5"), required=True)
p.add_argument("--cell", type=int, nargs=3, default=(5, 5, 5))
p.add_argument("--out", type=Path, required=True)
p.add_argument("--cache", type=Path)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--warmup", type=int, default=12)
p.add_argument("--samples", type=int, default=30)
p.add_argument("--legacy-pro", action="store_true")
a = p.parse_args()
assert a.warmup >= 3 and a.samples >= 10
a.out.mkdir(parents=True, exist_ok=False)
cache = a.cache or a.out / "cache"


def write(name, data):
    (a.out / name).write_text(json.dumps(data, indent=2, allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    write("result.json", {"status": "busy", "pids": busy})
    raise SystemExit(3)
for key in list(os.environ):
    if key.startswith("DP_"):
        del os.environ[key]
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
os.environ.update(
    CUDA_VISIBLE_DEVICES=str(a.gpu),
    DP_COMPILE_INFER="1",
    DP_COMPILE="0",
    DP_TRITON_INFER="0" if a.route == "TORCH" else "2",
    DP_CUDA_INFER="0",
    DP_TF32_INFER="0",
    DP_AMP_INFER="0",
    DP_SO2_SCALAR_CAPTURE_INFER="1" if a.route == "CSC5" else "0",
    OMP_NUM_THREADS="1",
    DP_INTRA_OP_PARALLELISM_THREADS="1",
    DP_INTER_OP_PARALLELISM_THREADS="1",
    PYTORCH_ALLOC_CONF="expandable_segments:True",
    TORCHINDUCTOR_COMPILE_THREADS="1",
    TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
    TRITON_CACHE_DIR=str(cache / "triton"),
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
)
sys.path.insert(0, str(a.root.resolve()))
sys.meta_path[:] = [
    f
    for f in sys.meta_path
    if not type(f).__module__.startswith("_editable_skbc_deepmd_kit")
]
import deepmd

assert Path(deepmd.__file__).resolve().parent.parent == a.root.resolve()
# Permit a source checkout to use its matching installed native extension.
for installed in site.getsitepackages():
    lib = Path(installed) / "deepmd"
    if (lib / "lib").is_dir() and str(lib) not in deepmd.__path__:
        deepmd.__path__.append(str(lib))
import numpy as np
import torch
import triton
from ase.build import bulk
from deepmd.calculator import DP
from deepmd.utils.errors import OutOfMemoryError as DPOutOfMemoryError

torch.set_num_threads(1)
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(False)
torch.cuda.reset_peak_memory_stats()
write(
    "manifest.json",
    {
        "args": {k: str(v) for k, v in vars(a).items()},
        "model_sha256": sha(a.model),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=a.root, text=True
        ).strip(),
        "source_sha256": sha(
            a.root / "deepmd/pt_expt/kernels/triton/sezm/so2_value_path.py"
        ),
        "benchmark_sha256": sha(__file__),
        "flags": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("DP_", "TORCHINDUCTOR", "TRITON_CACHE", "PYTORCH_ALLOC"))
        },
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(a.gpu),
                "--query-gpu=name,driver_version,uuid,clocks.sm,clocks.mem,temperature.gpu",
                "--format=csv",
            ],
            text=True,
        ),
    },
)
result = {
    "status": "running",
    "route": a.route,
    "n_atoms": int(8 * np.prod(a.cell)),
    "atol": 1e-5,
    "rtol": 1e-4,
}
arrays = {}
started = time.time()
try:
    atoms = bulk("C", "diamond", a=3.567, cubic=True).repeat(tuple(a.cell))
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
    load = torch.load
    if a.legacy_pro:
        assert (
            sha(a.model)
            == "446a6e893101fc52190cfccabb9d2328863d651aca36981fd8d3401c71a61221"
        )

        def legacy_load(file, *args, **kwargs):
            obj = load(file, *args, **kwargs)
            if (
                isinstance(file, (str, Path))
                and Path(file).resolve() == a.model.resolve()
            ):
                state = obj["model"] if "model" in obj else obj
                state["_extra_state"] = copy.deepcopy(state["_extra_state"])
                fitting = state["_extra_state"]["model_params"]["fitting_net"]
                assert fitting["type"] == "ener"
                fitting["type"] = "dpa4_ener"
            return obj

        torch.load = legacy_load
        result["legacy_pro"] = (
            "in-memory fitting type alias ener -> dpa4_ener; weights unchanged"
        )
    try:
        calc = DP(model=str(a.model), freeze=False)
    finally:
        torch.load = load
    assert not calc.dp.get_model().training
    result["parameters"] = sum(t.numel() for t in calc.dp.get_model().parameters())
    write("model_definition.json", calc.dp.get_model_def_script())

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

    def peak():
        return {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }

    def stats(values):
        return {
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "min": min(values),
            "max": max(values),
            "samples": values,
        }

    result["accuracy_wall_ms"] = []
    for repeat in range(3):
        for frame, pos in enumerate(frames):
            tick = time.perf_counter()
            out = evaluate(pos)
            torch.cuda.synchronize()
            assert all(np.isfinite(v).all() for v in out.values())
            result["accuracy_wall_ms"].append(1000 * (time.perf_counter() - tick))
            arrays.update({f"{repeat}_{frame}_{k}": v for k, v in out.items()})
            np.savez_compressed(a.out / "outputs.npz", **arrays)
            if repeat == frame == 0:
                result["first_peak"] = peak()
            print(a.route, len(atoms), repeat, frame, flush=True)
    for i in range(a.warmup):
        evaluate(frames[i % 3])
    wall, events = [], []
    for i in range(a.samples):
        torch.cuda.synchronize()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        tick = time.perf_counter()
        begin.record()
        out = evaluate(frames[i % 3])
        end.record()
        torch.cuda.synchronize()
        wall.append(1000 * (time.perf_counter() - tick))
        events.append(begin.elapsed_time(end))
        assert all(np.isfinite(v).all() for v in out.values())
    full = peak()
    torch.cuda.reset_peak_memory_stats()
    for pos in frames:
        evaluate(pos)
    torch.cuda.synchronize()
    result["steady_peak"] = peak()
    result["complete_peak"] = {k: max(full[k], result["steady_peak"][k]) for k in full}
    from torch._inductor.codecache import PyCodeCache

    wrappers = [
        str(m.__file__)
        for m in PyCodeCache.modules
        if "def call(" in Path(m.__file__).read_text()
    ]
    assert wrappers
    result.update(
        status="completed",
        finite=True,
        wall_ms=stats(wall),
        event_ms=stats(events),
        wrappers=wrappers,
    )
except Exception as exc:
    result.update(
        status="oom"
        if isinstance(exc, (torch.OutOfMemoryError, DPOutOfMemoryError))
        else "failed",
        exception=traceback.format_exc(),
        complete_peak={
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        },
    )
    print(result["exception"], flush=True)
finally:
    result["elapsed_s"] = time.time() - started
    write("result.json", result)
raise SystemExit(0 if result["status"] == "completed" else 1)
