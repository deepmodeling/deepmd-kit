# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: T201 -- manual benchmark CLI output
"""Reproduce full-model first-call, global and steady SO(2) memory results.

This manual CUDA benchmark is not collected by pytest. Checkpoint files and
generated compiler caches are intentionally not included in the repository.
"""

import argparse
import os
from pathlib import Path
import runpy
import sys

import _so2_edge_cohort_benchmark_common as common

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument(
    "--source-root", type=Path, default=Path(__file__).resolve().parents[4]
)
parser.add_argument("--mode", choices=("base", "ecs", "torch"), default="base")
parser.add_argument("--gpu", default="0")
parser.add_argument("--nrep", type=int, default=2)
parser.add_argument("--warmup", type=int, default=8)
parser.add_argument("--samples", type=int, default=30)
parser.add_argument("--reference", type=Path)
parser.add_argument("--cache-dir", type=Path)
parser.add_argument("--allow-warm-cache", action="store_true")
args = parser.parse_args()
common.ROOT = args.source_root.resolve()
original_setup = common.setup


def setup(gpu, out, level="2"):
    """Select code and precision before importing the model implementation."""
    original_setup(gpu, out, level)
    os.environ["DP_SO2_EDGE_COHORT_INFER"] = "1" if args.mode == "ecs" else "0"
    if args.cache_dir is not None:
        cache = args.cache_dir.resolve()
        if cache.exists() and any(cache.iterdir()) and not args.allow_warm_cache:
            raise ValueError(
                "nonempty cache: pass --allow-warm-cache or use a new directory"
            )
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
        os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")


common.setup = setup
implementation = Path(__file__).with_name("_so2_edge_cohort_benchmark_impl.py")
sys.argv = [
    str(implementation),
    "--model",
    str(args.model),
    "--out",
    str(args.out),
    "--gpu",
    args.gpu,
    "--nrep",
    str(args.nrep),
    "--warmup",
    str(args.warmup),
    "--samples",
    str(args.samples),
    "--variant",
    "TORCH" if args.mode == "torch" else "BASE",
]
if args.reference is not None:
    sys.argv += ["--reference", str(args.reference)]
namespace = runpy.run_path(str(implementation), run_name="__main__")
import torch

result = namespace["result"]
result["variant"] = args.mode
memory = result["memory"]
# The implementation resets counters only for its separate steady replay.
# Preserve the preceding high-water mark, including model load and first JIT.
memory["after_validation"] = {
    "allocated": torch.cuda.max_memory_allocated(),
    "reserved": torch.cuda.max_memory_reserved(),
}
memory["global"] = {
    key: max(
        memory[phase][key]
        for phase in ("first", "complete", "steady", "after_validation")
    )
    for key in ("allocated", "reserved")
}
memory["global_first_observed_phase"] = next(
    phase
    for phase in ("first", "complete", "steady", "after_validation")
    if memory[phase]["allocated"] == memory["global"]["allocated"]
)
result["protocol"] = {
    "seed": 0,
    "coordinate_std_angstrom": 0.02,
    "cache": "warm_allowed"
    if args.allow_warm_cache
    else "new_inductor_and_triton_cache",
    "primary_memory": "global high-water mark including model load and first JIT; never substitute steady",
    "tf32": False,
    "amp": False,
    "weight_gradients": False,
    "cold_scope": "new process and application compiler caches; driver/system caches are not purged",
}
result["source_hashes"] = {
    str(p.relative_to(common.ROOT)): common.sha(p)
    for p in [
        common.ROOT / "deepmd/pt_expt/kernels/triton/sezm/so2_value_path.py",
        common.ROOT / "deepmd/pt_expt/kernels/triton/sezm/so2_edge_cohort.py",
    ]
    if p.exists()
}
assert torch.get_float32_matmul_precision() == "highest"
assert not torch.backends.cuda.matmul.allow_tf32
assert not torch.backends.cudnn.allow_tf32
common.write(args.out / "result.json", result)
if args.reference is not None and not result["precision_pass"]:
    raise SystemExit("accuracy check failed; thresholds were not relaxed")
if not all(v["passed"] for repeat in result["repeat_noise"] for v in repeat.values()):
    raise SystemExit("within-process reproducibility check failed")
print(
    {
        "mode": args.mode,
        "p50_ms": result["wall_ms"]["median"],
        "first_peak": memory["first"],
        "global_peak": memory["global"],
        "steady_peak": memory["steady"],
    }
)
