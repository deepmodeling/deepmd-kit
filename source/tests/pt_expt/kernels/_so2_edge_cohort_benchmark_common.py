# SPDX-License-Identifier: LGPL-3.0-or-later
"""Isolated-process setup and serialization for the SO(2) benchmark."""

import hashlib
import json
import os
from pathlib import Path
import site
import subprocess
import sys

LAB = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[4]


def setup(gpu, out, level="2"):
    out.mkdir(parents=True, exist_ok=False)
    busy = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(gpu),
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if busy:
        raise RuntimeError(f"GPU {gpu} has compute processes: {busy}")
    for key in list(os.environ):
        if key.startswith("DP_"):
            del os.environ[key]
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        DP_COMPILE_INFER="1",
        DP_COMPILE="0",
        DP_TRITON_INFER=level,
        DP_CUDA_INFER="0",
        DP_TF32_INFER="0",
        DP_AMP_INFER="0",
        OMP_NUM_THREADS="1",
        DP_INTRA_OP_PARALLELISM_THREADS="1",
        DP_INTER_OP_PARALLELISM_THREADS="1",
        PYTORCH_ALLOC_CONF=os.getenv("LAB_ALLOC_CONF", "expandable_segments:True"),
        TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="1",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        TORCHINDUCTOR_CACHE_DIR=str(out.resolve() / "inductor"),
        TRITON_CACHE_DIR=str(out.resolve() / "triton"),
    )
    sys.path.insert(0, str(ROOT))
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if not type(f).__module__.startswith("_editable_skbc_deepmd_kit")
    ]
    import deepmd

    for installed in site.getsitepackages():
        p = Path(installed) / "deepmd"
        if (p / "lib").is_dir():
            deepmd.__path__.append(str(p))
    assert Path(deepmd.__file__).resolve().parent.parent == ROOT
    import torch

    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write(path, obj):
    Path(path).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import torch
    import triton

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,driver_version,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                "--format=csv",
            ],
            text=True,
        ),
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "flags": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(
                (
                    "DP_",
                    "PBS_",
                    "COHORT_",
                    "TORCHINDUCTOR",
                    "TRITON_CACHE",
                    "CUDA_VISIBLE",
                    "PYTORCH_ALLOC",
                )
            )
        },
        "harness_hashes": {
            p.name: sha(p) for p in LAB.glob("*so2_edge_cohort*benchmark*.py")
        },
    }


def stats(values):
    import numpy as np

    return {
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "min": float(min(values)),
        "max": float(max(values)),
        "samples": values,
    }
