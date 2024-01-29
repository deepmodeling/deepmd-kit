# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np
import torch

PRECISION = os.environ.get("PRECISION", "float64")
GLOBAL_NP_FLOAT_PRECISION = getattr(np, PRECISION)
GLOBAL_PT_FLOAT_PRECISION = getattr(torch, PRECISION)
GLOBAL_ENER_FLOAT_PRECISION = getattr(np, PRECISION)
SAMPLER_RECORD = os.environ.get("SAMPLER_RECORD", False)
try:
    # only linux
    ncpus = len(os.sched_getaffinity(0))
except AttributeError:
    ncpus = os.cpu_count()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(8, ncpus)))
# Make sure DDP uses correct device if applicable
LOCAL_RANK = os.environ.get("LOCAL_RANK")
LOCAL_RANK = int(0 if LOCAL_RANK is None else LOCAL_RANK)

if os.environ.get("DEVICE") == "cpu" or torch.cuda.is_available() is False:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

if os.environ.get("PREPROCESS_DEVICE") == "gpu":
    PREPROCESS_DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
else:
    PREPROCESS_DEVICE = torch.device("cpu")

JIT = False
CACHE_PER_SYS = 5  # keep at most so many sets per sys in memory
ENERGY_BIAS_TRAINABLE = True

PRECISION_DICT = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
}
DEFAULT_PRECISION = "float64"
