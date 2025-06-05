# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import multiprocessing
import os

import numpy as np
import torch

from deepmd.common import (
    VALID_PRECISION,
)
from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    get_default_nthreads,
    set_default_nthreads,
)

SAMPLER_RECORD = os.environ.get("SAMPLER_RECORD", False)
DP_DTYPE_PROMOTION_STRICT = os.environ.get("DP_DTYPE_PROMOTION_STRICT", "0") == "1"
try:
    # only linux
    ncpus = len(os.sched_getaffinity(0))
except AttributeError:
    ncpus = os.cpu_count()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(4, ncpus)))
if multiprocessing.get_start_method() != "fork":
    # spawn or forkserver does not support NUM_WORKERS > 0 for DataLoader
    log = logging.getLogger(__name__)
    log.warning(
        "NUM_WORKERS > 0 is not supported with spawn or forkserver start method. "
        "Setting NUM_WORKERS to 0."
    )
    NUM_WORKERS = 0

# Make sure DDP uses correct device if applicable
LOCAL_RANK = os.environ.get("LOCAL_RANK")
LOCAL_RANK = int(0 if LOCAL_RANK is None else LOCAL_RANK)

if os.environ.get("DEVICE") == "cpu" or torch.cuda.is_available() is False:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

JIT = False
CACHE_PER_SYS = 5  # keep at most so many sets per sys in memory
ENERGY_BIAS_TRAINABLE = True
CUSTOM_OP_USE_JIT = False

PRECISION_DICT = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
}
GLOBAL_PT_FLOAT_PRECISION = PRECISION_DICT[np.dtype(GLOBAL_NP_FLOAT_PRECISION).name]
GLOBAL_PT_ENER_FLOAT_PRECISION = PRECISION_DICT[
    np.dtype(GLOBAL_ENER_FLOAT_PRECISION).name
]
PRECISION_DICT["default"] = GLOBAL_PT_FLOAT_PRECISION
assert VALID_PRECISION.issubset(PRECISION_DICT.keys())
# cannot automatically generated
RESERVED_PRECISION_DICT = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bfloat16: "bfloat16",
    torch.bool: "bool",
}
assert set(PRECISION_DICT.values()) == set(RESERVED_PRECISION_DICT.keys())
DEFAULT_PRECISION = "float64"

# throw warnings if threads not set
set_default_nthreads()
inter_nthreads, intra_nthreads = get_default_nthreads()
if inter_nthreads > 0:  # the behavior of 0 is not documented
    torch.set_num_interop_threads(inter_nthreads)
if intra_nthreads > 0:
    torch.set_num_threads(intra_nthreads)

__all__ = [
    "CACHE_PER_SYS",
    "CUSTOM_OP_USE_JIT",
    "DEFAULT_PRECISION",
    "DEVICE",
    "ENERGY_BIAS_TRAINABLE",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_PT_ENER_FLOAT_PRECISION",
    "GLOBAL_PT_FLOAT_PRECISION",
    "JIT",
    "LOCAL_RANK",
    "NUM_WORKERS",
    "PRECISION_DICT",
    "RESERVED_PRECISION_DICT",
    "SAMPLER_RECORD",
]
