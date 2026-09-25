# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import multiprocessing
import os
import sys

import numpy as np

from deepmd.common import (
    VALID_PRECISION,
)
from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
    get_default_nthreads,
    set_default_nthreads,
)

log = logging.getLogger(__name__)
import torch

if sys.platform != "win32":
    try:
        multiprocessing.set_start_method("fork", force=True)
        log.debug("Successfully set multiprocessing start method to 'fork'.")
    except (RuntimeError, ValueError) as err:
        log.warning(f"Could not set multiprocessing start method: {err}")
else:
    log.debug("Skipping fork start method on Windows (not supported).")

SAMPLER_RECORD = os.environ.get("SAMPLER_RECORD", False)
DP_DTYPE_PROMOTION_STRICT = os.environ.get("DP_DTYPE_PROMOTION_STRICT", "0") == "1"
# Number of Hessian rows evaluated per second-order backward pass. The Hessian
# is built from Hessian-vector products; batching them over replicated frames
# trades memory for far fewer kernel launches. 1 keeps the one-row-at-a-time
# path and reproduces the pre-batching behaviour exactly.
#
# Left unset, the batch is chosen per call: one Hessian-vector product is run to
# measure what a replica costs, and the batch is what the free memory affords,
# clamped to [1, DP_HESSIAN_HVP_BATCH_CAP]. That is not a tuning preference but
# a correctness matter, because peak memory is linear in this value while the
# speedup is not. Measured on one H20 with DPA-4.0.1-Pro-MPtrj in eval mode,
# float32, TF32 off:
#
#     peak(MiB) = 782 + [4.50 + 3.27*(B-1)]*edges + [10.9 + 7.96*(B-1)]*natoms
#
# so at fcc-solid density a 96 GiB card holds ~381 atoms at B=1 but only ~63 at
# B=8, while the speedup falls from 5.06x at 72 edges to 1.24x at 5832 edges --
# batching recovers kernel-launch overhead, which stops mattering once a single
# Hessian-vector product already saturates the device. A fixed large value
# therefore costs the size ceiling and buys nothing on the systems that need it.
#
# An explicit value is honoured as given, including above the cap. The
# out-of-memory fallback still applies to it: halving the batch changes how the
# Hessian is computed, never what it is, so a run that would have died is
# finished instead, with a warning naming the batch actually used.
_hessian_hvp_batch = os.environ.get("DP_HESSIAN_HVP_BATCH")
DP_HESSIAN_HVP_BATCH: int | None = (
    int(_hessian_hvp_batch) if _hessian_hvp_batch is not None else None
)
# Ceiling for the automatic choice. Past this the speedup has flattened on every
# system measured, so more batch would only cost memory.
DP_HESSIAN_HVP_BATCH_CAP = 8
# Share of the free memory the automatic choice plans for. The rest absorbs the
# gap between one replica's measured cost and the marginal cost of the next.
DP_HESSIAN_HVP_MEMORY_FRACTION = 0.5
try:
    # only linux
    ncpus = len(os.sched_getaffinity(0))
except AttributeError:
    ncpus = os.cpu_count() or 1
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(4, ncpus)))
if multiprocessing.get_start_method() != "fork":
    # spawn or forkserver does not support NUM_WORKERS > 0 for DataLoader
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
intra_nthreads, inter_nthreads = get_default_nthreads()
if inter_nthreads > 0:  # the behavior of 0 is not documented
    # torch.set_num_interop_threads can only be called once per process.
    # Guard to avoid RuntimeError when both pt and pt_expt env modules are imported.
    try:
        if torch.get_num_interop_threads() != inter_nthreads:
            torch.set_num_interop_threads(inter_nthreads)
    except RuntimeError as err:
        log.warning(f"Could not set torch interop threads: {err}")
if intra_nthreads > 0:
    # torch.set_num_threads can also fail if called after threads are created.
    try:
        if torch.get_num_threads() != intra_nthreads:
            torch.set_num_threads(intra_nthreads)
    except RuntimeError as err:
        log.warning(f"Could not set torch intra threads: {err}")

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
