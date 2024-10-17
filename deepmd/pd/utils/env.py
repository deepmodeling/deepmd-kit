# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os

import numpy as np
import paddle

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

SAMPLER_RECORD = os.environ.get("SAMPLER_RECORD", False)
try:
    # only linux
    ncpus = len(os.sched_getaffinity(0))
except AttributeError:
    ncpus = os.cpu_count()
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(0, ncpus)))
# Make sure DDP uses correct device if applicable
LOCAL_RANK = paddle.distributed.get_rank()

if os.environ.get("DEVICE") == "cpu" or paddle.device.cuda.device_count() <= 0:
    DEVICE = "cpu"
else:
    DEVICE = f"gpu:{LOCAL_RANK}"

JIT = False
CACHE_PER_SYS = 5  # keep at most so many sets per sys in memory
ENERGY_BIAS_TRAINABLE = True

PRECISION_DICT = {
    "float16": paddle.float16,
    "float32": paddle.float32,
    "float64": paddle.float64,
    "half": paddle.float16,
    "single": paddle.float32,
    "double": paddle.float64,
    "int32": paddle.int32,
    "int64": paddle.int64,
    "bfloat16": paddle.bfloat16,
    "bool": paddle.bool,
}
GLOBAL_PD_FLOAT_PRECISION = PRECISION_DICT[np.dtype(GLOBAL_NP_FLOAT_PRECISION).name]
GLOBAL_PD_ENER_FLOAT_PRECISION = PRECISION_DICT[
    np.dtype(GLOBAL_ENER_FLOAT_PRECISION).name
]
PRECISION_DICT["default"] = GLOBAL_PD_FLOAT_PRECISION
assert VALID_PRECISION.issubset(PRECISION_DICT.keys())
# cannot automatically generated
RESERVED_PRECISON_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
    paddle.int32: "int32",
    paddle.int64: "int64",
    paddle.bfloat16: "bfloat16",
    paddle.bool: "bool",
}
assert set(PRECISION_DICT.values()) == set(RESERVED_PRECISON_DICT.keys())
DEFAULT_PRECISION = "float64"

# throw warnings if threads not set
set_default_nthreads()
inter_nthreads, intra_nthreads = get_default_nthreads()
# if inter_nthreads > 0:  # the behavior of 0 is not documented
#     paddle.set_num_interop_threads(inter_nthreads)
# if intra_nthreads > 0:
#     paddle.framework.core.set_num_threads(intra_nthreads)


def enable_prim(enable: bool = True):
    """Enable running program in primitive C++ API in eager/static mode."""
    if enable:
        from paddle.framework import (
            core,
        )

        core.set_prim_eager_enabled(True)
        core._set_prim_all_enabled(True)
        log.info("Enable prim in eager and static mode.")


__all__ = [
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_PD_FLOAT_PRECISION",
    "GLOBAL_PD_ENER_FLOAT_PRECISION",
    "DEFAULT_PRECISION",
    "PRECISION_DICT",
    "RESERVED_PRECISON_DICT",
    "SAMPLER_RECORD",
    "NUM_WORKERS",
    "DEVICE",
    "JIT",
    "CACHE_PER_SYS",
    "ENERGY_BIAS_TRAINABLE",
    "LOCAL_RANK",
]
