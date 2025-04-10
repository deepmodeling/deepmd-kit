# SPDX-License-Identifier: LGPL-3.0-or-later
from __future__ import (
    annotations,
)

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

paddle.device.set_device(DEVICE)


def to_bool(flag: int | bool | str) -> bool:
    if isinstance(flag, int):
        if flag not in [0, 1]:
            raise ValueError(f"flag must be either 0 or 1, but received {flag}")
        return bool(flag)

    elif isinstance(flag, str):
        flag = flag.lower()
        if flag not in ["1", "0", "true", "false"]:
            raise ValueError(
                f"flag must be either '0', '1', 'true', 'false', but received '{flag}'"
            )
        return flag in ["1", "true"]

    elif isinstance(flag, bool):
        return flag

    else:
        raise ValueError(
            f"flag must be either int, bool, or str, but received {type(flag).__name__}"
        )


JIT = to_bool(os.environ.get("JIT", False))
CINN = to_bool(os.environ.get("CINN", False))
if CINN:
    assert paddle.device.is_compiled_with_cinn(), (
        "CINN is set to True, but PaddlePaddle is not compiled with CINN support. "
        "Ensure that your PaddlePaddle installation supports CINN by checking your "
        "installation or recompiling with CINN enabled."
    )

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
RESERVED_PRECISION_DICT = {
    paddle.float16: "float16",
    paddle.float32: "float32",
    paddle.float64: "float64",
    paddle.int32: "int32",
    paddle.int64: "int64",
    paddle.bfloat16: "bfloat16",
    paddle.bool: "bool",
}
assert set(PRECISION_DICT.values()) == set(RESERVED_PRECISION_DICT.keys())
DEFAULT_PRECISION = "float64"

# throw warnings if threads not set
set_default_nthreads()
inter_nthreads, intra_nthreads = get_default_nthreads()
# if inter_nthreads > 0:  # the behavior of 0 is not documented
#     os.environ['OMP_NUM_THREADS'] = str(inter_nthreads)
# if intra_nthreads > 0:
#     os.environ['CPU_NUM'] = str(intra_nthreads)


def enable_prim(enable: bool = True):
    # NOTE: operators in list below will not use composite
    # operator but kernel instead for better performance
    EAGER_COMP_OP_BLACK_LIST = [
        "abs_grad",
        "cast_grad",
        # "concat_grad",
        "cos_double_grad",
        "cos_grad",
        "cumprod_grad",
        "cumsum_grad",
        "dropout_grad",
        "erf_grad",
        "exp_grad",
        "expand_grad",
        "floor_grad",
        "gather_grad",
        "gather_nd_grad",
        "gelu_grad",
        "group_norm_grad",
        "instance_norm_grad",
        "layer_norm_grad",
        "leaky_relu_grad",
        "log_grad",
        "max_grad",
        "pad_grad",
        "pow_double_grad",
        "pow_grad",
        "prod_grad",
        "relu_grad",
        "roll_grad",
        "rsqrt_grad",
        "scatter_grad",
        "scatter_nd_add_grad",
        "sigmoid_grad",
        "silu_grad",
        "sin_double_grad",
        "sin_grad",
        "slice_grad",
        # "split_grad",
        "sqrt_grad",
        "stack_grad",
        "sum_grad",
        "tanh_double_grad",
        "tanh_grad",
        "topk_grad",
        "transpose_grad",
        "add_double_grad",
        "add_grad",
        "assign_grad",
        "batch_norm_grad",
        "divide_grad",
        "elementwise_pow_grad",
        "maximum_grad",
        "min_grad",
        "minimum_grad",
        "multiply_grad",
        "subtract_grad",
        "tile_grad",
    ]
    EAGER_COMP_OP_BLACK_LIST = list(set(EAGER_COMP_OP_BLACK_LIST))

    """Enable running program with primitive operators in eager/static mode."""
    if JIT or CINN:
        # jit mode
        paddle.framework.core._set_prim_all_enabled(enable)
        if enable:
            # No need to set a blacklist for now in JIT mode.
            pass
    else:
        # eager mode
        paddle.framework.core.set_prim_eager_enabled(enable)
        if enable:
            # Set a blacklist (i.e., disable several composite operators) in eager mode
            # to enhance computational performance.
            paddle.framework.core._set_prim_backward_blacklist(
                *EAGER_COMP_OP_BLACK_LIST
            )

    log = logging.getLogger(__name__)
    log.info(f"{'Enable' if enable else 'Disable'} prim in eager and static mode.")


__all__ = [
    "CACHE_PER_SYS",
    "CINN",
    "DEFAULT_PRECISION",
    "DEVICE",
    "ENERGY_BIAS_TRAINABLE",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_PD_ENER_FLOAT_PRECISION",
    "GLOBAL_PD_FLOAT_PRECISION",
    "JIT",
    "LOCAL_RANK",
    "NUM_WORKERS",
    "PRECISION_DICT",
    "RESERVED_PRECISION_DICT",
    "SAMPLER_RECORD",
    "enable_prim",
]
