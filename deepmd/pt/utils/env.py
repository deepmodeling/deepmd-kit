# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import numpy as np
import torch
import torch.nn.functional as F

from typing import (
    Callable,
)

from deepmd.common import (
    VALID_PRECISION,
    VALID_ACTIVATION,
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
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", min(8, ncpus)))
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

ACTIVATION_FN_DICT = {
    "relu": F.relu,
    "relu6": F.relu6,
    "softplus": F.softplus,
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    # PyTorch has no gelu_tf
    "gelu_tf": lambda x: x * 0.5 * (1.0 + torch.erf(x / 1.41421)),
    "linear": lambda x: x,
    "none": lambda x: x,
}
assert VALID_ACTIVATION.issubset(ACTIVATION_FN_DICT.keys())

def get_activation_func(
    activation_fn
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get activation function callable based on string name.

    Parameters
    ----------
    activation_fn : _ACTIVATION
        One of the defined activation functions

    Returns
    -------
    Callable[[torch.Tensor], torch.Tensor]
        Corresponding PyTorch callable

    Raises
    ------
    RuntimeError
        If unknown activation function is specified
    """
    if activation_fn is None:
        activation_fn = "none"
    assert activation_fn is not None
    if activation_fn.lower() not in ACTIVATION_FN_DICT:
        raise RuntimeError(f"{activation_fn} is not a valid activation function")
    return ACTIVATION_FN_DICT[activation_fn.lower()]

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
RESERVED_PRECISON_DICT = {
    torch.float16: "float16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bfloat16: "bfloat16",
    torch.bool: "bool",
}
assert set(PRECISION_DICT.values()) == set(RESERVED_PRECISON_DICT.keys())
DEFAULT_PRECISION = "float64"

# throw warnings if threads not set
set_default_nthreads()
inter_nthreads, intra_nthreads = get_default_nthreads()
if inter_nthreads > 0:  # the behavior of 0 is not documented
    torch.set_num_interop_threads(inter_nthreads)
if intra_nthreads > 0:
    torch.set_num_threads(intra_nthreads)

__all__ = [
    "GLOBAL_ENER_FLOAT_PRECISION",
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_PT_FLOAT_PRECISION",
    "GLOBAL_PT_ENER_FLOAT_PRECISION",
    "DEFAULT_PRECISION",
    "ACTIVATION_FN_DICT",
    "PRECISION_DICT",
    "RESERVED_PRECISON_DICT",
    "SAMPLER_RECORD",
    "NUM_WORKERS",
    "DEVICE",
    "JIT",
    "CACHE_PER_SYS",
    "ENERGY_BIAS_TRAINABLE",
    "LOCAL_RANK",
    "get_activation_func"
]
