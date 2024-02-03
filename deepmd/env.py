# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
import os
from typing import (
    Tuple,
)

import numpy as np

__all__ = [
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "global_float_prec",
]

log = logging.getLogger(__name__)

# FLOAT_PREC
dp_float_prec = os.environ.get("DP_INTERFACE_PREC", "high").lower()
if dp_float_prec in ("high", ""):
    # default is high
    GLOBAL_NP_FLOAT_PRECISION = np.float64
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "double"
elif dp_float_prec == "low":
    GLOBAL_NP_FLOAT_PRECISION = np.float32
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "float"
else:
    raise RuntimeError(
        "Unsupported float precision option: %s. Supported: high,"
        "low. Please set precision with environmental variable "
        "DP_INTERFACE_PREC." % dp_float_prec
    )


def set_env_if_empty(key: str, value: str, verbose: bool = True):
    """Set environment variable only if it is empty.

    Parameters
    ----------
    key : str
        env variable name
    value : str
        env variable value
    verbose : bool, optional
        if True action will be logged, by default True
    """
    if os.environ.get(key) is None:
        os.environ[key] = value
        if verbose:
            log.warning(
                f"Environment variable {key} is empty. Use the default value {value}"
            )


def set_default_nthreads():
    """Set internal number of threads to default=automatic selection.

    Notes
    -----
    `DP_INTRA_OP_PARALLELISM_THREADS` and `DP_INTER_OP_PARALLELISM_THREADS`
    control configuration of multithreading.
    """
    if (
        "OMP_NUM_THREADS" not in os.environ
        # for backward compatibility
        or (
            "DP_INTRA_OP_PARALLELISM_THREADS" not in os.environ
            and "TF_INTRA_OP_PARALLELISM_THREADS" not in os.environ
        )
        or (
            "DP_INTER_OP_PARALLELISM_THREADS" not in os.environ
            and "TF_INTER_OP_PARALLELISM_THREADS" not in os.environ
        )
    ):
        log.warning(
            "To get the best performance, it is recommended to adjust "
            "the number of threads by setting the environment variables "
            "OMP_NUM_THREADS, DP_INTRA_OP_PARALLELISM_THREADS, and "
            "DP_INTER_OP_PARALLELISM_THREADS. See "
            "https://deepmd.rtfd.io/parallelism/ for more information."
        )
    if "TF_INTRA_OP_PARALLELISM_THREADS" not in os.environ:
        set_env_if_empty("DP_INTRA_OP_PARALLELISM_THREADS", "0", verbose=False)
    if "TF_INTER_OP_PARALLELISM_THREADS" not in os.environ:
        set_env_if_empty("DP_INTER_OP_PARALLELISM_THREADS", "0", verbose=False)


def get_default_nthreads() -> Tuple[int, int]:
    """Get paralellism settings.

    The method will first read the environment variables with the prefix `DP_`.
    If not found, it will read the environment variables with the prefix `TF_`
    for backward compatibility.

    Returns
    -------
    Tuple[int, int]
        number of `DP_INTRA_OP_PARALLELISM_THREADS` and
        `DP_INTER_OP_PARALLELISM_THREADS`
    """
    return int(
        os.environ.get(
            "DP_INTRA_OP_PARALLELISM_THREADS",
            os.environ.get("TF_INTRA_OP_PARALLELISM_THREADS", "0"),
        )
    ), int(
        os.environ.get(
            "DP_INTER_OP_PARALLELISM_THREADS",
            os.environ.get("TF_INTRA_OP_PARALLELISM_THREADS", "0"),
        )
    )
