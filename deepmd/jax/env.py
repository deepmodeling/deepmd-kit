# SPDX-License-Identifier: LGPL-3.0-or-later
import os

from deepmd.env import (
    get_default_nthreads,
    set_default_nthreads,
)

set_default_nthreads()
inter_nthreads, intra_nthreads = get_default_nthreads()
if inter_nthreads > 0 and "inter_op_parallelism_threads=" not in os.environ.get(
    "XLA_FLAGS", ""
):
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + (
        f" inter_op_parallelism_threads={inter_nthreads}"
    )
if intra_nthreads > 0 and "intra_op_parallelism_threads=" not in os.environ.get(
    "XLA_FLAGS", ""
):
    os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + (
        f" intra_op_parallelism_threads={intra_nthreads}"
    )
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from flax import (
    nnx,
)
from jax import export as jax_export

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

__all__ = [
    "jax",
    "jnp",
    "nnx",
    "jax_export",
]
