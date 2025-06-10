# SPDX-License-Identifier: LGPL-3.0-or-later
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from flax import (
    nnx,
)
from jax import export as jax_export

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# https://github.com/jax-ml/jax/issues/24909
jax.config.update("jax_default_matmul_precision", "tensorfloat32")

if os.environ.get("DP_DTYPE_PROMOTION_STRICT") == "1":
    jax.config.update("jax_numpy_dtype_promotion", "strict")

__all__ = [
    "jax",
    "jax_export",
    "jnp",
    "nnx",
]
