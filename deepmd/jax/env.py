# SPDX-License-Identifier: LGPL-3.0-or-later
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from flax import __version__ as flax_version
from flax import (
    nnx,
)
from flax.nnx.training.optimizer import (
    OptState,
)
from jax import export as jax_export

jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
# https://github.com/jax-ml/jax/issues/24909
if "JAX_DEFAULT_MATMUL_PRECISION" not in os.environ:
    jax.config.update("jax_default_matmul_precision", "highest")

if os.environ.get("DP_DTYPE_PROMOTION_STRICT") == "1":
    jax.config.update("jax_numpy_dtype_promotion", "strict")

__all__ = [
    "OptState",
    "flax_version",
    "jax",
    "jax_export",
    "jnp",
    "nnx",
]
