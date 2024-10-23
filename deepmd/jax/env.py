# SPDX-License-Identifier: LGPL-3.0-or-later
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import jax
import jax.numpy as jnp
from flax import (
    nnx,
)

jax.config.update("jax_enable_x64", True)

__all__ = [
    "jax",
    "jnp",
    "nnx",
]
