# SPDX-License-Identifier: LGPL-3.0-or-later
"""JAX matrix-product precision for training and evaluation."""

from collections.abc import (
    Generator,
)
from contextlib import (
    contextmanager,
)
from contextvars import (
    ContextVar,
)

from deepmd.dpmodel.train.inference import (
    infer_matmul_precision,
)
from deepmd.jax.env import (
    jax,
    jnp,
)

_MATMUL_AUTOCAST: ContextVar[bool | None] = ContextVar(
    "deepmd_jax_matmul_autocast",
    default=None,
)


@contextmanager
def autocast_matmul(*, fp32_output: bool = True) -> Generator[None, None, None]:
    """Trace eligible matrix products with BF16 operands.

    The scope is entered inside the traced model according to its static
    training or inference policy. It does not change an already compiled
    executable. Nested scopes and independent tracing threads retain their
    own context.

    Parameters
    ----------
    fp32_output : bool, default True
        Retain FP32 outputs for higher-order training derivatives. False
        preserves BF16 activation storage under the inference AMP policy.
    """
    token = _MATMUL_AUTOCAST.set(fp32_output)
    try:
        yield
    finally:
        _MATMUL_AUTOCAST.reset(token)


def _matmul_arguments(
    *values: jax.Array,
) -> tuple[tuple[jax.Array, ...], jnp.dtype | None]:
    operands = tuple(jnp.asarray(value) for value in values)
    fp32_output = _MATMUL_AUTOCAST.get()
    if fp32_output is not None and all(
        value.dtype in (jnp.float32, jnp.bfloat16) for value in operands
    ):
        # FP32 outputs keep nonlinear activations and higher-order adjoints
        # out of BF16 arithmetic. The multiplicands retain the reduced
        # precision used by tensor-core matrix products.
        return (
            tuple(value.astype(jnp.bfloat16) for value in operands),
            jnp.dtype(jnp.float32) if fp32_output else None,
        )
    return operands, None


def matmul(left: jax.Array, right: jax.Array) -> jax.Array:
    """Evaluate a matrix product under the active tracing precision policy."""
    operands, output_dtype = _matmul_arguments(left, right)
    return jnp.matmul(*operands, preferred_element_type=output_dtype)


def einsum(subscripts: str, *operands: jax.Array) -> jax.Array:
    """Evaluate a contraction under the active tracing precision policy."""
    operands, output_dtype = _matmul_arguments(*operands)
    return jnp.einsum(subscripts, *operands, preferred_element_type=output_dtype)


class MatmulPrecisionPolicy:
    """Keep training TF32 and inference precision independently configurable."""

    def __init__(self, enable_tf32: bool = False) -> None:
        self.train_precision = "high" if enable_tf32 else "highest"
        self.eval_precision = infer_matmul_precision()

    @contextmanager
    def applied(self, *, training: bool) -> Generator[None, None, None]:
        """Trace matrix products with the requested accumulation precision."""
        level = self.train_precision if training else self.eval_precision
        precision = {"highest": "highest", "high": "high", "medium": "default"}
        with jax.default_matmul_precision(precision[level]):
            yield
