# SPDX-License-Identifier: LGPL-3.0-or-later
"""Matrix-product precision scopes preserve parameters and wider arithmetic."""

import numpy as np
import pytest

from deepmd.dpmodel.array_api import (
    xp_einsum,
    xp_matmul,
)
from deepmd.dpmodel.descriptor.dpa4_nn.lora import (
    LoRASO3,
)
from deepmd.dpmodel.descriptor.dpa4_nn.so3 import (
    SO3Linear,
)
from deepmd.jax.common import (
    ArrayAPIVariable,
)
from deepmd.jax.env import (
    jax,
    jnp,
)
from deepmd.jax.utils.precision import (
    autocast_matmul,
)


@pytest.mark.parametrize("contraction", [False, True])
@pytest.mark.parametrize("right_dtype", [jnp.float32, jnp.float64])
def test_autocast_accepts_variable_operands_and_preserves_float64(
    contraction: bool,
    right_dtype: jnp.dtype,
) -> None:
    """Only wholly FP32/BF16 operations enter the lower-precision policy."""
    left = ArrayAPIVariable(jnp.arange(6, dtype=jnp.float32).reshape(2, 3))
    right = jnp.arange(12, dtype=right_dtype).reshape(3, 4)

    def product() -> jax.Array:
        return (
            xp_einsum("ij,jk->ik", left, right)
            if contraction
            else xp_matmul(left, right)
        )

    assert product().dtype == right_dtype
    with autocast_matmul():
        expected_dtype = right_dtype
        assert product().dtype == expected_dtype
        with autocast_matmul():
            assert product().dtype == expected_dtype
        assert product().dtype == expected_dtype
    assert product().dtype == right_dtype
    assert left.dtype == jnp.float32


def test_autocast_scope_restores_after_exception() -> None:
    """A failed trace cannot alter the next model's precision policy."""
    value = jnp.asarray([[1.001, 0.12345], [0.03456, 1.003]], dtype=jnp.float32)
    reference = xp_matmul(value, value)
    with pytest.raises(ValueError, match="incompatible test shapes"):
        with autocast_matmul():
            assert xp_matmul(value, value).dtype == jnp.float32
            assert not np.array_equal(xp_matmul(value, value), reference)
            raise ValueError("incompatible test shapes")
    np.testing.assert_array_equal(xp_matmul(value, value), reference)


def test_inference_storage_restores_after_nested_training_scope() -> None:
    """Training precision does not change the enclosing inference policy."""
    value = jnp.eye(2, dtype=jnp.float32)
    with autocast_matmul(fp32_output=False):
        assert xp_matmul(value, value).dtype == jnp.bfloat16
        with autocast_matmul():
            assert xp_matmul(value, value).dtype == jnp.float32
        assert xp_matmul(value, value).dtype == jnp.bfloat16
    assert xp_matmul(value, value).dtype == jnp.float32


@pytest.mark.parametrize("adapter", [False, True])
def test_so3_bias_preserves_full_projection_activation_dtype(adapter: bool) -> None:
    """Full and scalar projections preserve FP32 AMP activations with a bias."""
    options = {
        "lmax": 1,
        "in_channels": 2,
        "out_channels": 3,
        "mlp_bias": True,
        "precision": "float32",
        "seed": 42,
    }
    layer = LoRASO3(**options, lora_rank=1) if adapter else SO3Linear(**options)
    layer.bias = np.full(3, 0.123456, dtype=np.float32)
    values = jnp.arange(16, dtype=jnp.float32).reshape(2, 4, 1, 2) / 10
    with autocast_matmul():
        assert layer(values).dtype == jnp.float32
        assert layer.call_scalar(values).dtype == jnp.float32


@pytest.mark.parametrize("contraction", [False, True])
def test_autocast_preserves_fp32_nonlinear_derivatives(contraction: bool) -> None:
    """BF16 operands feed FP32 activations and their higher-order adjoints."""
    left = jnp.linspace(-1.2, 1.1, 12, dtype=jnp.float32).reshape(4, 3)
    right = jnp.linspace(-0.9, 1.3, 15, dtype=jnp.float32).reshape(3, 5)

    def objective(value: jax.Array) -> jax.Array:
        with autocast_matmul():
            product = (
                xp_einsum("ij,jk->ik", value, right)
                if contraction
                else xp_matmul(value, right)
            )
        return jnp.sum(jnp.tanh(product / 3) ** 2)

    def reference(value: jax.Array) -> jax.Array:
        lhs = value.astype(jnp.bfloat16).astype(jnp.float32)
        rhs = right.astype(jnp.bfloat16).astype(jnp.float32)
        return jnp.sum(jnp.tanh(jnp.matmul(lhs, rhs) / 3) ** 2)

    with jax.default_matmul_precision("highest"):
        np.testing.assert_allclose(
            objective(left), reference(left), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            jax.grad(objective)(left), jax.grad(reference)(left), rtol=1e-5, atol=1e-5
        )
        direction = jnp.ones_like(left)
        actual = jax.jvp(jax.grad(objective), (left,), (direction,))[1]
        expected = jax.jvp(jax.grad(reference), (left,), (direction,))[1]
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
