# SPDX-License-Identifier: LGPL-3.0-or-later
"""Adam transforms with per-parameter clocks for shared multi-task training."""

from typing import (
    Any,
    NamedTuple,
)

import optax

from deepmd.jax.env import (
    jax,
    jnp,
)


class AdamState(NamedTuple):
    """First and second moments and the active update count of each parameter."""

    count: Any
    mu: Any
    nu: Any


def build_adam(
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
    decoupled: bool = False,
) -> optax.GradientTransformationExtraArgs:
    """Build Adam or AdamW with explicit learning rate and active-parameter mask.

    A parameter used by several tasks has one moment trajectory. Parameters
    belonging to an inactive task retain their values, moments, and bias-
    correction clocks, matching a PyTorch parameter whose gradient is absent.

    Parameters
    ----------
    beta1, beta2 : float
        Exponential decay rates of the first and second moments.
    epsilon : float
        Positive stabilizer added after the square root of the second moment.
    weight_decay : float
        Nonnegative regularization coefficient.
    decoupled : bool
        Apply AdamW decay to the update when true, or Adam's L2 term to the
        gradient when false.

    Returns
    -------
    optax.GradientTransformationExtraArgs
        Transformation accepting ``learning_rate`` and an optional ``active``
        tree with one scalar boolean per parameter.

    Raises
    ------
    ValueError
        If a decay rate lies outside ``[0, 1)``, epsilon is nonpositive, or
        weight decay is negative.
    """
    if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise ValueError("Adam beta coefficients must lie in [0, 1).")
    if epsilon <= 0 or weight_decay < 0:
        raise ValueError("Adam epsilon must be positive and weight_decay nonnegative.")

    def init(params: Any) -> AdamState:
        return AdamState(
            jax.tree.map(lambda _: jnp.zeros((), dtype=jnp.int32), params),
            jax.tree.map(jnp.zeros_like, params),
            jax.tree.map(jnp.zeros_like, params),
        )

    def update(
        grads: Any,
        state: AdamState,
        params: Any,
        *,
        learning_rate: jnp.ndarray,
        active: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, AdamState]:
        del kwargs
        if active is None:
            active = jax.tree.map(lambda _: jnp.asarray(True), params)
        if weight_decay and not decoupled:
            grads = jax.tree.map(lambda g, p: g + weight_decay * p, grads, params)
        count = jax.tree.map(lambda n, a: n + a.astype(n.dtype), state.count, active)
        mu = jax.tree.map(
            lambda m, g, a: jnp.where(a, beta1 * m + (1 - beta1) * g, m),
            state.mu,
            grads,
            active,
        )
        nu = jax.tree.map(
            lambda v, g, a: jnp.where(a, beta2 * v + (1 - beta2) * jnp.square(g), v),
            state.nu,
            grads,
            active,
        )

        def parameter_update(
            m: jnp.ndarray,
            v: jnp.ndarray,
            n: jnp.ndarray,
            p: jnp.ndarray,
            a: jnp.ndarray,
        ) -> jnp.ndarray:
            step = jnp.maximum(n, 1)
            corrected_m = m / (1 - beta1**step)
            corrected_v = v / (1 - beta2**step)
            direction = corrected_m / (jnp.sqrt(corrected_v) + epsilon)
            if decoupled:
                direction = direction + weight_decay * p
            return jnp.where(a, -learning_rate * direction, jnp.zeros_like(p))

        updates = jax.tree.map(parameter_update, mu, nu, count, params, active)
        return updates, AdamState(count, mu, nu)

    return optax.GradientTransformationExtraArgs(init, update)
