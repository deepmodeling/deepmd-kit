# SPDX-License-Identifier: LGPL-3.0-or-later
"""Structural parameter participation in scalar training objectives."""

from collections.abc import (
    Callable,
    Sequence,
)
from typing import (
    Any,
)

from jax.extend import (
    core,
)

from deepmd.jax.env import (
    jax,
    jnp,
    nnx,
)


def _input_dependencies(
    jaxpr: core.Jaxpr, live_outputs: Sequence[bool]
) -> tuple[bool, ...]:
    """Propagate output participation through a traced primal computation."""
    live = {
        variable
        for variable, used in zip(jaxpr.outvars, live_outputs, strict=True)
        if used and isinstance(variable, core.Var)
    }
    for equation in reversed(jaxpr.eqns):
        outputs = tuple(variable in live for variable in equation.outvars)
        if not any(outputs) or equation.primitive.name == "stop_gradient":
            continue
        inputs = (True,) * len(equation.invars)
        # Ordinary staged calls retain a one-to-one correspondence between
        # their operands and the nested graph's input variables. Following
        # these graphs avoids marking unused arguments of a jitted model.
        if equation.primitive.name in {
            "jit",
            "pjit",
            "xla_call",
            "closed_call",
            "call",
            "remat2",
        }:
            nested = equation.params.get("jaxpr", equation.params.get("call_jaxpr"))
            if nested is not None:
                nested = getattr(nested, "jaxpr", nested)
                if len(nested.invars) == len(inputs) and len(nested.outvars) == len(
                    outputs
                ):
                    inputs = _input_dependencies(nested, outputs)
        live.update(
            variable
            for variable, used in zip(equation.invars, inputs, strict=True)
            if used and isinstance(variable, core.Var)
        )
    return tuple(variable in live for variable in jaxpr.invars)


def parameter_activity(objective: Callable[[Any], jax.Array], models: Any) -> Any:
    """
    Identify parameter leaves that structurally contribute to a scalar loss.

    Parameters
    ----------
    objective : Callable
        Pure scalar objective accepting the model graph. Batch arrays and
        other arguments may be captured from an enclosing JIT trace.
    models : Any
        NNX model graph whose ``Param`` leaves are optimized.

    Returns
    -------
    Any
        Boolean scalar leaves with the same tree structure as the parameter
        state. An unused leaf is false; a participating leaf remains true
        even when its numerical gradient is zero.

    Notes
    -----
    Call this function while tracing the compiled training step. Dependency
    analysis runs in Python for that specialization, and only boolean
    constants enter the executable. It follows the primal loss instead of
    testing gradient values, so zero-initialized adapters and stationary
    parameters retain their optimizer clocks and weight decay. Unknown
    control-flow and custom-derivative primitives conservatively retain all
    participating operands.
    """
    graph, parameters, buffers = nnx.split(models, nnx.Param, ...)

    def scalar_loss(current: Any) -> jax.Array:
        return objective(nnx.merge(graph, current, buffers))

    traced = jax.make_jaxpr(scalar_loss)(parameters)
    graph_jaxpr = getattr(traced, "jaxpr", traced)
    if len(graph_jaxpr.outvars) != 1 or graph_jaxpr.outvars[0].aval.shape != ():
        raise ValueError("Parameter activity requires a scalar training objective.")
    used = _input_dependencies(graph_jaxpr, (True,))
    structure = jax.tree_util.tree_structure(parameters)
    return jax.tree_util.tree_unflatten(
        structure, [jnp.asarray(value, dtype=jnp.bool_) for value in used]
    )
