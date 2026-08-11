# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN202
"""Quaternion monomial basis for the Wigner-D blocks of degree two and above.

A Wigner-D block of degree ``l`` is a homogeneous polynomial of degree ``2 * l``
in the four quaternion components, so the calculator evaluates a fixed monomial
basis and follows it with one coefficient matrix product. This kernel is the
basis evaluation: register power ladders and fully unrolled products, with the
exponent table baked into the generated source.

The backward is analytic rather than a replayed product tree: differentiating a
monomial with respect to one component leaves the same monomial with that
component's exponent reduced by one and multiplied by the original exponent.

Operator boundary
-----------------
The kernel is exposed as a functional ``custom_op`` paired with an explicit
closed-form backward operator, so it survives the ``make_fx`` force-autograd
trace and can be replayed under :func:`torch.no_grad` when the frozen inference
graph runs. A closed form -- rather than a nested :func:`torch.autograd.grad` --
is required because the backward operator is dispatched below autograd during
that replay. A ``custom_op`` is opaque to Inductor: nothing inside it fuses with
the surrounding graph and its buffers are invisible to the memory planner, so
only tensors that must cross the boundary do.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..common import CUTILE_AVAILABLE, Emitter, generated_module, kernel_variant
from .tile_configs import tile_config

if TYPE_CHECKING:
    from types import ModuleType

if CUTILE_AVAILABLE:
    import cuda.tile as ct

__all__ = ["wigner_monomials"]

_HEADER = '''# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generated cuTile quaternion monomials: {count} terms of degree up to {power}."""

from typing import Annotated

import cuda.tile as ct

BigArray = Annotated[ct.Array, ct.ArrayAnnotation(index_dtype=ct.int64)]
BE = {be}
'''


def _generate(exponents: tuple[int, ...], max_power: int, block_edges: int) -> str:
    """Return the source of the monomial forward and backward kernels."""
    count = len(exponents) // 4
    source = [_HEADER.format(count=count, power=max_power, be=block_edges)]

    def emit_ladder(emit: Emitter) -> None:
        """Emit the register power ladder of each quaternion component."""
        for component in range(4):
            emit(
                f"q{component} = ct.load(quat, (edge, {component}), (BE, 1),"
                " padding_mode=ct.PaddingMode.ZERO)"
            )
            emit(f"p{component}_0 = ct.ones((BE, 1), dtype=ct.float32)")
            for power in range(1, max_power + 1):
                emit(f"p{component}_{power} = p{component}_{power - 1} * q{component}")

    emit = Emitter()
    emit("edge = ct.bid(0)")
    emit_ladder(emit)
    for term in range(count):
        powers = exponents[4 * term : 4 * term + 4]
        product = " * ".join(f"p{c}_{powers[c]}" for c in range(4))
        emit(f"ct.store(out, (edge, {term}), {product})")
    source.append(
        emit.render(
            "\n@ct.kernel",
            [
                "def monomials_forward(quat: BigArray, out: BigArray):",
                '    """Evaluate every monomial of one edge tile."""',
            ],
        )
    )

    emit = Emitter()
    emit("edge = ct.bid(0)")
    emit_ladder(emit)
    for component in range(4):
        terms = []
        for term in range(count):
            powers = exponents[4 * term : 4 * term + 4]
            if powers[component] == 0:
                continue
            factors = [
                f"p{other}_{powers[other]}" for other in range(4) if other != component
            ]
            factors.append(f"p{component}_{powers[component] - 1}")
            terms.append(
                f"ct.load(gout, (edge, {term}), (BE, 1),"
                " padding_mode=ct.PaddingMode.ZERO)"
                f" * {float(powers[component])!r} * " + " * ".join(factors)
            )
        expression = (
            " + ".join(terms) if terms else "ct.zeros((BE, 1), dtype=ct.float32)"
        )
        emit(f"ct.store(gquat, (edge, {component}), {expression})")
    source.append(
        emit.render(
            "\n@ct.kernel",
            [
                "def monomials_backward(quat: BigArray, gout: BigArray,",
                "                       gquat: BigArray):",
                '    """Accumulate the analytic quaternion gradient of every monomial."""',
            ],
        )
    )
    return "".join(source)


def _module(exponents: tuple[int, ...], max_power: int, block_edges: int) -> ModuleType:
    key = abs(hash(exponents)) % (1 << 32)
    stem = (
        f"sezm_monomials_m{len(exponents) // 4}_p{max_power}_b{block_edges}_{key:08x}"
    )
    return generated_module(stem, _generate(exponents, max_power, block_edges))


def _launch_forward(quat: Tensor, exponents: list[int], max_power: int) -> Tensor:
    """Evaluate the quaternion monomial basis.

    Parameters
    ----------
    quat : Tensor
        Per-edge quaternion, ``(E, 4)``.
    exponents : list[int]
        Flat exponent table, four entries per monomial.
    max_power : int
        Highest single-component power appearing in the table.

    Returns
    -------
    Tensor
        Monomial values, ``(E, M)``.
    """
    n_edge = quat.shape[0]
    count = len(exponents) // 4
    out = quat.new_empty((n_edge, count))
    config = tile_config("wigner_monomials")
    module = _module(tuple(exponents), max_power, config.tile)
    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(n_edge / config.tile),),
        kernel_variant(module.monomials_forward, **config.hints),
        (quat.contiguous(), out),
    )
    return out


def _launch_backward(
    grad_out: Tensor, quat: Tensor, exponents: list[int], max_power: int
) -> Tensor:
    """Return the quaternion gradient of the monomial basis, ``(E, 4)``."""
    n_edge = quat.shape[0]
    grad_quat = torch.empty_like(quat)
    config = tile_config("wigner_monomials")
    module = _module(tuple(exponents), max_power, config.tile)
    ct.launch(
        torch.cuda.current_stream(),
        (math.ceil(n_edge / config.tile),),
        kernel_variant(module.monomials_backward, **config.hints),
        (quat.contiguous(), grad_out.contiguous(), grad_quat),
    )
    return grad_quat


@torch.library.custom_op("sezm_cutile::wigner_monomials", mutates_args=())
def _monomials_op(q: Tensor, exponents: list[int], max_power: int) -> Tensor:
    return _launch_forward(q, exponents, max_power)


@_monomials_op.register_fake
def _(q, exponents, max_power):
    return q.new_empty((q.shape[0], len(exponents) // 4))


@torch.library.custom_op("sezm_cutile::wigner_monomials_bwd", mutates_args=())
def _monomials_bwd_op(
    grad_out: Tensor, q: Tensor, exponents: list[int], max_power: int
) -> Tensor:
    return _launch_backward(grad_out, q, exponents, max_power)


@_monomials_bwd_op.register_fake
def _(grad_out, q, exponents, max_power):
    return torch.empty_like(q)


def _monomials_setup(ctx, inputs, output):
    q, exponents, max_power = inputs
    ctx.save_for_backward(q)
    ctx.meta = (exponents, max_power)


def _monomials_backward_rule(ctx, grad_out):
    (q,) = ctx.saved_tensors
    exponents, max_power = ctx.meta
    return _monomials_bwd_op(grad_out, q, exponents, max_power), None, None


_monomials_op.register_autograd(
    _monomials_backward_rule, setup_context=_monomials_setup
)


def wigner_monomials(q: Tensor, exponents: list[int], max_power: int) -> Tensor:
    """Evaluate the quaternion monomial basis of the Wigner-D blocks."""
    return _monomials_op(q, exponents, max_power)
