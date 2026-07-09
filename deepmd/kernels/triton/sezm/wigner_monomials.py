# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202, RUF005
"""Quaternion monomial design matrices with compile-time exponent tables.

The Wigner-D construction for degrees ``l >= 2`` evaluates, per edge, a fixed
monomial basis of the unit quaternion

    ``M[e, m] = q0^a_m * q1^b_m * q2^c_m * q3^d_m``,

with ``a_m + b_m + c_m + d_m`` equal to the kernel degree, followed by one
matrix multiply against a precomputed coefficient table.  The reference chain
(power table, ``gather``, ``prod``) materializes three ``(4, P + 1, E)``
intermediates per degree kernel, and its ``prod`` backward lowers to a
``cumprod`` scan pair -- several milliseconds per model call at typical edge
counts.  Here the exponent table is a compile-time constant: the kernel
builds the four scalar power ladders in registers and emits every monomial
(and, in the backward, its four leave-one-out derivatives
``d M_m / d q_i = e_i * q_i^{e_i - 1} * prod_{j != i} q_j^{e_j}``) as an
unrolled register product.  No intermediate ever touches DRAM.

The operator is functional (``mutates_args=()``) with a fake kernel and an
autograd formula whose backward is itself a ``triton_op``, so it composes
with the SeZM ``make_fx`` lowering and the AOTInductor freeze exactly like
the other ``sezm_triton`` operators.  The exponent table is passed as a
Python ``list[int]`` and must be extracted from the coefficient buffers in
eager context (module construction), never at trace time: a trace-time
``.tolist()`` on a tensor creates unbacked symbols and aborts export.
"""

from __future__ import (
    annotations,
)

import torch
from torch import (
    Tensor,
)
from torch.library import (
    wrap_triton,
)

__all__ = [
    "WIGNER_MONOMIALS_TRITON_AVAILABLE",
    "wigner_monomials",
]

try:
    import triton
    import triton.language as tl

    WIGNER_MONOMIALS_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    WIGNER_MONOMIALS_TRITON_AVAILABLE = False

_BLOCK_EDGES = 256


# ======================================================================
# Eager reference / fallback implementations
# ======================================================================
def _monomials_reference(q: Tensor, exponents: list[int], max_power: int) -> Tensor:
    """Eager ground truth: explicit power ladder and per-monomial products."""
    n_mono = len(exponents) // 4
    powers = [torch.ones_like(q)]
    for _ in range(max_power):
        powers.append(powers[-1] * q)
    table = torch.stack(powers, dim=1)  # (E, max_power + 1, 4)
    columns = [
        (table[:, exponents[4 * m + 0], 0] * table[:, exponents[4 * m + 1], 1])
        * (table[:, exponents[4 * m + 2], 2] * table[:, exponents[4 * m + 3], 3])
        for m in range(n_mono)
    ]
    return torch.stack(columns, dim=1)


def _monomials_backward_reference(
    grad_out: Tensor, q: Tensor, exponents: list[int], max_power: int
) -> Tensor:
    """Closed-form eager backward returning ``grad_q`` with shape (E, 4)."""
    n_mono = len(exponents) // 4
    powers = [torch.ones_like(q)]
    for _ in range(max_power):
        powers.append(powers[-1] * q)
    table = torch.stack(powers, dim=1)  # (E, max_power + 1, 4)
    grad_q = torch.zeros_like(q)
    for m in range(n_mono):
        e = exponents[4 * m : 4 * m + 4]
        g = grad_out[:, m]
        for i in range(4):
            if e[i] == 0:
                continue
            partial = g * float(e[i]) * table[:, e[i] - 1, i]
            for j in range(4):
                if j != i:
                    partial = partial * table[:, e[j], j]
            grad_q[:, i] += partial
    return grad_q


# ======================================================================
# Triton kernels
# ======================================================================
if WIGNER_MONOMIALS_TRITON_AVAILABLE:

    @triton.jit
    def _monomials_fwd_kernel(
        q_ptr,  # (E, 4) contiguous
        out_ptr,  # (E, M)
        n_edge,
        EXPS: tl.constexpr,  # flat exponent tuple (a0, b0, c0, d0, a1, ...)
        M: tl.constexpr,
        MAXP: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Register power ladders and fully unrolled monomial products."""
        pid = tl.program_id(0)
        offs = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        mask = offs < n_edge

        q0 = tl.load(q_ptr + offs * 4 + 0, mask=mask, other=0.0)
        q1 = tl.load(q_ptr + offs * 4 + 1, mask=mask, other=0.0)
        q2 = tl.load(q_ptr + offs * 4 + 2, mask=mask, other=0.0)
        q3 = tl.load(q_ptr + offs * 4 + 3, mask=mask, other=0.0)

        ones = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
        p0 = (ones,)
        p1 = (ones,)
        p2 = (ones,)
        p3 = (ones,)
        for _ in tl.static_range(MAXP):
            p0 = p0 + (p0[-1] * q0,)
            p1 = p1 + (p1[-1] * q1,)
            p2 = p2 + (p2[-1] * q2,)
            p3 = p3 + (p3[-1] * q3,)

        # ``+ 0`` forces the tuple index to a constexpr expression, which the
        # Triton frontend requires for subscripting loop-carried tuples.
        for m in tl.static_range(M):
            val = (p0[EXPS[4 * m + 0] + 0] * p1[EXPS[4 * m + 1] + 0]) * (
                p2[EXPS[4 * m + 2] + 0] * p3[EXPS[4 * m + 3] + 0]
            )
            tl.store(out_ptr + offs * M + m, val, mask=mask)

    @triton.jit
    def _monomials_bwd_kernel(
        g_ptr,  # (E, M)
        q_ptr,  # (E, 4)
        gq_ptr,  # (E, 4)
        n_edge,
        EXPS: tl.constexpr,
        M: tl.constexpr,
        MAXP: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Analytic leave-one-out backward accumulated in registers."""
        pid = tl.program_id(0)
        offs = (pid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        mask = offs < n_edge

        q0 = tl.load(q_ptr + offs * 4 + 0, mask=mask, other=0.0)
        q1 = tl.load(q_ptr + offs * 4 + 1, mask=mask, other=0.0)
        q2 = tl.load(q_ptr + offs * 4 + 2, mask=mask, other=0.0)
        q3 = tl.load(q_ptr + offs * 4 + 3, mask=mask, other=0.0)

        ones = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
        p0 = (ones,)
        p1 = (ones,)
        p2 = (ones,)
        p3 = (ones,)
        for _ in tl.static_range(MAXP):
            p0 = p0 + (p0[-1] * q0,)
            p1 = p1 + (p1[-1] * q1,)
            p2 = p2 + (p2[-1] * q2,)
            p3 = p3 + (p3[-1] * q3,)

        g0 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        g1 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        g2 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        g3 = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for m in tl.static_range(M):
            g = tl.load(g_ptr + offs * M + m, mask=mask, other=0.0)
            if EXPS[4 * m + 0] > 0:
                g0 += (g * (EXPS[4 * m + 0] + 0.0)) * (
                    (p0[EXPS[4 * m + 0] - 1] * p1[EXPS[4 * m + 1] + 0])
                    * (p2[EXPS[4 * m + 2] + 0] * p3[EXPS[4 * m + 3] + 0])
                )
            if EXPS[4 * m + 1] > 0:
                g1 += (g * (EXPS[4 * m + 1] + 0.0)) * (
                    (p0[EXPS[4 * m + 0] + 0] * p1[EXPS[4 * m + 1] - 1])
                    * (p2[EXPS[4 * m + 2] + 0] * p3[EXPS[4 * m + 3] + 0])
                )
            if EXPS[4 * m + 2] > 0:
                g2 += (g * (EXPS[4 * m + 2] + 0.0)) * (
                    (p0[EXPS[4 * m + 0] + 0] * p1[EXPS[4 * m + 1] + 0])
                    * (p2[EXPS[4 * m + 2] - 1] * p3[EXPS[4 * m + 3] + 0])
                )
            if EXPS[4 * m + 3] > 0:
                g3 += (g * (EXPS[4 * m + 3] + 0.0)) * (
                    (p0[EXPS[4 * m + 0] + 0] * p1[EXPS[4 * m + 1] + 0])
                    * (p2[EXPS[4 * m + 2] + 0] * p3[EXPS[4 * m + 3] - 1])
                )

        tl.store(gq_ptr + offs * 4 + 0, g0, mask=mask)
        tl.store(gq_ptr + offs * 4 + 1, g1, mask=mask)
        tl.store(gq_ptr + offs * 4 + 2, g2, mask=mask)
        tl.store(gq_ptr + offs * 4 + 3, g3, mask=mask)


# ======================================================================
# Dispatch, operator registration and public API
# ======================================================================
def _use_triton(tensor: Tensor) -> bool:
    return (
        WIGNER_MONOMIALS_TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype is torch.float32
    )


def _forward_impl(q: Tensor, exponents: list[int], max_power: int) -> Tensor:
    if not _use_triton(q):
        return _monomials_reference(q, exponents, int(max_power))
    n_edge = q.shape[0]
    n_mono = len(exponents) // 4
    out = torch.empty((n_edge, n_mono), device=q.device, dtype=q.dtype)
    if type(n_edge) is int and n_edge == 0:
        return out
    wrap_triton(_monomials_fwd_kernel)[(triton.cdiv(n_edge, _BLOCK_EDGES),)](
        q.contiguous(),
        out,
        n_edge,
        EXPS=tuple(exponents),
        M=n_mono,
        MAXP=int(max_power),
        BLOCK_M=_BLOCK_EDGES,
        num_warps=4,
        num_stages=2,
    )
    return out


def _backward_impl(
    grad_out: Tensor, q: Tensor, exponents: list[int], max_power: int
) -> Tensor:
    if not _use_triton(q):
        return _monomials_backward_reference(grad_out, q, exponents, int(max_power))
    n_edge = q.shape[0]
    grad_q = torch.empty((n_edge, 4), device=q.device, dtype=q.dtype)
    if type(n_edge) is int and n_edge == 0:
        return grad_q
    wrap_triton(_monomials_bwd_kernel)[(triton.cdiv(n_edge, _BLOCK_EDGES),)](
        grad_out.contiguous(),
        q.contiguous(),
        grad_q,
        n_edge,
        EXPS=tuple(exponents),
        M=len(exponents) // 4,
        MAXP=int(max_power),
        BLOCK_M=_BLOCK_EDGES,
        num_warps=4,
        num_stages=2,
    )
    return grad_q


_monomials_op = torch.library.triton_op(
    "sezm_triton::wigner_monomials", mutates_args=()
)(_forward_impl)

_monomials_bwd_op = torch.library.triton_op(
    "sezm_triton::wigner_monomials_bwd", mutates_args=()
)(_backward_impl)


@_monomials_op.register_fake
def _(q, exponents, max_power):
    return q.new_empty((q.shape[0], len(exponents) // 4))


@_monomials_bwd_op.register_fake
def _(grad_out, q, exponents, max_power):
    return q.new_empty((q.shape[0], 4))


def _setup_context(ctx, inputs, output):
    q, exponents, max_power = inputs
    ctx.save_for_backward(q)
    ctx.exponents = exponents
    ctx.max_power = max_power


def _backward(ctx, grad_out):
    (q,) = ctx.saved_tensors
    grad_q = _monomials_bwd_op(grad_out.contiguous(), q, ctx.exponents, ctx.max_power)
    return grad_q, None, None


_monomials_op.register_autograd(_backward, setup_context=_setup_context)


def wigner_monomials(q: Tensor, exponents: list[int], max_power: int) -> Tensor:
    """Evaluate a fixed quaternion monomial basis per edge.

    Parameters
    ----------
    q : Tensor
        Unit quaternions with shape (E, 4).
    exponents : list[int]
        Flattened exponent table ``(a0, b0, c0, d0, a1, ...)`` with
        ``4 * M`` entries; must be a Python list of compile-time constants
        (extracted in eager context, never at trace time).
    max_power : int
        Largest exponent appearing in the table (the power-ladder depth).

    Returns
    -------
    Tensor
        Monomial design matrix with shape (E, M), where column ``m`` is
        ``q0^a_m * q1^b_m * q2^c_m * q3^d_m``.
    """
    return _monomials_op(q, exponents, int(max_power))
