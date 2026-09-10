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

    @triton.jit
    def _monomials_bwd2_kernel(
        g_ptr,  # (E, M) first-order output cotangent
        q_ptr,  # (E, 4)
        h_ptr,  # (E, 4) cotangent of the first-order quaternion gradient
        exp_ptr,  # (M, 4) int32 exponent table
        gg_ptr,  # (E, M) cotangent of g
        gq_ptr,  # (E, 4) Hessian contraction onto q
        n_edge,
        M: tl.constexpr,
        MAXP: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_MONO: tl.constexpr,
    ):
        """Tile the analytic second order over edges and monomials.

        Dynamic exponent loads keep the generated program independent of the
        number of monomials. Each tile reduces its Hessian-vector contribution
        over ``BLOCK_MONO`` columns before four fp32 atomic additions per edge;
        no per-monomial Hessian surface is materialized.
        """
        offs_e = (tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
        offs_m = (tl.program_id(1) * BLOCK_MONO + tl.arange(0, BLOCK_MONO)).to(tl.int64)
        mask_e = offs_e < n_edge
        mask_m = offs_m < M
        mask = mask_e[:, None] & mask_m[None, :]

        q0 = tl.load(q_ptr + offs_e * 4 + 0, mask=mask_e, other=0.0)[:, None]
        q1 = tl.load(q_ptr + offs_e * 4 + 1, mask=mask_e, other=0.0)[:, None]
        q2 = tl.load(q_ptr + offs_e * 4 + 2, mask=mask_e, other=0.0)[:, None]
        q3 = tl.load(q_ptr + offs_e * 4 + 3, mask=mask_e, other=0.0)[:, None]
        v0 = tl.load(h_ptr + offs_e * 4 + 0, mask=mask_e, other=0.0)[:, None]
        v1 = tl.load(h_ptr + offs_e * 4 + 1, mask=mask_e, other=0.0)[:, None]
        v2 = tl.load(h_ptr + offs_e * 4 + 2, mask=mask_e, other=0.0)[:, None]
        v3 = tl.load(h_ptr + offs_e * 4 + 3, mask=mask_e, other=0.0)[:, None]
        e0 = tl.load(exp_ptr + offs_m * 4 + 0, mask=mask_m, other=0)[None, :]
        e1 = tl.load(exp_ptr + offs_m * 4 + 1, mask=mask_m, other=0)[None, :]
        e2 = tl.load(exp_ptr + offs_m * 4 + 2, mask=mask_m, other=0)[None, :]
        e3 = tl.load(exp_ptr + offs_m * 4 + 3, mask=mask_m, other=0)[None, :]

        one = tl.full((BLOCK_E, BLOCK_MONO), 1.0, dtype=tl.float32)
        zero = tl.zeros((BLOCK_E, BLOCK_MONO), dtype=tl.float32)
        a = tl.where(e0 == 0, one, zero)
        b = tl.where(e1 == 0, one, zero)
        c = tl.where(e2 == 0, one, zero)
        d = tl.where(e3 == 0, one, zero)
        am1 = zero
        bm1 = zero
        cm1 = zero
        dm1 = zero
        am2 = zero
        bm2 = zero
        cm2 = zero
        dm2 = zero
        pa = one
        pb = one
        pc = one
        pd = one
        pa_prev = zero
        pb_prev = zero
        pc_prev = zero
        pd_prev = zero
        for power in tl.static_range(1, MAXP + 1):
            pa_prev2 = pa_prev
            pb_prev2 = pb_prev
            pc_prev2 = pc_prev
            pd_prev2 = pd_prev
            pa_prev = pa
            pb_prev = pb
            pc_prev = pc
            pd_prev = pd
            pa *= q0
            pb *= q1
            pc *= q2
            pd *= q3
            a = tl.where(e0 == power, pa, a)
            b = tl.where(e1 == power, pb, b)
            c = tl.where(e2 == power, pc, c)
            d = tl.where(e3 == power, pd, d)
            am1 = tl.where(e0 == power, pa_prev, am1)
            bm1 = tl.where(e1 == power, pb_prev, bm1)
            cm1 = tl.where(e2 == power, pc_prev, cm1)
            dm1 = tl.where(e3 == power, pd_prev, dm1)
            am2 = tl.where(e0 == power, pa_prev2, am2)
            bm2 = tl.where(e1 == power, pb_prev2, bm2)
            cm2 = tl.where(e2 == power, pc_prev2, cm2)
            dm2 = tl.where(e3 == power, pd_prev2, dm2)

        ef0 = e0.to(tl.float32)
        ef1 = e1.to(tl.float32)
        ef2 = e2.to(tl.float32)
        ef3 = e3.to(tl.float32)
        d0 = ef0 * am1 * ((b * c) * d)
        d1 = ef1 * bm1 * ((a * c) * d)
        d2 = ef2 * cm1 * ((a * b) * d)
        d3 = ef3 * dm1 * ((a * b) * c)

        h00 = ef0 * (ef0 - 1.0) * am2 * ((b * c) * d)
        h11 = ef1 * (ef1 - 1.0) * bm2 * ((a * c) * d)
        h22 = ef2 * (ef2 - 1.0) * cm2 * ((a * b) * d)
        h33 = ef3 * (ef3 - 1.0) * dm2 * ((a * b) * c)
        h01 = ef0 * ef1 * am1 * bm1 * (c * d)
        h02 = ef0 * ef2 * am1 * cm1 * (b * d)
        h03 = ef0 * ef3 * am1 * dm1 * (b * c)
        h12 = ef1 * ef2 * bm1 * cm1 * (a * d)
        h13 = ef1 * ef3 * bm1 * dm1 * (a * c)
        h23 = ef2 * ef3 * cm1 * dm1 * (a * b)

        g_offsets = offs_e[:, None] * M + offs_m[None, :]
        g = tl.load(g_ptr + g_offsets, mask=mask, other=0.0)
        tl.store(
            gg_ptr + g_offsets,
            v0 * d0 + v1 * d1 + v2 * d2 + v3 * d3,
            mask=mask,
        )
        partial0 = tl.sum(g * (v0 * h00 + v1 * h01 + v2 * h02 + v3 * h03), axis=1)
        partial1 = tl.sum(g * (v0 * h01 + v1 * h11 + v2 * h12 + v3 * h13), axis=1)
        partial2 = tl.sum(g * (v0 * h02 + v1 * h12 + v2 * h22 + v3 * h23), axis=1)
        partial3 = tl.sum(g * (v0 * h03 + v1 * h13 + v2 * h23 + v3 * h33), axis=1)
        tl.atomic_add(gq_ptr + offs_e * 4 + 0, partial0, mask=mask_e)
        tl.atomic_add(gq_ptr + offs_e * 4 + 1, partial1, mask=mask_e)
        tl.atomic_add(gq_ptr + offs_e * 4 + 2, partial2, mask=mask_e)
        tl.atomic_add(gq_ptr + offs_e * 4 + 3, partial3, mask=mask_e)


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


def _second_order_impl(
    grad_out: Tensor,
    q: Tensor,
    grad_grad_q: Tensor,
    exponents: list[int],
    max_power: int,
) -> tuple[Tensor, Tensor]:
    n_edge = q.shape[0]
    n_mono = len(exponents) // 4
    grad_grad_out = torch.empty_like(grad_out)
    grad_q = torch.zeros_like(q)
    if type(n_edge) is int and n_edge == 0:
        return grad_grad_out, grad_q
    exponent_table = torch.tensor(exponents, dtype=torch.int32, device=q.device)
    block_e = 16
    block_mono = 32
    wrap_triton(_monomials_bwd2_kernel)[
        (triton.cdiv(n_edge, block_e), triton.cdiv(n_mono, block_mono))
    ](
        grad_out.contiguous(),
        q.contiguous(),
        grad_grad_q.contiguous(),
        exponent_table,
        grad_grad_out,
        grad_q,
        n_edge,
        M=n_mono,
        MAXP=int(max_power),
        BLOCK_E=block_e,
        BLOCK_MONO=block_mono,
        num_warps=8,
        num_stages=1,
    )
    return grad_grad_out, grad_q


_monomials_op = torch.library.triton_op(
    "sezm_triton::wigner_monomials", mutates_args=()
)(_forward_impl)

_monomials_bwd_op = torch.library.triton_op(
    "sezm_triton::wigner_monomials_bwd", mutates_args=()
)(_backward_impl)

_monomials_bwd2_op = torch.library.triton_op(
    "sezm_triton::wigner_monomials_bwd2", mutates_args=()
)(_second_order_impl)


@_monomials_op.register_fake
def _(q, exponents, max_power):
    return q.new_empty((q.shape[0], len(exponents) // 4))


@_monomials_bwd_op.register_fake
def _(grad_out, q, exponents, max_power):
    return q.new_empty((q.shape[0], 4))


@_monomials_bwd2_op.register_fake
def _(grad_out, q, grad_grad_q, exponents, max_power):
    return grad_out.new_empty(grad_out.shape), q.new_empty(q.shape)


def _setup_context(ctx, inputs, output):
    q, exponents, max_power = inputs
    ctx.save_for_backward(q)
    ctx.exponents = exponents
    ctx.max_power = max_power


def _backward(ctx, grad_out):
    (q,) = ctx.saved_tensors
    grad_q = _monomials_bwd_op(grad_out.contiguous(), q, ctx.exponents, ctx.max_power)
    return grad_q, None, None


def _bwd_setup_context(ctx, inputs, output):
    grad_out, q, exponents, max_power = inputs
    ctx.save_for_backward(grad_out, q)
    ctx.exponents = exponents
    ctx.max_power = max_power


def _bwd_backward(ctx, grad_grad_q):
    """Second order of the monomial basis.

    Unlike the rotation and mixing operators this basis is a polynomial in
    ``q``, not a multilinear form, so the second order is a Hessian contraction
    and cannot be assembled from the first-order kernels. The eager closed form
    is differentiable to all orders and its operand is only ``(E, 4)``, so
    differentiating it is exact and costs nothing measurable beside the rotation
    kernels it feeds.
    """
    grad_out, q = ctx.saved_tensors
    if grad_grad_q is None:
        return None, None, None, None
    if _use_triton(q) and not torch.is_grad_enabled():
        grad_grad_out, grad_q_out = _monomials_bwd2_op(
            grad_out,
            q,
            grad_grad_q,
            ctx.exponents,
            ctx.max_power,
        )
        return grad_grad_out, grad_q_out, None, None
    with torch.enable_grad():
        grad_out_leaf = grad_out.detach().requires_grad_()
        q_leaf = q.detach().requires_grad_()
        grad_q = _monomials_backward_reference(
            grad_out_leaf, q_leaf, ctx.exponents, ctx.max_power
        )
        grad_grad_out, grad_q_out = torch.autograd.grad(
            grad_q,
            (grad_out_leaf, q_leaf),
            grad_grad_q,
            create_graph=torch.is_grad_enabled(),
        )
    return grad_grad_out, grad_q_out, None, None


_monomials_op.register_autograd(_backward, setup_context=_setup_context)
_monomials_bwd_op.register_autograd(_bwd_backward, setup_context=_bwd_setup_context)


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
