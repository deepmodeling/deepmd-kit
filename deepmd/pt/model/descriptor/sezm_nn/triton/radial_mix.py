# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""Fused Triton dynamic radial degree mixer for the SeZM/DPA4 descriptor.

This module provides a clean-room Triton implementation of the
``degree_channel`` branch of :class:`DynamicRadialDegreeMixer` for the
``mmax == 1`` reduced layout.  The eager reference applies, per edge ``e`` and
output coefficient ``o``::

    out[e, o, c] = sum_r channel_basis[r, c] * sum_i K_r[e, o, i] * x[e, i, c]

where ``K_r`` is the edge-conditioned degree kernel obtained by scattering the
projected radial features ``compact`` into a ``(reduced_dim, reduced_dim)``
matrix.  ``K_r`` is block-diagonal over the ``|m|`` groups, so for
``mmax == 1`` only a ``(lmax+1) x (lmax+1)`` block (orders ``m = 0``) and two
identical ``lmax x lmax`` blocks (orders ``m = -1`` and ``m = +1``) are
non-zero.

Design goals
------------
1. **Skip the structural zeros and the dense scratch.** The eager path
   materializes the dense kernel ``(E, reduced_dim, reduced_dim, rank)`` via a
   scatter and then contracts it with a batched ``einsum``/``bmm`` whose matrices
   are tiny (``reduced_dim <= 16``), which is inefficient on cuBLAS and wastes
   roughly two thirds of the multiply-adds on off-block zeros. The kernel
   instead reads ``compact`` directly and contracts only the structural
   non-zeros, with the channel axis vectorized and one program per edge.
2. **Match eager fp32 accuracy.** Accumulation is in fp32, matching the smooth
   potential-energy surface contract used throughout the SeZM descriptor.
3. **Compose with the SeZM ``make_fx`` lowering.** The forward and backward are
   functional ``torch.library.custom_op`` instances (``mutates_args=()``) with
   registered fake kernels and an autograd formula, so
   ``make_fx(tracing_mode="symbolic")`` captures the energy path together with
   the force autograd graph used by inference.

Inference-only contract
-----------------------
The operator is opt-in through ``DP_TRITON_INFER`` and is only used in
evaluation, where the force is obtained from ``autograd.grad(energy, coord)``.
The backward therefore returns gradients with respect to ``compact`` and
``x_local`` (both of which carry a path to the coordinates) and ``None`` for
``channel_basis``, which is a parameter and never differentiated by the force
computation.
"""

from __future__ import (
    annotations,
)

import torch
from torch import (
    Tensor,
)

__all__ = [
    "RADIAL_MIX_TRITON_AVAILABLE",
    "radial_mix_block",
    "radial_mix_reference",
]

try:
    import triton
    import triton.language as tl

    RADIAL_MIX_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    RADIAL_MIX_TRITON_AVAILABLE = False


# ======================================================================
# Eager reference / fallback implementation
# ======================================================================
def _block_layout(lmax: int) -> list[tuple[int, int, int]]:
    """Return ``(coeff_start, compact_start, num_l)`` for the ``mmax == 1`` blocks.

    The reduced m-major layout keeps, for each degree ``l``, the orders
    ``m = 0`` (the leading ``lmax + 1`` coefficients) followed by ``m = -1`` and
    ``m = +1`` (``lmax`` coefficients each). The degree kernel for the two
    signed-``m`` blocks is shared, hence the identical ``compact_start``.
    """
    num_l0 = lmax + 1
    return [
        (0, 0, num_l0),
        (num_l0, num_l0 * num_l0, lmax),
        (num_l0 + lmax, num_l0 * num_l0, lmax),
    ]


def radial_mix_reference(
    compact: Tensor, x_local: Tensor, channel_basis: Tensor, lmax: int
) -> Tensor:
    """Eager ground truth for :func:`radial_mix_block`.

    Parameters
    ----------
    compact : Tensor
        Projected radial degree kernel with shape ``(E, degree_kernel_size, R)``.
    x_local : Tensor
        Edge-local reduced features with shape ``(E, reduced_dim, C)``.
    channel_basis : Tensor
        Per-rank channel basis with shape ``(R, C)``.
    lmax : int
        Maximum spherical-harmonic degree.

    Returns
    -------
    Tensor
        Mixed features with shape ``(E, reduced_dim, C)``.
    """
    n_edge, reduced_dim, channels = x_local.shape
    out = x_local.new_zeros(n_edge, reduced_dim, channels)
    for coeff0, comp0, num_l in _block_layout(int(lmax)):
        # K[e, o, i, r] = compact[e, comp0 + i * num_l + o, r]
        block = compact[:, comp0 : comp0 + num_l * num_l, :].reshape(
            n_edge, num_l, num_l, -1
        )
        block = block.permute(0, 2, 1, 3)  # (E, o, i, R)
        x_block = x_local[:, coeff0 : coeff0 + num_l, :]  # (E, i, C)
        inner = torch.einsum("eoir,eic->eocr", block, x_block)  # (E, o, C, R)
        out[:, coeff0 : coeff0 + num_l, :] = torch.einsum(
            "eocr,rc->eoc", inner, channel_basis
        )
    return out


def _radial_mix_backward_reference(
    grad_out: Tensor, compact: Tensor, x_local: Tensor, channel_basis: Tensor, lmax: int
) -> tuple[Tensor, Tensor]:
    """Eager backward returning ``(grad_compact, grad_x_local)`` via autograd."""
    with torch.enable_grad():
        compact_req = compact.detach().requires_grad_(True)
        x_req = x_local.detach().requires_grad_(True)
        out = radial_mix_reference(compact_req, x_req, channel_basis, lmax)
        grad_compact, grad_x = torch.autograd.grad(out, [compact_req, x_req], grad_out)
    return grad_compact, grad_x


# ======================================================================
# Triton kernels (mmax == 1; LMAX and RANK are constexpr; channels vectorized)
# ======================================================================
if RADIAL_MIX_TRITON_AVAILABLE:
    # The per-edge work is tiny and memory-light, so only the warp count and
    # pipeline depth are swept, keyed on the channel width.
    _CONFIGS = [
        triton.Config({}, num_warps=1, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=1),
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
    ]
    _KEY = ["channels"]

    @triton.jit
    def _mix_fwd_block(
        edge,
        chan,
        cmask,
        x_ptr,
        k_ptr,
        cb_ptr,
        out_ptr,
        x_se,
        x_sr,
        x_sc,
        k_se,
        k_sk,
        k_sr,
        cb_sr,
        cb_sc,
        o_se,
        o_sr,
        o_sc,
        COEFF0: tl.constexpr,
        COMPACT0: tl.constexpr,
        NUM_L: tl.constexpr,
        RANK: tl.constexpr,
    ):
        """Contract one diagonal block: ``out[o] = sum_r cb[r] sum_i K_r[o,i] x[i]``."""
        for o in tl.static_range(0, NUM_L):
            acc = tl.zeros(chan.shape, dtype=tl.float32)
            for r in tl.static_range(0, RANK):
                partial = tl.zeros(chan.shape, dtype=tl.float32)
                for i in tl.static_range(0, NUM_L):
                    kval = tl.load(
                        k_ptr
                        + edge * k_se
                        + (COMPACT0 + i * NUM_L + o) * k_sk
                        + r * k_sr
                    ).to(tl.float32)
                    x_vec = tl.load(
                        x_ptr + edge * x_se + (COEFF0 + i) * x_sr + chan * x_sc,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    partial += kval * x_vec
                cb_vec = tl.load(
                    cb_ptr + r * cb_sr + chan * cb_sc, mask=cmask, other=0.0
                ).to(tl.float32)
                acc += partial * cb_vec
            tl.store(
                out_ptr + edge * o_se + (COEFF0 + o) * o_sr + chan * o_sc,
                acc.to(out_ptr.dtype.element_ty),
                mask=cmask,
            )

    @triton.autotune(configs=_CONFIGS, key=_KEY)
    @triton.jit
    def _radial_mix_fwd_kernel(
        x_ptr,
        k_ptr,
        cb_ptr,
        out_ptr,
        n_edge,
        channels,
        x_se,
        x_sr,
        x_sc,
        k_se,
        k_sk,
        k_sr,
        cb_sr,
        cb_sc,
        o_se,
        o_sr,
        o_sc,
        LMAX: tl.constexpr,
        RANK: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        edge = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < channels
        num_l0: tl.constexpr = LMAX + 1
        strides = (
            x_se,
            x_sr,
            x_sc,
            k_se,
            k_sk,
            k_sr,
            cb_sr,
            cb_sc,
            o_se,
            o_sr,
            o_sc,
        )
        # m = 0 block, then the shared m = -1 and m = +1 blocks.
        _mix_fwd_block(
            edge,
            chan,
            cmask,
            x_ptr,
            k_ptr,
            cb_ptr,
            out_ptr,
            *strides,
            0,
            0,
            num_l0,
            RANK,
        )
        _mix_fwd_block(
            edge,
            chan,
            cmask,
            x_ptr,
            k_ptr,
            cb_ptr,
            out_ptr,
            *strides,
            num_l0,
            num_l0 * num_l0,
            LMAX,
            RANK,
        )
        _mix_fwd_block(
            edge,
            chan,
            cmask,
            x_ptr,
            k_ptr,
            cb_ptr,
            out_ptr,
            *strides,
            num_l0 + LMAX,
            num_l0 * num_l0,
            LMAX,
            RANK,
        )

    @triton.jit
    def _mix_bwd_block(
        edge,
        chan,
        cmask,
        go_ptr,
        x_ptr,
        k_ptr,
        cb_ptr,
        gx_ptr,
        gk_ptr,
        go_se,
        go_sr,
        go_sc,
        x_se,
        x_sr,
        x_sc,
        k_se,
        k_sk,
        k_sr,
        cb_sr,
        cb_sc,
        gx_se,
        gx_sr,
        gx_sc,
        gk_se,
        gk_sk,
        gk_sr,
        COEFF0: tl.constexpr,
        COMPACT0: tl.constexpr,
        NUM_L: tl.constexpr,
        RANK: tl.constexpr,
    ):
        """Backward of one diagonal block.

        ``grad_x[i] = sum_r cb[r] sum_o K_r[o,i] grad_out[o]`` and
        ``grad_K_r[o,i] = sum_c cb[r,c] x[i,c] grad_out[o,c]``. Both accumulators
        are scattered with ``atomic_add`` into the zero-initialized outputs: the
        ``m = -1`` and ``m = +1`` blocks share the ``compact`` slots, so their
        contributions must sum.
        """
        for i in tl.static_range(0, NUM_L):
            grad_x = tl.zeros(chan.shape, dtype=tl.float32)
            for r in tl.static_range(0, RANK):
                cb_vec = tl.load(
                    cb_ptr + r * cb_sr + chan * cb_sc, mask=cmask, other=0.0
                ).to(tl.float32)
                partial = tl.zeros(chan.shape, dtype=tl.float32)
                for o in tl.static_range(0, NUM_L):
                    kval = tl.load(
                        k_ptr
                        + edge * k_se
                        + (COMPACT0 + i * NUM_L + o) * k_sk
                        + r * k_sr
                    ).to(tl.float32)
                    go_vec = tl.load(
                        go_ptr + edge * go_se + (COEFF0 + o) * go_sr + chan * go_sc,
                        mask=cmask,
                        other=0.0,
                    ).to(tl.float32)
                    partial += kval * go_vec
                grad_x += cb_vec * partial
            tl.atomic_add(
                gx_ptr + edge * gx_se + (COEFF0 + i) * gx_sr + chan * gx_sc,
                grad_x,
                mask=cmask,
            )
        for o in tl.static_range(0, NUM_L):
            go_vec = tl.load(
                go_ptr + edge * go_se + (COEFF0 + o) * go_sr + chan * go_sc,
                mask=cmask,
                other=0.0,
            ).to(tl.float32)
            for i in tl.static_range(0, NUM_L):
                x_vec = tl.load(
                    x_ptr + edge * x_se + (COEFF0 + i) * x_sr + chan * x_sc,
                    mask=cmask,
                    other=0.0,
                ).to(tl.float32)
                for r in tl.static_range(0, RANK):
                    cb_vec = tl.load(
                        cb_ptr + r * cb_sr + chan * cb_sc, mask=cmask, other=0.0
                    ).to(tl.float32)
                    grad_k = tl.sum(tl.where(cmask, go_vec * x_vec * cb_vec, 0.0))
                    tl.atomic_add(
                        gk_ptr
                        + edge * gk_se
                        + (COMPACT0 + i * NUM_L + o) * gk_sk
                        + r * gk_sr,
                        grad_k,
                    )

    @triton.autotune(configs=_CONFIGS, key=_KEY, reset_to_zero=["gx_ptr", "gk_ptr"])
    @triton.jit
    def _radial_mix_bwd_kernel(
        go_ptr,
        x_ptr,
        k_ptr,
        cb_ptr,
        gx_ptr,
        gk_ptr,
        n_edge,
        channels,
        go_se,
        go_sr,
        go_sc,
        x_se,
        x_sr,
        x_sc,
        k_se,
        k_sk,
        k_sr,
        cb_sr,
        cb_sc,
        gx_se,
        gx_sr,
        gx_sc,
        gk_se,
        gk_sk,
        gk_sr,
        LMAX: tl.constexpr,
        RANK: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        edge = tl.program_id(0).to(tl.int64)
        chan = tl.arange(0, BLOCK_C)
        cmask = chan < channels
        num_l0: tl.constexpr = LMAX + 1
        strides = (
            go_se,
            go_sr,
            go_sc,
            x_se,
            x_sr,
            x_sc,
            k_se,
            k_sk,
            k_sr,
            cb_sr,
            cb_sc,
            gx_se,
            gx_sr,
            gx_sc,
            gk_se,
            gk_sk,
            gk_sr,
        )
        _mix_bwd_block(
            edge,
            chan,
            cmask,
            go_ptr,
            x_ptr,
            k_ptr,
            cb_ptr,
            gx_ptr,
            gk_ptr,
            *strides,
            0,
            0,
            num_l0,
            RANK,
        )
        _mix_bwd_block(
            edge,
            chan,
            cmask,
            go_ptr,
            x_ptr,
            k_ptr,
            cb_ptr,
            gx_ptr,
            gk_ptr,
            *strides,
            num_l0,
            num_l0 * num_l0,
            LMAX,
            RANK,
        )
        _mix_bwd_block(
            edge,
            chan,
            cmask,
            go_ptr,
            x_ptr,
            k_ptr,
            cb_ptr,
            gx_ptr,
            gk_ptr,
            *strides,
            num_l0 + LMAX,
            num_l0 * num_l0,
            LMAX,
            RANK,
        )


# ======================================================================
# Triton launch wrappers
# ======================================================================
def _tile_channels(channels: int) -> int:
    """Smallest power-of-two channel tile of at least 16 covering ``channels``."""
    tile = 16
    while tile < int(channels):
        tile *= 2
    return tile


def _launch_forward(
    x_local: Tensor, compact: Tensor, channel_basis: Tensor, lmax: int
) -> Tensor:
    n_edge, reduced_dim, channels = x_local.shape
    rank = int(compact.shape[-1])
    out = torch.empty_like(x_local)
    if n_edge == 0:
        return out
    _radial_mix_fwd_kernel[(n_edge,)](
        x_local,
        compact,
        channel_basis,
        out,
        n_edge,
        channels,
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        compact.stride(0),
        compact.stride(1),
        compact.stride(2),
        channel_basis.stride(0),
        channel_basis.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        LMAX=int(lmax),
        RANK=rank,
        BLOCK_C=_tile_channels(channels),
    )
    return out


def _launch_backward(
    grad_out: Tensor,
    x_local: Tensor,
    compact: Tensor,
    channel_basis: Tensor,
    lmax: int,
) -> tuple[Tensor, Tensor]:
    n_edge, reduced_dim, channels = x_local.shape
    rank = int(compact.shape[-1])
    grad_x = torch.zeros_like(x_local)
    grad_compact = torch.zeros_like(compact)
    if n_edge == 0:
        return grad_compact, grad_x
    _radial_mix_bwd_kernel[(n_edge,)](
        grad_out.contiguous(),
        x_local,
        compact,
        channel_basis,
        grad_x,
        grad_compact,
        n_edge,
        channels,
        grad_out.stride(0),
        grad_out.stride(1),
        grad_out.stride(2),
        x_local.stride(0),
        x_local.stride(1),
        x_local.stride(2),
        compact.stride(0),
        compact.stride(1),
        compact.stride(2),
        channel_basis.stride(0),
        channel_basis.stride(1),
        grad_x.stride(0),
        grad_x.stride(1),
        grad_x.stride(2),
        grad_compact.stride(0),
        grad_compact.stride(1),
        grad_compact.stride(2),
        LMAX=int(lmax),
        RANK=rank,
        BLOCK_C=_tile_channels(channels),
    )
    return grad_compact, grad_x


# ======================================================================
# Dispatch helpers (triton on CUDA float, eager otherwise)
# ======================================================================
def _use_triton(tensor: Tensor) -> bool:
    return (
        RADIAL_MIX_TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _forward_impl(
    compact: Tensor, x_local: Tensor, channel_basis: Tensor, lmax: int
) -> Tensor:
    if not _use_triton(x_local):
        return radial_mix_reference(compact, x_local, channel_basis, lmax)
    return _launch_forward(
        x_local.contiguous(),
        compact.contiguous(),
        channel_basis.contiguous(),
        int(lmax),
    )


def _backward_impl(
    grad_out: Tensor,
    compact: Tensor,
    x_local: Tensor,
    channel_basis: Tensor,
    lmax: int,
) -> tuple[Tensor, Tensor]:
    if not _use_triton(x_local):
        return _radial_mix_backward_reference(
            grad_out, compact, x_local, channel_basis, lmax
        )
    return _launch_backward(
        grad_out,
        x_local.contiguous(),
        compact.contiguous(),
        channel_basis.contiguous(),
        int(lmax),
    )


# ======================================================================
# Functional custom ops + fake + autograd registration
# ======================================================================
_radial_mix_op = torch.library.custom_op(
    "sezm_triton::radial_mix_block", mutates_args=()
)(_forward_impl)

_radial_mix_bwd_op = torch.library.custom_op(
    "sezm_triton::radial_mix_block_bwd", mutates_args=()
)(_backward_impl)


@_radial_mix_op.register_fake
def _(compact, x_local, channel_basis, lmax):
    return torch.empty_like(x_local)


@_radial_mix_bwd_op.register_fake
def _(grad_out, compact, x_local, channel_basis, lmax):
    return torch.empty_like(compact), torch.empty_like(x_local)


def _radial_mix_setup_context(ctx, inputs, output):
    compact, x_local, channel_basis, lmax = inputs
    ctx.save_for_backward(compact, x_local, channel_basis)
    ctx.lmax = lmax


def _radial_mix_backward(ctx, grad_out):
    compact, x_local, channel_basis = ctx.saved_tensors
    grad_compact, grad_x = _radial_mix_bwd_op(
        grad_out, compact, x_local, channel_basis, ctx.lmax
    )
    # ``channel_basis`` is a parameter; the inference force differentiates only
    # w.r.t. coordinates, so its gradient is intentionally not produced.
    return grad_compact, grad_x, None, None


_radial_mix_op.register_autograd(
    _radial_mix_backward, setup_context=_radial_mix_setup_context
)


# ======================================================================
# Public API
# ======================================================================
def radial_mix_block(
    compact: Tensor, x_local: Tensor, channel_basis: Tensor, lmax: int
) -> Tensor:
    """Apply the block-diagonal dynamic radial degree mixer (``mmax == 1``).

    Computes the same operation as :func:`radial_mix_reference` while avoiding
    the dense scattered kernel and the tiny batched matmul on CUDA.

    Parameters
    ----------
    compact : Tensor
        Projected radial degree kernel with shape ``(E, degree_kernel_size, R)``.
    x_local : Tensor
        Edge-local reduced features with shape ``(E, reduced_dim, C)``.
    channel_basis : Tensor
        Per-rank channel basis with shape ``(R, C)``.
    lmax : int
        Maximum spherical-harmonic degree.

    Returns
    -------
    Tensor
        Mixed features with shape ``(E, reduced_dim, C)``.
    """
    return _radial_mix_op(compact, x_local, channel_basis, int(lmax))
