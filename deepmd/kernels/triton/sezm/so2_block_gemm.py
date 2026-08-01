# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""Triton ``BN=64`` block-diagonal fp32 GEMM for the SeZM/DPA4 ``SO2Linear``.

The ``SO2Linear`` mixing contracts an activation ``x`` of shape ``(F, E, K)``
with a per-focus block-diagonal weight ``W`` of shape ``(F, K, N)``. The eager
path (``_block_diagonal_matmul``) slices the diagonal ``|m|`` blocks and issues a
``torch.bmm`` per block, concatenating the outputs. Two structural costs remain:

* the assembled weight is presented as a *strided* view
  (``permute(1, 0, 2)`` of the stored ``(K, F, N)`` parameter), and the block
  concatenation materializes a fresh output; and
* the pow2 column tiling that cuBLAS/Triton default to wastes ~25% on the
  ``N = 192`` block (192 rounds up to 256).

This module drives the block-diagonal contraction with one Triton launch per
diagonal ``|m|`` block (the block dims are ``constexpr`` so the contraction loop
is statically sized and fully pipelined). Each launch

1. consumes the strided ``(F, K, N)`` weight and the ``(F, E, K)`` activation
   *without any contiguity copy* (all access is via strides), streaming only its
   block's contraction range and never touching the structural off-``|m|`` zeros
   or concatenating a fresh output, and
2. tiles the output ``N`` axis at exactly ``BN = 64`` -- a divisor of both 128
   and 192 -- so no column is padded.

Every ``tl.dot`` runs with ``input_precision="ieee"`` (true IEEE fp32, no TF32),
matching the smooth potential-energy-surface contract of the descriptor.

Composability
-------------
The forward and backward are functional ``torch.library.triton_op`` instances
(``mutates_args=()``) with registered fake kernels and an autograd formula, so
``make_fx(tracing_mode="symbolic") -> aot_module_simplified -> Inductor`` captures
the energy path together with the force autograd graph. ``triton_op`` +
``wrap_triton`` (vs ``custom_op``) lets Inductor see through to the Triton kernel
and bake the cubin into the AOTInductor ``.pt2``, exactly as
``so2_rotation.py`` / ``radial_mix.py`` do.

Inference-only contract
-----------------------
The operator is opt-in and only used in evaluation, where the force is obtained
from ``autograd.grad(energy, coord)``. The block-diagonal weight is a parameter
(never a function of the coordinates), so the backward returns the gradient
w.r.t. the activation ``x`` -- which carries the coordinate path -- and ``None``
for the weight, mirroring the parameter handling in ``radial_mix.py``.
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

# Activation-gradient backend for the force path.
#
# ``False`` (default): the backward uses the eager per-block ``bmm``, whose
# ``grad_out`` slices the Inductor memory planner reads through
# ``reinterpret_tensor`` (no copy), so the compiled force graph reuses the
# edge-sized gradient buffers and peak memory tracks the committed baseline. The
# Triton backward is faster in isolation (~1.4x over cuBLAS) but issues one
# launch per diagonal block; Inductor materializes a separate ``grad_out`` copy
# for each such consumer, inflating the compiled force-graph peak by ~13% for a
# ~1-2% end-to-end gain. The trade is unfavorable, so it remains opt-in.
_TRITON_BACKWARD = False

__all__ = [
    "SO2_BLOCK_GEMM_TRITON_AVAILABLE",
    "block_diag_gemm",
    "block_diag_gemm_reference",
    "slices_supported",
]

try:
    import triton
    import triton.language as tl

    SO2_BLOCK_GEMM_TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without triton
    SO2_BLOCK_GEMM_TRITON_AVAILABLE = False


# The BM=128, BN=64, BK=32 / 8-warp / 3-stage tile is the cuBLAS-beating config
# for these block shapes. BN is *fixed* at 64 -- the divisor of 128 and 192 that
# removes the pow2 column padding -- because the N-tile -> block mapping assumes
# each tile lies wholly inside one diagonal block.
_BM = 128
_BN = 64
_BK = 32
_NUM_WARPS = 8
_NUM_STAGES = 3


# ======================================================================
# Eager reference / fallback implementation
# ======================================================================
def block_diag_gemm_reference(
    x_flat: Tensor, weight: Tensor, slices: list[tuple[int, int, int, int]]
) -> Tensor:
    """Eager ground truth: per-block ``bmm`` on the strided operands + concat.

    Parameters
    ----------
    x_flat : Tensor
        Activation with shape ``(F, E, K)``.
    weight : Tensor
        Block-diagonal weight presented as ``(F, K, N)`` (a strided view).
    slices : list of (int, int, int, int)
        The ``(in0, in1, out0, out1)`` diagonal blocks in m-major order.

    Returns
    -------
    Tensor
        Output with shape ``(F, E, N)``.
    """
    blocks = [
        torch.bmm(x_flat[:, :, in0:in1], weight[:, in0:in1, out0:out1])
        for in0, in1, out0, out1 in slices
    ]
    return torch.cat(blocks, dim=-1)


def _block_diag_gemm_bwd_reference(
    grad_out: Tensor, weight: Tensor, slices: list[tuple[int, int, int, int]]
) -> Tensor:
    """Eager backward: ``grad_x = grad_out @ W^T`` per diagonal block.

    Only the activation gradient is produced; the weight is a parameter and is
    never differentiated on the inference force path.

    Parameters
    ----------
    grad_out : Tensor
        Upstream gradient with shape ``(F, E, N)``.
    weight : Tensor
        Block-diagonal weight presented as ``(F, K, N)`` (a strided view).
    slices : list of (int, int, int, int)
        The ``(in0, in1, out0, out1)`` diagonal blocks in m-major order.

    Returns
    -------
    Tensor
        Activation gradient with shape ``(F, E, K)``.
    """
    n_focus, n_edge = grad_out.shape[0], grad_out.shape[1]
    k_total = weight.shape[1]
    grad_x = grad_out.new_zeros(n_focus, n_edge, k_total)
    for in0, in1, out0, out1 in slices:
        grad_x[:, :, in0:in1] = torch.bmm(
            grad_out[:, :, out0:out1],
            weight[:, in0:in1, out0:out1].transpose(1, 2),
        )
    return grad_x


# ======================================================================
# Triton kernels (one launch per diagonal block; the block dims KLEN / NLEN are
# constexpr so the contraction loop is statically sized and fully pipelined)
# ======================================================================
if SO2_BLOCK_GEMM_TRITON_AVAILABLE:

    @triton.jit
    def _block_gemm_fwd_kernel(
        a_ptr,
        w_ptr,
        c_ptr,
        n_edge,
        k0,
        n0,
        sab,
        sae,
        sak,
        swb,
        swk,
        swn,
        scb,
        sce,
        scn,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        KLEN: tl.constexpr,
        NLEN: tl.constexpr,
    ):
        """One diagonal block: ``C[:, :, n0:n0+NLEN] = X[:, :, k0:k0+KLEN] @ W``.

        The strided operands (``x``, the permuted ``weight``, and the whole
        output) are passed as full buffers with the block offsets ``k0 / n0``
        applied inside the kernel; this keeps every access on the parent buffers
        (no ``reinterpret_tensor`` slice, which Inductor would otherwise clone on
        the strided focus-major layout). ``N`` is tiled at ``BN`` (a divisor of
        ``NLEN``) so no column is padded; the ``KLEN`` contraction is statically
        unrolled and pipelined.
        """
        pid = tl.program_id(0)
        bid = tl.program_id(1).to(tl.int64)
        num_pid_n: tl.constexpr = NLEN // BN
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        offs_n = (n0 + pid_n * BN + tl.arange(0, BN)).to(tl.int64)
        offs_k = tl.arange(0, BK)

        a_ptrs = (
            a_ptr + bid * sab + (offs_m[:, None] * sae + (k0 + offs_k[None, :]) * sak)
        )
        w_ptrs = (
            w_ptr + bid * swb + ((k0 + offs_k[:, None]) * swk + offs_n[None, :] * swn)
        )

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        m_mask = offs_m[:, None] < n_edge
        for _ in range(0, KLEN, BK):
            a = tl.load(a_ptrs, mask=m_mask, other=0.0)
            w = tl.load(w_ptrs)
            acc = tl.dot(a, w, acc, input_precision="ieee")
            a_ptrs += BK * sak
            w_ptrs += BK * swk

        c_ptrs = c_ptr + bid * scb + (offs_m[:, None] * sce + offs_n[None, :] * scn)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=m_mask)

    @triton.jit
    def _block_gemm_dx_kernel(
        go_ptr,
        wt_ptr,
        gx_ptr,
        n_edge,
        k0,
        n0,
        sgb,
        sge,
        sgn,
        swb,
        swn,
        swk,
        sxb,
        sxe,
        sxk,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
        KLEN: tl.constexpr,
        NLEN: tl.constexpr,
    ):
        """One diagonal block of the activation gradient ``GX = GO @ W^T``.

        Contraction over the block's ``NLEN`` output axis: ``grad_x[e, k] =
        sum_n grad_out[e, n] W[k, n]``. The full ``grad_out`` / ``grad_x`` buffers
        are addressed with the block offsets ``n0 / k0`` applied inside the kernel
        (no slice views, which Inductor clones on the strided focus-major grad).
        ``weight_t`` is ``Wt[n, k]`` with a contiguous ``k`` axis so both operands
        load coalesced with no register transpose.
        """
        pid = tl.program_id(0)
        bid = tl.program_id(1).to(tl.int64)
        num_pid_k: tl.constexpr = KLEN // BN
        pid_m = pid // num_pid_k
        pid_k = pid % num_pid_k

        offs_m = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)
        offs_k = (k0 + pid_k * BN + tl.arange(0, BN)).to(tl.int64)
        offs_c = tl.arange(0, BK)  # contraction over the block's N axis

        go_ptrs = (
            go_ptr + bid * sgb + (offs_m[:, None] * sge + (n0 + offs_c[None, :]) * sgn)
        )
        wt_ptrs = (
            wt_ptr + bid * swb + ((n0 + offs_c[:, None]) * swn + offs_k[None, :] * swk)
        )

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        m_mask = offs_m[:, None] < n_edge
        for _ in range(0, NLEN, BK):
            go = tl.load(go_ptrs, mask=m_mask, other=0.0)
            wt = tl.load(wt_ptrs)
            acc = tl.dot(go, wt, acc, input_precision="ieee")
            go_ptrs += BK * sgn
            wt_ptrs += BK * swn

        gx_ptrs = gx_ptr + bid * sxb + (offs_m[:, None] * sxe + offs_k[None, :] * sxk)
        tl.store(gx_ptrs, acc.to(gx_ptr.dtype.element_ty), mask=m_mask)


# ======================================================================
# Launch wrappers
# ======================================================================
def _has_no_edges(n_edge) -> bool:
    """Return true only for eager zero-edge calls; never guards a SymInt."""
    return type(n_edge) is int and n_edge == 0


def _launch_forward(
    x_flat: Tensor, weight: Tensor, slices: list[tuple[int, int, int, int]], n_out: int
) -> Tensor:
    n_focus, n_edge, _ = x_flat.shape
    out = torch.empty(
        (n_focus, n_edge, n_out), dtype=x_flat.dtype, device=x_flat.device
    )
    if _has_no_edges(n_edge):
        return out
    m_tiles = triton.cdiv(n_edge, _BM)
    for in0, in1, out0, out1 in slices:
        klen, nlen = in1 - in0, out1 - out0
        wrap_triton(_block_gemm_fwd_kernel)[(m_tiles * (nlen // _BN), n_focus)](
            x_flat,
            weight,
            out,
            n_edge,
            in0,
            out0,
            x_flat.stride(0),
            x_flat.stride(1),
            x_flat.stride(2),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            BM=_BM,
            BN=_BN,
            BK=_BK,
            KLEN=klen,
            NLEN=nlen,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
    return out


def _launch_backward_dx(
    grad_out: Tensor,
    weight: Tensor,
    slices: list[tuple[int, int, int, int]],
    k_out: int,
) -> Tensor:
    n_focus, n_edge, _ = grad_out.shape
    grad_x = torch.empty(
        (n_focus, n_edge, k_out), dtype=grad_out.dtype, device=grad_out.device
    )
    if _has_no_edges(n_edge):
        return grad_x
    # Pre-transpose the (small, constant) weight to (F, N, K) with a contiguous
    # K axis so the N-contraction kernel loads it coalesced (see kernel doc).
    # The weight is a parameter, so Inductor constant-folds this in the frozen
    # graph; eagerly it is a sub-megabyte copy.
    weight_t = weight.transpose(1, 2).contiguous()
    m_tiles = triton.cdiv(n_edge, _BM)
    for in0, in1, out0, out1 in slices:
        klen, nlen = in1 - in0, out1 - out0
        wrap_triton(_block_gemm_dx_kernel)[(m_tiles * (klen // _BN), n_focus)](
            grad_out,
            weight_t,
            grad_x,
            n_edge,
            in0,
            out0,
            grad_out.stride(0),
            grad_out.stride(1),
            grad_out.stride(2),
            weight_t.stride(0),
            weight_t.stride(1),
            weight_t.stride(2),
            grad_x.stride(0),
            grad_x.stride(1),
            grad_x.stride(2),
            BM=_BM,
            BN=_BN,
            BK=_BK,
            KLEN=klen,
            NLEN=nlen,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
        )
    return grad_x


# ======================================================================
# Dispatch helpers (triton on CUDA float, eager otherwise)
# ======================================================================
def _use_triton(tensor: Tensor) -> bool:
    return (
        SO2_BLOCK_GEMM_TRITON_AVAILABLE
        and tensor.is_cuda
        and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )


def _unflatten_slices(slices_flat: list[int]) -> list[tuple[int, int, int, int]]:
    """Rebuild ``(in0, in1, out0, out1)`` blocks from the flat ``list[int]``.

    ``triton_op`` schema inference accepts ``list[int]`` but not
    ``list[list[int]]``, so the block table is carried as groups of four.
    """
    return [
        (slices_flat[i], slices_flat[i + 1], slices_flat[i + 2], slices_flat[i + 3])
        for i in range(0, len(slices_flat), 4)
    ]


def _forward_impl(x_flat: Tensor, weight: Tensor, slices_flat: list[int]) -> Tensor:
    slices = _unflatten_slices(slices_flat)
    if not _use_triton(x_flat):
        return block_diag_gemm_reference(x_flat, weight, slices)
    n_out = max(out1 for _, _, _, out1 in slices)
    return _launch_forward(x_flat, weight, slices, n_out)


def _backward_impl(grad_out: Tensor, weight: Tensor, slices_flat: list[int]) -> Tensor:
    slices = _unflatten_slices(slices_flat)
    if not _TRITON_BACKWARD or not _use_triton(grad_out):
        return _block_diag_gemm_bwd_reference(grad_out, weight, slices)
    k_out = max(in1 for _, in1, _, _ in slices)
    return _launch_backward_dx(grad_out, weight, slices, k_out)


# ======================================================================
# Functional triton_op + fake + autograd registration
# ======================================================================
_bd_gemm_op = torch.library.triton_op(
    "sezm_triton::so2_block_diag_gemm", mutates_args=()
)(_forward_impl)

_bd_gemm_bwd_op = torch.library.triton_op(
    "sezm_triton::so2_block_diag_gemm_bwd", mutates_args=()
)(_backward_impl)


@_bd_gemm_op.register_fake
def _(x_flat, weight, slices_flat):
    n_out = max(slices_flat[3::4])
    return x_flat.new_empty((x_flat.shape[0], x_flat.shape[1], n_out))


@_bd_gemm_bwd_op.register_fake
def _(grad_out, weight, slices_flat):
    k_total = max(slices_flat[1::4])
    return grad_out.new_empty((grad_out.shape[0], grad_out.shape[1], k_total))


def _bd_gemm_setup_context(ctx, inputs, output):
    x_flat, weight, slices_flat = inputs
    ctx.save_for_backward(weight)
    ctx.slices_flat = slices_flat


def _bd_gemm_backward(ctx, grad_out):
    (weight,) = ctx.saved_tensors
    grad_x = _bd_gemm_bwd_op(grad_out, weight, ctx.slices_flat)
    # weight is a parameter (never a function of the coordinates); the inference
    # force differentiates only w.r.t. the activation, so its gradient is not
    # produced. ``slices_flat`` is a static block table.
    return grad_x, None, None


_bd_gemm_op.register_autograd(_bd_gemm_backward, setup_context=_bd_gemm_setup_context)


# ======================================================================
# Public API
# ======================================================================
def slices_supported(
    slices: list[tuple[int, int, int, int]], block_n: int = _BN
) -> bool:
    """Return whether every block boundary/width aligns to ``block_n``.

    The BN-tiled kernel maps each ``BN``-wide output (input) tile to a single
    diagonal block, which requires every block edge and width to be a multiple of
    ``block_n`` so no tile straddles two blocks. Callers gate the Triton path on
    this (e.g. an even ``lmax`` makes the ``m=0`` block width ``(lmax+1)*C`` an
    odd count and may break alignment); unsupported layouts fall back to eager.

    Parameters
    ----------
    slices : list of (int, int, int, int)
        The ``(in0, in1, out0, out1)`` diagonal blocks in m-major order.
    block_n : int
        Column tile width; every block edge and width must be a multiple of it.

    Returns
    -------
    bool
        ``True`` when every block boundary and width is a multiple of ``block_n``.
    """
    return all(
        edge % block_n == 0
        for in0, in1, out0, out1 in slices
        for edge in (in0, in1, out0, out1)
    )


def block_diag_gemm(
    x_flat: Tensor,
    weight: Tensor,
    slices: list[tuple[int, int, int, int]],
) -> Tensor:
    """Apply the ``BN=64`` block-diagonal GEMM ``(F, E, K) -> (F, E, N)``.

    Computes the same result as :func:`block_diag_gemm_reference` while avoiding
    both the block concatenation and any contiguity copy of the strided weight:
    one Triton launch per diagonal block streams only that block's contraction
    range from the strided operands, with the output ``N`` axis tiled at 64.

    Parameters
    ----------
    x_flat : Tensor
        Activation with shape ``(F, E, K)``.
    weight : Tensor
        Block-diagonal weight presented as ``(F, K, N)`` (may be strided).
    slices : list of (int, int, int, int)
        The ``(in0, in1, out0, out1)`` diagonal blocks in m-major order.

    Returns
    -------
    Tensor
        Output with shape ``(F, E, N)``.
    """
    slices_flat = [int(v) for s in slices for v in s]
    return _bd_gemm_op(x_flat, weight, slices_flat)
