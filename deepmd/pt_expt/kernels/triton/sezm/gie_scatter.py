# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN202
"""Fused zonal message build and destination reduction for the GIE head.

The geometric initial embedding forms, per edge, the message

    ``msg[e, r, c] = zonal[e, r] * radial[e, slot(r), c]``

and reduces it onto the destination node rows of the packed (l, m) layout.
The reference composition materializes the ``(E, R, C)`` message first -- at
the production shape that surface is 1.2 GB and it stays alive until the
backward has re-read it, so it also sits in the peak-residency set. The fused
forward keeps the message in registers: one node-parallel program walks the
destination CSR (stable-sorted, so the reduction order matches the eager
``index_add_`` on CPU exactly) and writes the already-padded ``(N, R+1, C)``
output with a zero scalar row. The fused backward is edge-parallel and emits
both operand gradients in one pass, so neither direction ever forms the edge
message.

The degree grouping (rows are contiguous per degree, ``2l+1`` rows each) is a
compile-time tuple like the monomial exponent table; the operator therefore
specializes per ``lmax`` at launch. The layout assumptions are checked by the
binding site, which falls back to the reference composition when the packed
row order or the degree slots are not the canonical ones.
"""

from __future__ import (
    annotations,
)

import torch
import triton
import triton.language as tl
from torch import (
    Tensor,
)
from torch.library import (
    wrap_triton,
)
from triton import (
    cdiv,
)

__all__ = [
    "GIE_SCATTER_TRITON_AVAILABLE",
    "gie_zonal_scatter",
]

GIE_SCATTER_TRITON_AVAILABLE = torch.cuda.is_available() and triton is not None


def _slot_layout(lmax: int) -> tuple[list[int], list[int], list[int]]:
    """Row counts, row bases and padded widths of the per-degree row groups."""
    counts = [2 * ell + 1 for ell in range(1, lmax + 1)]
    bases = [sum(counts[:i]) for i in range(len(counts))]
    pads = [triton.next_power_of_2(c) for c in counts]
    return counts, bases, pads


def _reference(
    zonal: Tensor,
    radial: Tensor,
    dst: Tensor,
    n_nodes: int,
) -> Tensor:
    """Eager fallback: same math, same stable within-node addition order."""
    n_row = zonal.shape[1]
    n_slot = radial.shape[1]
    counts = [2 * ell + 1 for ell in range(1, n_slot + 1)]
    slot_of_row = torch.repeat_interleave(
        torch.arange(n_slot, device=zonal.device, dtype=torch.long),
        torch.tensor(counts, device=zonal.device, dtype=torch.long),
    )
    msg = zonal.unsqueeze(-1) * radial[:, slot_of_row]  # (E, R, C)
    out = zonal.new_zeros(n_nodes, n_row + 1, radial.shape[2])
    out[:, 1:, :].index_add_(0, dst, msg)
    return out


@triton.jit
def _gie_scatter_fwd_kernel(
    zonal_ptr,
    radial_ptr,
    order_ptr,
    row_ptr_ptr,
    out_ptr,
    z_stride,
    r_stride0,
    r_stride1,
    C: tl.constexpr,
    SLOTS: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    N_ROW: tl.constexpr,
    BC: tl.constexpr,
):
    # tl.arange needs a power-of-two extent, so channels ride a padded block
    n = tl.program_id(0)
    start = tl.load(row_ptr_ptr + n).to(tl.int64)
    end = tl.load(row_ptr_ptr + n + 1).to(tl.int64)
    offs_c = tl.arange(0, BC)
    cmask = offs_c < C
    # One register accumulator per degree group (row counts are
    # compile-time constants and each group's rows are contiguous)
    r0 = tl.arange(0, P0)
    r1 = tl.arange(0, P1)
    r2 = tl.arange(0, P2)
    m0 = r0 < R0
    m1 = r1 < R1
    m2 = r2 < R2
    acc0 = tl.zeros((P0, BC), dtype=tl.float32)
    acc1 = tl.zeros((P1, BC), dtype=tl.float32)
    acc2 = tl.zeros((P2, BC), dtype=tl.float32)
    for e in range(start, end):
        # Edges inside a CSR segment map back to the original edge
        # order before zonal/radial can be read
        eo = tl.load(order_ptr + e).to(tl.int64)
        # zonal/radial may be strided views (radial_feat[:, 1:, :]),
        # so address them through their strides
        z_base = eo * z_stride
        rad_base = eo * r_stride0
        z0 = tl.load(zonal_ptr + z_base + B0 + r0, mask=m0, other=0.0)
        z1 = tl.load(zonal_ptr + z_base + B1 + r1, mask=m1, other=0.0)
        z2 = tl.load(zonal_ptr + z_base + B2 + r2, mask=m2, other=0.0)
        rd0 = tl.load(
            radial_ptr + rad_base + 0 * r_stride1 + offs_c, mask=cmask, other=0.0
        )
        rd1 = tl.load(
            radial_ptr + rad_base + 1 * r_stride1 + offs_c, mask=cmask, other=0.0
        )
        rd2 = tl.load(
            radial_ptr + rad_base + 2 * r_stride1 + offs_c, mask=cmask, other=0.0
        )
        acc0 += z0[:, None] * rd0[None, :]
        acc1 += z1[:, None] * rd1[None, :]
        acc2 += z2[:, None] * rd2[None, :]
    out_base = n * (N_ROW + 1) * C
    tl.store(
        out_ptr + out_base + 0 * C + offs_c,
        tl.zeros((BC,), dtype=tl.float32),
        mask=cmask,
    )
    tl.store(
        out_ptr + out_base + ((1 + B0 + r0) * C)[:, None] + offs_c[None, :],
        acc0,
        mask=m0[:, None] & cmask[None, :],
    )
    tl.store(
        out_ptr + out_base + ((1 + B1 + r1) * C)[:, None] + offs_c[None, :],
        acc1,
        mask=m1[:, None] & cmask[None, :],
    )
    tl.store(
        out_ptr + out_base + ((1 + B2 + r2) * C)[:, None] + offs_c[None, :],
        acc2,
        mask=m2[:, None] & cmask[None, :],
    )


@triton.jit
def _gie_scatter_bwd_kernel(
    grad_ptr,
    zonal_ptr,
    radial_ptr,
    dst_ptr,
    gz_ptr,
    gr_ptr,
    n_edge,
    g_nstride,
    g_rstride,
    z_stride,
    r_stride0,
    r_stride1,
    C: tl.constexpr,
    SLOTS: tl.constexpr,
    R0: tl.constexpr,
    R1: tl.constexpr,
    R2: tl.constexpr,
    P0: tl.constexpr,
    P1: tl.constexpr,
    P2: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    B2: tl.constexpr,
    N_ROW: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BC: tl.constexpr,
):
    # tl.arange needs a power-of-two extent, so channels ride a padded block
    pid = tl.program_id(0)
    offs_e = (pid * BLOCK_E + tl.arange(0, BLOCK_E)).to(tl.int64)
    mask_e = offs_e < n_edge
    offs_c = tl.arange(0, BC)
    cmask = offs_c < C
    r0 = tl.arange(0, P0)
    r1 = tl.arange(0, P1)
    r2 = tl.arange(0, P2)
    m0 = r0 < R0
    m1 = r1 < R1
    m2 = r2 < R2
    d = tl.load(dst_ptr + offs_e, mask=mask_e, other=0).to(tl.int64)
    # The node grad surface carries the zero scalar row at index 0 and
    # the message rows from 1; honor its strides
    g_base = (d * g_nstride + g_rstride)[:, None, None]
    g0 = tl.load(
        grad_ptr + g_base + ((B0 + r0) * C)[None, :, None] + offs_c[None, None, :],
        mask=mask_e[:, None, None] & m0[None, :, None] & cmask[None, None, :],
        other=0.0,
    )
    g1 = tl.load(
        grad_ptr + g_base + ((B1 + r1) * C)[None, :, None] + offs_c[None, None, :],
        mask=mask_e[:, None, None] & m1[None, :, None] & cmask[None, None, :],
        other=0.0,
    )
    g2 = tl.load(
        grad_ptr + g_base + ((B2 + r2) * C)[None, :, None] + offs_c[None, None, :],
        mask=mask_e[:, None, None] & m2[None, :, None] & cmask[None, None, :],
        other=0.0,
    )
    z_base = (offs_e * z_stride)[:, None]
    z0 = tl.load(
        zonal_ptr + z_base + B0 + r0[None, :],
        mask=mask_e[:, None] & m0[None, :],
        other=0.0,
    )
    z1 = tl.load(
        zonal_ptr + z_base + B1 + r1[None, :],
        mask=mask_e[:, None] & m1[None, :],
        other=0.0,
    )
    z2 = tl.load(
        zonal_ptr + z_base + B2 + r2[None, :],
        mask=mask_e[:, None] & m2[None, :],
        other=0.0,
    )
    rad_base = (offs_e * r_stride0)[:, None] + offs_c[None, :]
    rd0 = tl.load(
        radial_ptr + rad_base + 0 * r_stride1, mask=mask_e[:, None], other=0.0
    )
    rd1 = tl.load(
        radial_ptr + rad_base + 1 * r_stride1, mask=mask_e[:, None], other=0.0
    )
    rd2 = tl.load(
        radial_ptr + rad_base + 2 * r_stride1, mask=mask_e[:, None], other=0.0
    )
    gz0 = tl.sum(g0 * rd0[:, None, :], 2)
    gz1 = tl.sum(g1 * rd1[:, None, :], 2)
    gz2 = tl.sum(g2 * rd2[:, None, :], 2)
    gr0 = tl.sum(g0 * z0[:, :, None], 1)
    gr1 = tl.sum(g1 * z1[:, :, None], 1)
    gr2 = tl.sum(g2 * z2[:, :, None], 1)
    gz_base = (offs_e * N_ROW)[:, None]
    tl.store(
        gz_ptr + gz_base + B0 + r0[None, :], gz0, mask=mask_e[:, None] & m0[None, :]
    )
    tl.store(
        gz_ptr + gz_base + B1 + r1[None, :], gz1, mask=mask_e[:, None] & m1[None, :]
    )
    tl.store(
        gz_ptr + gz_base + B2 + r2[None, :], gz2, mask=mask_e[:, None] & m2[None, :]
    )
    gr_store = (offs_e * SLOTS * C)[:, None] + offs_c[None, :]
    tl.store(gr_ptr + gr_store + 0 * C, gr0, mask=mask_e[:, None] & cmask[None, :])
    tl.store(gr_ptr + gr_store + 1 * C, gr1, mask=mask_e[:, None] & cmask[None, :])
    tl.store(gr_ptr + gr_store + 2 * C, gr2, mask=mask_e[:, None] & cmask[None, :])


def _forward_impl(
    zonal: Tensor,
    radial: Tensor,
    dst: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    n_nodes: int,
    lmax: int,
) -> Tensor:
    if not (
        GIE_SCATTER_TRITON_AVAILABLE
        and zonal.is_cuda
        and zonal.dtype is torch.float32
        and lmax == 3
    ):
        return _reference(zonal, radial, dst, n_nodes)
    n_edge = zonal.shape[0]
    n_row = zonal.shape[1]
    c = radial.shape[2]
    if zonal.stride(-1) != 1:
        zonal = zonal.contiguous()
    if radial.stride(-1) != 1:
        radial = radial.contiguous()
    out = torch.empty((n_nodes, n_row + 1, c), device=zonal.device, dtype=torch.float32)
    if type(n_edge) is int and n_edge == 0:
        out.zero_()
        return out
    counts, bases, pads = _slot_layout(int(lmax))
    # Derive the grid from the row-pointer shape: under tracing this
    # keeps a symbolic expression, while a python-int argument would be
    # baked into a wrong constant (measured: grid baked to 35)
    wrap_triton(_gie_scatter_fwd_kernel)[(row_ptr.shape[0] - 1,)](
        zonal,
        radial,
        order,
        row_ptr,
        out,
        zonal.stride(0),
        radial.stride(0),
        radial.stride(1),
        C=c,
        BC=triton.next_power_of_2(c),
        SLOTS=len(counts),
        R0=counts[0],
        R1=counts[1],
        R2=counts[2],
        P0=pads[0],
        P1=pads[1],
        P2=pads[2],
        B0=bases[0],
        B1=bases[1],
        B2=bases[2],
        N_ROW=n_row,
        num_warps=4,
        num_stages=2,
    )
    return out


def _backward_impl(
    grad_out: Tensor,
    zonal: Tensor,
    radial: Tensor,
    dst: Tensor,
    lmax: int,
) -> tuple[Tensor, Tensor]:
    if not (
        GIE_SCATTER_TRITON_AVAILABLE
        and zonal.is_cuda
        and zonal.dtype is torch.float32
        and lmax == 3
    ):
        return _backward_reference(grad_out, zonal, radial, dst, int(lmax))
    n_edge, n_row = zonal.shape
    c = radial.shape[2]
    if zonal.stride(-1) != 1:
        zonal = zonal.contiguous()
    if radial.stride(-1) != 1:
        radial = radial.contiguous()
    if grad_out.stride(-1) != 1 or grad_out.stride(1) != grad_out.shape[2]:
        grad_out = grad_out.contiguous()
    gz = torch.empty((n_edge, n_row), device=zonal.device, dtype=torch.float32)
    gr = torch.empty(
        (n_edge, radial.shape[1], c), device=zonal.device, dtype=torch.float32
    )
    if type(n_edge) is int and n_edge == 0:
        return gz, gr
    counts, bases, pads = _slot_layout(int(lmax))
    block_e = 16
    wrap_triton(_gie_scatter_bwd_kernel)[(cdiv(n_edge, block_e),)](
        grad_out,
        zonal,
        radial,
        dst,
        gz,
        gr,
        n_edge,
        grad_out.stride(0),
        grad_out.stride(1),
        zonal.stride(0),
        radial.stride(0),
        radial.stride(1),
        C=c,
        BC=triton.next_power_of_2(c),
        SLOTS=len(counts),
        R0=counts[0],
        R1=counts[1],
        R2=counts[2],
        P0=pads[0],
        P1=pads[1],
        P2=pads[2],
        B0=bases[0],
        B1=bases[1],
        B2=bases[2],
        N_ROW=n_row,
        BLOCK_E=block_e,
        num_warps=4,
        num_stages=2,
    )
    return gz, gr


def _backward_reference(
    grad_out: Tensor,
    zonal: Tensor,
    radial: Tensor,
    dst: Tensor,
    lmax: int,
) -> tuple[Tensor, Tensor]:
    """Eager fallback of the backward: matches the kernel's grouping."""
    g = grad_out[:, 1:, :]  # (N, R, C) 跳过零行
    g_msg = g[dst]  # (E, R, C)
    n_slot = radial.shape[1]
    counts = [2 * ell + 1 for ell in range(1, n_slot + 1)]
    slot_of_row = torch.repeat_interleave(
        torch.arange(n_slot, device=zonal.device, dtype=torch.long),
        torch.tensor(counts, device=zonal.device, dtype=torch.long),
    )
    radial_rows = radial[:, slot_of_row]
    gz = (g_msg * radial_rows).sum(-1)
    gr = torch.zeros_like(radial)
    base = 0
    for s, cnt in enumerate(counts):
        gr[:, s, :] = (
            g_msg[:, base : base + cnt, :] * zonal[:, base : base + cnt, None]
        ).sum(1)
        base += cnt
    return gz, gr


_gie_scatter_op = torch.library.triton_op(
    "sezm_triton::gie_zonal_scatter", mutates_args=()
)(_forward_impl)

_gie_scatter_bwd_op = torch.library.triton_op(
    "sezm_triton::gie_zonal_scatter_bwd", mutates_args=()
)(_backward_impl)


@_gie_scatter_op.register_fake
def _(zonal, radial, dst, order, row_ptr, n_nodes, lmax):
    return zonal.new_empty((n_nodes, zonal.shape[1] + 1, radial.shape[2]))


@_gie_scatter_bwd_op.register_fake
def _(grad_out, zonal, radial, dst, lmax):
    return zonal.new_empty(zonal.shape), radial.new_empty(radial.shape)


def _setup_context(ctx, inputs, output):
    zonal, radial, dst, order, row_ptr, n_nodes, lmax = inputs
    # The backward reads the grad surface in the original edge order,
    # so only the three small operands need saving
    ctx.save_for_backward(zonal, radial, dst)
    ctx.lmax = lmax


def _backward(ctx, grad_out):
    zonal, radial, dst = ctx.saved_tensors
    gz, gr = _gie_scatter_bwd_op(grad_out.contiguous(), zonal, radial, dst, ctx.lmax)
    return gz, gr, None, None, None, None, None


torch.library.register_autograd(
    "sezm_triton::gie_zonal_scatter",
    _backward,
    setup_context=_setup_context,
)


def gie_zonal_scatter(
    zonal: Tensor,
    radial: Tensor,
    dst: Tensor,
    order: Tensor,
    row_ptr: Tensor,
    n_nodes: int,
    lmax: int,
) -> Tensor:
    """Public entry: fused zonal message build + destination reduction.

    Parameters
    ----------
    zonal
        Per-edge zonal coupling with shape (E, R), R = D-1 for the packed
        non-scalar rows.
    radial
        Per-edge radial features with shape (E, lmax, C), one slot per degree
        ``l >= 1``.
    dst
        Original-order destination index with shape (E,).
    order, row_ptr
        Stable destination CSR view (int64, from ``cached_edge_csr``).
    n_nodes
        Node count of the reduction target.
    lmax
        Maximum degree; the packed contiguous row layout is required.

    Returns
    -------
    torch.Tensor
        ``(N, R+1, C)`` node features with a zero scalar row, unnormalized.
    """
    return _gie_scatter_op(zonal, radial, dst, order, row_ptr, n_nodes, lmax)
