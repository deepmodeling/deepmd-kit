# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001, ANN201, ANN202, ANN204, TC002, UP035
"""Packed CuTe Wigner-D panel for the Neo K1 inference path."""

from __future__ import (
    annotations,
)

import threading
from dataclasses import (
    dataclass,
)
from typing import (
    Any,
    Callable,
)

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import (
    CUstream,
)
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

from . import (
    runtime_policy,
)
from .compile_cache import (
    device_aware_lru_cache,
)
from .k1_wigner_layout import PACKED_VALUE_COUNT as K1_PANEL_VALUES

L2_SPARSE_TERMS = 10
L3_SPARSE_TERMS = 20


def _load_dpa4_wignerd_calculator():
    """Load DeePMD's WignerDCalculator to reuse its coefficient tables."""
    from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
        WignerDCalculator,
    )

    return WignerDCalculator


def _sparsify_rows(
    coeffs: torch.Tensor, max_terms: int, threshold: float = 1.0e-12
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros(
        coeffs.shape[0], max_terms, device=coeffs.device, dtype=coeffs.dtype
    )
    indices = torch.zeros(
        coeffs.shape[0], max_terms, device=coeffs.device, dtype=torch.int32
    )
    for row in range(coeffs.shape[0]):
        nz = torch.nonzero(coeffs[row].abs() > threshold, as_tuple=False).flatten()
        if nz.numel() > max_terms:
            raise RuntimeError(
                f"sparse row {row} has {nz.numel()} terms, max_terms={max_terms}"
            )
        values[row, : nz.numel()] = coeffs[row, nz]
        indices[row, : nz.numel()] = nz.to(torch.int32)
    return values.contiguous(), indices.contiguous()


def _build_l2_l3_tables(
    dtype: torch.dtype, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    calc_cls = _load_dpa4_wignerd_calculator()
    try:
        cache = calc_cls._get_small_order_cache_cpu_fp64(3)
    except TypeError:
        cache = calc_cls._get_small_order_cache_cpu_fp64()
    c_l2 = cache["C_l2"]
    monomials_l2 = calc_cls._generate_monomials(4, 4)
    c_l2_flat = torch.zeros(
        25,
        len(monomials_l2),
        device=c_l2.device,
        dtype=c_l2.dtype,
    )
    for mono_idx, exponents in enumerate(monomials_l2):
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for d in range(4):
                        counts = (
                            int(a == 0) + int(b == 0) + int(c == 0) + int(d == 0),
                            int(a == 1) + int(b == 1) + int(c == 1) + int(d == 1),
                            int(a == 2) + int(b == 2) + int(c == 2) + int(d == 2),
                            int(a == 3) + int(b == 3) + int(c == 3) + int(d == 3),
                        )
                        if counts == exponents:
                            c_l2_flat[:, mono_idx] += c_l2[:, :, a, b, c, d].reshape(25)
    exp_l2 = torch.tensor(
        monomials_l2,
        device=c_l2.device,
        dtype=torch.int32,
    )
    c_l2_sparse, c_l2_sparse_idx = _sparsify_rows(c_l2_flat, L2_SPARSE_TERMS)
    c_l3_sparse, c_l3_sparse_idx = _sparsify_rows(cache["C_l3"], L3_SPARSE_TERMS)
    return (
        exp_l2.to(device=device, dtype=torch.int32).contiguous(),
        c_l2_sparse.to(device=device, dtype=dtype).contiguous(),
        c_l2_sparse_idx.to(device=device, dtype=torch.int32).contiguous(),
        cache["exp_l3"].to(device=device, dtype=torch.int32).contiguous(),
        c_l3_sparse.to(device=device, dtype=dtype).contiguous(),
        c_l3_sparse_idx.to(device=device, dtype=torch.int32).contiguous(),
    )


@dataclass(frozen=True)
class WignerDParams:
    q: cute.Tensor
    panel: cute.Tensor
    exp_l2: cute.Tensor
    c_l2_sparse: cute.Tensor
    c_l2_sparse_idx: cute.Tensor
    exp_l3: cute.Tensor
    c_l3_sparse: cute.Tensor
    c_l3_sparse_idx: cute.Tensor


class WignerDForward:
    def __init__(self, threads: int, dtype):
        if threads % 32 != 0:
            raise ValueError("threads must be a multiple of 32")
        self.threads = int(threads)
        self.warps = threads // 32
        self.dtype = dtype

    @cute.jit
    def rotmat(self, w, x, y, z, row, col):
        two = self.dtype(2.0)
        one = self.dtype(1.0)
        value = self.dtype(0.0)
        if row == 0 and col == 0:
            value = one - two * (y * y + z * z)
        elif row == 0 and col == 1:
            value = two * (x * y - w * z)
        elif row == 0 and col == 2:
            value = two * (x * z + w * y)
        elif row == 1 and col == 0:
            value = two * (x * y + w * z)
        elif row == 1 and col == 1:
            value = one - two * (x * x + z * z)
        elif row == 1 and col == 2:
            value = two * (y * z - w * x)
        elif row == 2 and col == 0:
            value = two * (x * z - w * y)
        elif row == 2 and col == 1:
            value = two * (y * z + w * x)
        elif row == 2 and col == 2:
            value = one - two * (x * x + y * y)
        return value

    @cute.jit
    def perm(self, idx):
        out = idx
        if idx == 0:
            out = 1
        elif idx == 1:
            out = 2
        elif idx == 2:
            out = 0
        return out

    @cute.jit
    def sign(self, idx):
        value = self.dtype(-1.0)
        if idx == 2:
            value = self.dtype(1.0)
        return value

    @cute.jit
    def d1(self, w, x, y, z, row, col):
        r = self.rotmat(w, x, y, z, self.perm(row), self.perm(col))
        return r * self.sign(row) * self.sign(col)

    @cute.jit
    def component(self, w, x, y, z, comp):
        value = w
        if comp == 1:
            value = x
        elif comp == 2:
            value = y
        elif comp == 3:
            value = z
        return value

    @cute.jit
    def pow_small(self, base, exponent):
        value = self.dtype(1.0)
        if exponent >= 1:
            value *= base
        if exponent >= 2:
            value *= base
        if exponent >= 3:
            value *= base
        if exponent >= 4:
            value *= base
        if exponent >= 5:
            value *= base
        if exponent >= 6:
            value *= base
        return value

    @cute.jit
    def monomial_l3(self, exp_l3, mono, w, x, y, z):
        value = self.dtype(1.0)
        for comp in cutlass.range_constexpr(4):
            value *= self.pow_small(
                self.component(w, x, y, z, comp),
                exp_l3[mono, comp],
            )
        return value

    @cute.kernel
    def kernel_warp_edges_panel(self, params: WignerDParams):
        tidx, _, _ = cute.arch.thread_idx()
        edge_block, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        edge_count, _ = params.q.shape
        edge = edge_block * self.warps + warp

        smem = cutlass.utils.SmemAllocator()
        l2_monomials = smem.allocate_tensor(self.dtype, self.warps * 35)
        l3_monomials = smem.allocate_tensor(self.dtype, self.warps * 84)

        if edge < edge_count:
            qw = params.q[edge, 0].to(self.dtype)
            qx = params.q[edge, 1].to(self.dtype)
            qy = params.q[edge, 2].to(self.dtype)
            qz = params.q[edge, 3].to(self.dtype)
            inv_norm = cute.rsqrt(
                qw * qw + qx * qx + qy * qy + qz * qz + self.dtype(1.0e-14)
            )
            qw = qw * inv_norm
            qx = qx * inv_norm
            qy = qy * inv_norm
            qz = qz * inv_norm

            if lane == 0:
                params.panel[edge, 0] = self.dtype(1.0).to(params.panel.element_type)

            for flat in cutlass.range(lane, 9, 32, unroll=1):
                row_slot = flat // 3
                col = flat - row_slot * 3
                row = cutlass.Int32(1)
                if row_slot == 1:
                    row = cutlass.Int32(0)
                elif row_slot == 2:
                    row = cutlass.Int32(2)
                value = self.d1(qw, qx, qy, qz, row, col)
                params.panel[edge, 1 + flat] = value.to(params.panel.element_type)

            l2_base = warp * 35
            for mono in cutlass.range(lane, 35, 32, unroll=1):
                l2_monomials[l2_base + mono] = self.monomial_l3(
                    params.exp_l2, mono, qw, qx, qy, qz
                )
            cute.arch.sync_warp()
            for flat in cutlass.range(lane, 15, 32, unroll=1):
                row_slot = flat // 5
                col = flat - row_slot * 5
                row = cutlass.Int32(2)
                if row_slot == 1:
                    row = cutlass.Int32(1)
                elif row_slot == 2:
                    row = cutlass.Int32(3)
                block_flat = row * 5 + col
                value = self.dtype(0.0)
                for term in cutlass.range_constexpr(L2_SPARSE_TERMS):
                    mono = params.c_l2_sparse_idx[block_flat, term]
                    value += (
                        params.c_l2_sparse[block_flat, term].to(self.dtype)
                        * l2_monomials[l2_base + mono]
                    )
                params.panel[edge, 10 + flat] = value.to(params.panel.element_type)

            l3_base = warp * 84
            for mono in cutlass.range(lane, 84, 32, unroll=1):
                l3_monomials[l3_base + mono] = self.monomial_l3(
                    params.exp_l3, mono, qw, qx, qy, qz
                )
            cute.arch.sync_warp()
            for flat in cutlass.range(lane, 21, 32, unroll=1):
                row_slot = flat // 7
                col = flat - row_slot * 7
                row = cutlass.Int32(3)
                if row_slot == 1:
                    row = cutlass.Int32(2)
                elif row_slot == 2:
                    row = cutlass.Int32(4)
                block_flat = row * 7 + col
                value = self.dtype(0.0)
                for term in cutlass.range_constexpr(L3_SPARSE_TERMS):
                    mono = params.c_l3_sparse_idx[block_flat, term]
                    value += (
                        params.c_l3_sparse[block_flat, term].to(self.dtype)
                        * l3_monomials[l3_base + mono]
                    )
                params.panel[edge, 25 + flat] = value.to(params.panel.element_type)


@dataclass(frozen=True)
class WignerDBwdParams:
    q: cute.Tensor
    grad_panel: cute.Tensor
    grad_q: cute.Tensor
    exp_l2: cute.Tensor
    c_l2_sparse: cute.Tensor
    c_l2_sparse_idx: cute.Tensor
    exp_l3: cute.Tensor
    c_l3_sparse: cute.Tensor
    c_l3_sparse_idx: cute.Tensor


class WignerDBackward:
    def __init__(self, threads: int, dtype):
        if threads % 32 != 0:
            raise ValueError("threads must be a multiple of 32")
        self.threads = int(threads)
        self.warps = threads // 32
        self.dtype = dtype

    @cute.jit
    def warp_sum(self, value):
        return cute.arch.warp_reduction_sum(value)

    @cute.jit
    def cta_sum(self, value, scratch, tidx):
        lane = tidx % 32
        warp = tidx // 32
        value = self.warp_sum(value)
        if lane == 0:
            scratch[warp] = value
        cute.arch.sync_threads()

        total = self.dtype(0.0)
        if tidx < self.warps:
            total = scratch[tidx]
        total = self.warp_sum(total)
        if tidx == 0:
            scratch[0] = total
        cute.arch.sync_threads()
        return scratch[0]

    @cute.jit
    def rotmat_grad(self, w, x, y, z, row, col, comp):
        two = self.dtype(2.0)
        four = self.dtype(4.0)
        value = self.dtype(0.0)
        if row == 0 and col == 0:
            if comp == 2:
                value = -four * y
            elif comp == 3:
                value = -four * z
        elif row == 0 and col == 1:
            if comp == 0:
                value = -two * z
            elif comp == 1:
                value = two * y
            elif comp == 2:
                value = two * x
            elif comp == 3:
                value = -two * w
        elif row == 0 and col == 2:
            if comp == 0:
                value = two * y
            elif comp == 1:
                value = two * z
            elif comp == 2:
                value = two * w
            elif comp == 3:
                value = two * x
        elif row == 1 and col == 0:
            if comp == 0:
                value = two * z
            elif comp == 1:
                value = two * y
            elif comp == 2:
                value = two * x
            elif comp == 3:
                value = two * w
        elif row == 1 and col == 1:
            if comp == 1:
                value = -four * x
            elif comp == 3:
                value = -four * z
        elif row == 1 and col == 2:
            if comp == 0:
                value = -two * x
            elif comp == 1:
                value = -two * w
            elif comp == 2:
                value = two * z
            elif comp == 3:
                value = two * y
        elif row == 2 and col == 0:
            if comp == 0:
                value = -two * y
            elif comp == 1:
                value = two * z
            elif comp == 2:
                value = -two * w
            elif comp == 3:
                value = two * x
        elif row == 2 and col == 1:
            if comp == 0:
                value = two * x
            elif comp == 1:
                value = two * w
            elif comp == 2:
                value = two * z
            elif comp == 3:
                value = two * y
        elif row == 2 and col == 2:
            if comp == 1:
                value = -four * x
            elif comp == 2:
                value = -four * y
        return value

    @cute.jit
    def perm(self, idx):
        out = idx
        if idx == 0:
            out = 1
        elif idx == 1:
            out = 2
        elif idx == 2:
            out = 0
        return out

    @cute.jit
    def sign(self, idx):
        value = self.dtype(-1.0)
        if idx == 2:
            value = self.dtype(1.0)
        return value

    @cute.jit
    def d1_grad(self, w, x, y, z, row, col, comp):
        r = self.rotmat_grad(w, x, y, z, self.perm(row), self.perm(col), comp)
        return r * self.sign(row) * self.sign(col)

    @cute.jit
    def component(self, w, x, y, z, comp):
        value = w
        if comp == 1:
            value = x
        elif comp == 2:
            value = y
        elif comp == 3:
            value = z
        return value

    @cute.jit
    def pow_small(self, base, exponent):
        value = self.dtype(1.0)
        if exponent >= 1:
            value *= base
        if exponent >= 2:
            value *= base
        if exponent >= 3:
            value *= base
        if exponent >= 4:
            value *= base
        if exponent >= 5:
            value *= base
        if exponent >= 6:
            value *= base
        return value

    @cute.jit
    def monomial_l3_grad(self, exp_l3, mono, w, x, y, z, comp):
        exp_comp = exp_l3[mono, comp]
        value = self.dtype(0.0)
        if exp_comp > 0:
            value = exp_comp.to(self.dtype)
            for c in cutlass.range_constexpr(4):
                exp_c = exp_l3[mono, c]
                if c == comp:
                    exp_c = exp_c - 1
                value *= self.pow_small(self.component(w, x, y, z, c), exp_c)
        return value

    @cute.kernel
    def kernel_sparse_panel(self, params: WignerDBwdParams):
        tidx, _, _ = cute.arch.thread_idx()
        edge, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        scratch = smem.allocate_tensor(self.dtype, self.warps)
        dmono_l2 = smem.allocate_tensor(self.dtype, 4 * 35)
        dmono_l3 = smem.allocate_tensor(self.dtype, 4 * 84)

        raw_w = params.q[edge, 0].to(self.dtype)
        raw_x = params.q[edge, 1].to(self.dtype)
        raw_y = params.q[edge, 2].to(self.dtype)
        raw_z = params.q[edge, 3].to(self.dtype)
        inv_norm = cute.rsqrt(
            raw_w * raw_w
            + raw_x * raw_x
            + raw_y * raw_y
            + raw_z * raw_z
            + self.dtype(1.0e-14)
        )
        w = raw_w * inv_norm
        x = raw_x * inv_norm
        y = raw_y * inv_norm
        z = raw_z * inv_norm

        for idx in cutlass.range(tidx, 4 * 35, self.threads, unroll=1):
            comp = idx // 35
            mono = idx - comp * 35
            dmono_l2[idx] = self.monomial_l3_grad(params.exp_l2, mono, w, x, y, z, comp)
        for idx in cutlass.range(tidx, 4 * 84, self.threads, unroll=1):
            comp = idx // 84
            mono = idx - comp * 84
            dmono_l3[idx] = self.monomial_l3_grad(params.exp_l3, mono, w, x, y, z, comp)
        cute.arch.sync_threads()

        gw_local = self.dtype(0.0)
        gx_local = self.dtype(0.0)
        gy_local = self.dtype(0.0)
        gz_local = self.dtype(0.0)

        for flat in cutlass.range(tidx, 9, self.threads, unroll=1):
            row_slot = flat // 3
            col = flat - row_slot * 3
            row = cutlass.Int32(1)
            if row_slot == 1:
                row = cutlass.Int32(0)
            elif row_slot == 2:
                row = cutlass.Int32(2)
            grad = params.grad_panel[edge, 1 + flat].to(self.dtype)
            gw_local += grad * self.d1_grad(w, x, y, z, row, col, 0)
            gx_local += grad * self.d1_grad(w, x, y, z, row, col, 1)
            gy_local += grad * self.d1_grad(w, x, y, z, row, col, 2)
            gz_local += grad * self.d1_grad(w, x, y, z, row, col, 3)

        for flat in cutlass.range(tidx, 15, self.threads, unroll=1):
            row_slot = flat // 5
            col = flat - row_slot * 5
            row = cutlass.Int32(2)
            if row_slot == 1:
                row = cutlass.Int32(1)
            elif row_slot == 2:
                row = cutlass.Int32(3)
            block_flat = row * 5 + col
            grad = params.grad_panel[edge, 10 + flat].to(self.dtype)
            for term in cutlass.range_constexpr(L2_SPARSE_TERMS):
                mono = params.c_l2_sparse_idx[block_flat, term]
                coeff = params.c_l2_sparse[block_flat, term].to(self.dtype)
                gw_local += grad * coeff * dmono_l2[mono]
                gx_local += grad * coeff * dmono_l2[35 + mono]
                gy_local += grad * coeff * dmono_l2[70 + mono]
                gz_local += grad * coeff * dmono_l2[105 + mono]

        for flat in cutlass.range(tidx, 21, self.threads, unroll=1):
            row_slot = flat // 7
            col = flat - row_slot * 7
            row = cutlass.Int32(3)
            if row_slot == 1:
                row = cutlass.Int32(2)
            elif row_slot == 2:
                row = cutlass.Int32(4)
            block_flat = row * 7 + col
            grad = params.grad_panel[edge, 25 + flat].to(self.dtype)
            for term in cutlass.range_constexpr(L3_SPARSE_TERMS):
                mono = params.c_l3_sparse_idx[block_flat, term]
                coeff = params.c_l3_sparse[block_flat, term].to(self.dtype)
                gw_local += grad * coeff * dmono_l3[mono]
                gx_local += grad * coeff * dmono_l3[84 + mono]
                gy_local += grad * coeff * dmono_l3[168 + mono]
                gz_local += grad * coeff * dmono_l3[252 + mono]

        gw = self.cta_sum(gw_local, scratch, tidx)
        cute.arch.sync_threads()
        gx = self.cta_sum(gx_local, scratch, tidx)
        cute.arch.sync_threads()
        gy = self.cta_sum(gy_local, scratch, tidx)
        cute.arch.sync_threads()
        gz = self.cta_sum(gz_local, scratch, tidx)

        if tidx == 0:
            dot = gw * raw_w + gx * raw_x + gy * raw_y + gz * raw_z
            inv3 = inv_norm * inv_norm * inv_norm
            params.grad_q[edge, 0] = (inv_norm * gw - raw_w * inv3 * dot).to(
                params.grad_q.element_type
            )
            params.grad_q[edge, 1] = (inv_norm * gx - raw_x * inv3 * dot).to(
                params.grad_q.element_type
            )
            params.grad_q[edge, 2] = (inv_norm * gy - raw_y * inv3 * dot).to(
                params.grad_q.element_type
            )
            params.grad_q[edge, 3] = (inv_norm * gz - raw_z * inv3 * dot).to(
                params.grad_q.element_type
            )


@cute.jit
def wignerd_panel_forward_warp_edges_jit(
    q: cute.Tensor,
    panel: cute.Tensor,
    exp_l2: cute.Tensor,
    c_l2_sparse: cute.Tensor,
    c_l2_sparse_idx: cute.Tensor,
    exp_l3: cute.Tensor,
    c_l3_sparse: cute.Tensor,
    c_l3_sparse_idx: cute.Tensor,
    threads: cutlass.Constexpr[int],
    stream: CUstream,
):
    params = WignerDParams(
        q=q,
        panel=panel,
        exp_l2=exp_l2,
        c_l2_sparse=c_l2_sparse,
        c_l2_sparse_idx=c_l2_sparse_idx,
        exp_l3=exp_l3,
        c_l3_sparse=c_l3_sparse,
        c_l3_sparse_idx=c_l3_sparse_idx,
    )
    edge_count, _ = q.shape
    warps = threads // 32
    edge_blocks = cute.ceil_div(edge_count, warps)
    WignerDForward(threads, cutlass.Float32).kernel_warp_edges_panel(params).launch(
        grid=[edge_blocks, 1, 1],
        block=[threads, 1, 1],
        stream=stream,
    )


@cute.jit
def wignerd_panel_backward_jit(
    q: cute.Tensor,
    grad_panel: cute.Tensor,
    grad_q: cute.Tensor,
    exp_l2: cute.Tensor,
    c_l2_sparse: cute.Tensor,
    c_l2_sparse_idx: cute.Tensor,
    exp_l3: cute.Tensor,
    c_l3_sparse: cute.Tensor,
    c_l3_sparse_idx: cute.Tensor,
    threads: cutlass.Constexpr[int],
    stream: CUstream,
):
    params = WignerDBwdParams(
        q=q,
        grad_panel=grad_panel,
        grad_q=grad_q,
        exp_l2=exp_l2,
        c_l2_sparse=c_l2_sparse,
        c_l2_sparse_idx=c_l2_sparse_idx,
        exp_l3=exp_l3,
        c_l3_sparse=c_l3_sparse,
        c_l3_sparse_idx=c_l3_sparse_idx,
    )
    edge_count, _ = q.shape
    WignerDBackward(threads, cutlass.Float32).kernel_sparse_panel(params).launch(
        grid=[edge_count, 1, 1],
        block=[threads, 1, 1],
        stream=stream,
    )


def compile_wignerd_panel_forward(
    threads: int,
) -> Callable:
    if threads != 32:
        raise ValueError("packed Wigner forward requires one warp per edge")
    e = cute.sym_int64()
    fake_q = make_fake_compact_tensor(cutlass.Float32, (e, 4), stride_order=(1, 0))
    fake_panel = make_fake_compact_tensor(
        cutlass.Float32,
        (e, K1_PANEL_VALUES),
        stride_order=(1, 0),
    )
    fake_exp_l2 = make_fake_compact_tensor(cutlass.Int32, (35, 4), stride_order=(1, 0))
    fake_c_l2_sparse = make_fake_compact_tensor(
        cutlass.Float32,
        (25, L2_SPARSE_TERMS),
        stride_order=(1, 0),
    )
    fake_c_l2_sparse_idx = make_fake_compact_tensor(
        cutlass.Int32,
        (25, L2_SPARSE_TERMS),
        stride_order=(1, 0),
    )
    fake_exp_l3 = make_fake_compact_tensor(cutlass.Int32, (84, 4), stride_order=(1, 0))
    fake_c_l3_sparse = make_fake_compact_tensor(
        cutlass.Float32,
        (49, L3_SPARSE_TERMS),
        stride_order=(1, 0),
    )
    fake_c_l3_sparse_idx = make_fake_compact_tensor(
        cutlass.Int32,
        (49, L3_SPARSE_TERMS),
        stride_order=(1, 0),
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        wignerd_panel_forward_warp_edges_jit,
        fake_q,
        fake_panel,
        fake_exp_l2,
        fake_c_l2_sparse,
        fake_c_l2_sparse_idx,
        fake_exp_l3,
        fake_c_l3_sparse,
        fake_c_l3_sparse_idx,
        threads,
        fake_stream,
        options="--enable-tvm-ffi",
    )


def compile_wignerd_panel_backward(
    threads: int,
) -> Callable:
    if threads % 32 != 0:
        raise ValueError("packed Wigner backward threads must be warp-aligned")
    e = cute.sym_int64()
    fake_q = make_fake_compact_tensor(cutlass.Float32, (e, 4), stride_order=(1, 0))
    fake_grad_panel = make_fake_compact_tensor(
        cutlass.Float32,
        (e, K1_PANEL_VALUES),
        stride_order=(1, 0),
    )
    fake_grad_q = make_fake_compact_tensor(cutlass.Float32, (e, 4), stride_order=(1, 0))
    fake_exp_l2 = make_fake_compact_tensor(cutlass.Int32, (35, 4), stride_order=(1, 0))
    fake_c_l2_sparse = make_fake_compact_tensor(
        cutlass.Float32,
        (25, L2_SPARSE_TERMS),
        stride_order=(1, 0),
    )
    fake_c_l2_sparse_idx = make_fake_compact_tensor(
        cutlass.Int32,
        (25, L2_SPARSE_TERMS),
        stride_order=(1, 0),
    )
    fake_exp_l3 = make_fake_compact_tensor(cutlass.Int32, (84, 4), stride_order=(1, 0))
    fake_c_l3_sparse = make_fake_compact_tensor(
        cutlass.Float32,
        (49, L3_SPARSE_TERMS),
        stride_order=(1, 0),
    )
    fake_c_l3_sparse_idx = make_fake_compact_tensor(
        cutlass.Int32,
        (49, L3_SPARSE_TERMS),
        stride_order=(1, 0),
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        wignerd_panel_backward_jit,
        fake_q,
        fake_grad_panel,
        fake_grad_q,
        fake_exp_l2,
        fake_c_l2_sparse,
        fake_c_l2_sparse_idx,
        fake_exp_l3,
        fake_c_l3_sparse,
        fake_c_l3_sparse_idx,
        threads,
        fake_stream,
        options="--enable-tvm-ffi",
    )


@device_aware_lru_cache(maxsize=16)
def _cached_wignerd_panel_forward(threads: int) -> Callable:
    return compile_wignerd_panel_forward(threads)


@device_aware_lru_cache(maxsize=16)
def _cached_wignerd_panel_backward(threads: int) -> Callable:
    return compile_wignerd_panel_backward(threads)


_TABLE_CACHE: dict[tuple[str, int], tuple[torch.Tensor, ...]] = {}
_TABLE_CACHE_LOCK = threading.Lock()


def _device_cache_key(device: torch.device) -> tuple[str, int]:
    index = -1 if device.index is None else int(device.index)
    return (device.type, index)


def _get_lmax3_tables(device: torch.device) -> tuple[torch.Tensor, ...]:
    key = _device_cache_key(device)
    tables = _TABLE_CACHE.get(key)
    if tables is None:
        with _TABLE_CACHE_LOCK:
            tables = _TABLE_CACHE.get(key)
            if tables is None:
                tables = _build_l2_l3_tables(torch.float32, device)
                _TABLE_CACHE[key] = tables
    return tables


def _wignerd_panel_impl(edge_quat: torch.Tensor) -> torch.Tensor:
    q = edge_quat.detach().contiguous()
    if q.dtype != torch.float32:
        raise TypeError(f"packed WignerD requires float32, got {q.dtype}")

    panel = torch.empty(
        q.shape[0],
        K1_PANEL_VALUES,
        device=q.device,
        dtype=q.dtype,
    )
    if q.shape[0] == 0:
        return panel

    (
        exp_l2,
        c_l2_sparse,
        c_l2_sparse_idx,
        exp_l3,
        c_l3_sparse,
        c_l3_sparse_idx,
    ) = _get_lmax3_tables(q.device)
    with torch.cuda.device(q.device):
        compiled = _cached_wignerd_panel_forward(32)
        compiled(
            q,
            panel,
            exp_l2,
            c_l2_sparse,
            c_l2_sparse_idx,
            exp_l3,
            c_l3_sparse,
            c_l3_sparse_idx,
        )
    return panel


def _wignerd_panel_bwd_impl(
    grad_panel: torch.Tensor,
    edge_quat: torch.Tensor,
) -> torch.Tensor:
    q = edge_quat.detach().contiguous()
    if tuple(grad_panel.shape) != (q.shape[0], K1_PANEL_VALUES):
        raise ValueError(
            f"packed Wigner gradient must have shape ({q.shape[0]}, {K1_PANEL_VALUES})"
        )
    if q.dtype != torch.float32:
        raise TypeError(f"packed WignerD requires float32, got {q.dtype}")
    if q.shape[0] == 0:
        return torch.empty_like(q)

    (
        exp_l2,
        c_l2_sparse,
        c_l2_sparse_idx,
        exp_l3,
        c_l3_sparse,
        c_l3_sparse_idx,
    ) = _get_lmax3_tables(q.device)
    grad_q = torch.empty_like(q)
    with torch.cuda.device(q.device):
        compiled = _cached_wignerd_panel_backward(128)
        compiled(
            q,
            grad_panel.detach().contiguous(),
            grad_q,
            exp_l2,
            c_l2_sparse,
            c_l2_sparse_idx,
            exp_l3,
            c_l3_sparse,
            c_l3_sparse_idx,
        )
    return grad_q


def _stateful_custom_op_tags() -> tuple[Any, ...] | None:
    """Tag hidden-state runner ops as unsafe for direct CUDA graph capture."""
    tag_type = getattr(getattr(torch, "_C", None), "Tag", None)
    cudagraph_unsafe = getattr(tag_type, "cudagraph_unsafe", None)
    if cudagraph_unsafe is None:
        return None
    return (cudagraph_unsafe,)


_K4_CUSTOM_OP_TAGS = _stateful_custom_op_tags()
_wignerd_panel_op = torch.library.custom_op(
    "sezm_cute::wignerd_k1_panel", mutates_args=(), tags=_K4_CUSTOM_OP_TAGS
)(_wignerd_panel_impl)
_wignerd_panel_bwd_op = torch.library.custom_op(
    "sezm_cute::wignerd_k1_panel_bwd", mutates_args=(), tags=_K4_CUSTOM_OP_TAGS
)(_wignerd_panel_bwd_impl)


@_wignerd_panel_op.register_fake
def _(edge_quat: torch.Tensor) -> torch.Tensor:
    return edge_quat.new_empty((edge_quat.shape[0], K1_PANEL_VALUES))


@_wignerd_panel_bwd_op.register_fake
def _(grad_panel: torch.Tensor, edge_quat: torch.Tensor) -> torch.Tensor:
    del grad_panel
    return torch.empty_like(edge_quat)


def _tensor_compute_capability(
    tensor: torch.Tensor,
) -> tuple[int, int] | None:
    """Resolve dispatch capability from the Wigner operand's CUDA device."""
    if tensor.device.type != "cuda":
        return None
    return tuple(torch.cuda.get_device_capability(tensor.device))


def _wignerd_panel_setup_context(
    ctx: Any,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    del output
    (edge_quat,) = inputs
    ctx.save_for_backward(edge_quat)


def _wignerd_panel_registered_backward_impl(
    ctx: Any,
    grad_panel: torch.Tensor,
) -> tuple[torch.Tensor]:
    (edge_quat,) = ctx.saved_tensors
    return (_wignerd_panel_bwd_op(grad_panel, edge_quat),)


def _wignerd_panel_backward(
    ctx: Any,
    grad_panel: torch.Tensor,
) -> tuple[torch.Tensor]:
    """Run packed K4 backward with its custom op visible to compilation."""
    return _wignerd_panel_registered_backward_impl(ctx, grad_panel)


_wignerd_panel_op.register_autograd(
    _wignerd_panel_backward,
    setup_context=_wignerd_panel_setup_context,
)


def _run_cute_wignerd_impl(
    edge_quat: torch.Tensor,
    wigner_calc: Any,
    *,
    packed_wigner: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return CuTe Wigner data for the Neo ``lmax=3`` CUDA path.

    Unsupported shapes/devices return ``None`` so the caller can use DeePMD's
    original implementation unchanged. A prevalidated strict-FP32 packed
    request returns the same ``(E,46)`` panel object in both tuple positions.
    """
    lmax = int(getattr(wigner_calc, "lmax", -1))
    if (
        not packed_wigner
        or lmax != 3
        or edge_quat.dim() != 2
        or edge_quat.shape[-1] != 4
        or not edge_quat.is_cuda
        or edge_quat.dtype != torch.float32
        or edge_quat.shape[0] == 0
    ):
        return None
    compute_capability = _tensor_compute_capability(edge_quat)
    if compute_capability is None or not runtime_policy.is_packed_wigner_enabled(
        compute_capability
    ):
        return None
    panel = _wignerd_panel_op(edge_quat)
    return panel, panel


def run_cute_wignerd(
    edge_quat: torch.Tensor,
    wigner_calc: Any,
    *,
    packed_wigner: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run K4 through its registered custom-op boundary."""
    return _run_cute_wignerd_impl(
        edge_quat,
        wigner_calc,
        packed_wigner=packed_wigner,
    )
