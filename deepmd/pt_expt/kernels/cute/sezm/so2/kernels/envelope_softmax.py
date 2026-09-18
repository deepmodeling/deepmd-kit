# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""CuTe DSL kernel for envelope-gated segmented softmax.

The input contract is intentionally strict: destination edges must already be
sorted and represented by CSR row pointers before calling the CuTe kernel.
"""

# ruff: noqa: ANN001, ANN201, ANN204, TC002

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
)

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cuda.bindings.driver import (
    CUstream,
)
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

from ...compile_cache import (
    device_aware_lru_cache,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


@dataclass(frozen=True)
class EnvelopeSoftmaxFwdParams:
    threads: int
    logits: cute.Tensor
    edge_gate: cute.Tensor
    dst_ptr: cute.Tensor
    z_bias_raw: cute.Tensor
    out: cute.Tensor
    group_max: cute.Tensor
    denom: cute.Tensor


class EnvelopeSoftmaxForward:
    def __init__(self, threads: int):
        if threads % 32 != 0:
            raise ValueError("threads must be a multiple of 32")
        self.threads = threads
        self.warps = threads // 32
        self.dtype = cutlass.Float32

    @cute.jit
    def warp_sum(self, value):
        return cute.arch.warp_reduction_sum(value)

    @cute.jit
    def warp_max(self, value):
        return cute.arch.warp_reduction_max(value)

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
    def cta_max(self, value, scratch, tidx):
        lane = tidx % 32
        warp = tidx // 32
        value = self.warp_max(value)
        if lane == 0:
            scratch[warp] = value
        cute.arch.sync_threads()

        neg_large = self.dtype(-3.4028234663852886e38)
        total = neg_large
        if tidx < self.warps:
            total = scratch[tidx]
        total = self.warp_max(total)
        if tidx == 0:
            scratch[0] = total
        cute.arch.sync_threads()
        return scratch[0]

    @cute.jit
    def softplus(self, value):
        zero = self.dtype(0.0)
        positive = cute.arch.fmax(value, zero)
        magnitude = cute.arch.fmax(value, -value)
        return positive + cute.log(self.dtype(1.0) + cute.exp(-magnitude))

    @cute.kernel
    def kernel(self, params: EnvelopeSoftmaxFwdParams, eps: cutlass.Constexpr[float]):
        tidx, _, _ = cute.arch.thread_idx()
        node, group, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        scratch = smem.allocate_tensor(self.dtype, self.warps)

        lo = params.dst_ptr[node]
        hi = params.dst_ptr[node + 1]
        null_mass = self.softplus(params.z_bias_raw[group].to(self.dtype)) + self.dtype(
            eps
        )
        local_max = cute.log(null_mass)
        for edge in cutlass.range(lo + tidx, hi, self.threads, unroll=1):
            gate = params.edge_gate[edge].to(self.dtype)
            if gate < self.dtype(0.0):
                gate = self.dtype(0.0)
            if gate > self.dtype(0.0):
                value = params.logits[edge, group].to(self.dtype) + self.dtype(
                    2.0
                ) * cute.log(gate)
                if value > local_max:
                    local_max = value

        group_max = self.cta_max(local_max, scratch, tidx)
        # Every warp must consume scratch[0] before cta_sum reuses it.
        cute.arch.sync_threads()

        local_sum = self.dtype(0.0)
        for edge in cutlass.range(lo + tidx, hi, self.threads, unroll=1):
            gate = params.edge_gate[edge].to(self.dtype)
            if gate < self.dtype(0.0):
                gate = self.dtype(0.0)
            if gate > self.dtype(0.0):
                effective_logit = params.logits[edge, group].to(
                    self.dtype
                ) + self.dtype(2.0) * cute.log(gate)
                local_sum += cute.exp(effective_logit - group_max)

        denom_sum = self.cta_sum(local_sum, scratch, tidx)
        denom = denom_sum + null_mass * cute.exp(-group_max)

        if tidx == 0:
            params.group_max[node, group] = group_max.to(params.group_max.element_type)
            params.denom[node, group] = denom.to(params.denom.element_type)
        cute.arch.sync_threads()

        for edge in cutlass.range(lo + tidx, hi, self.threads, unroll=1):
            gate = params.edge_gate[edge].to(self.dtype)
            if gate < self.dtype(0.0):
                gate = self.dtype(0.0)
            alpha = self.dtype(0.0)
            if gate > self.dtype(0.0):
                effective_logit = params.logits[edge, group].to(
                    self.dtype
                ) + self.dtype(2.0) * cute.log(gate)
                num = cute.exp(effective_logit - group_max)
                alpha = num / denom
            params.out[edge, group] = alpha.to(params.out.element_type)


@cute.jit
def envelope_softmax_forward_jit(
    logits: cute.Tensor,
    edge_gate: cute.Tensor,
    dst_ptr: cute.Tensor,
    z_bias_raw: cute.Tensor,
    out: cute.Tensor,
    group_max: cute.Tensor,
    denom: cute.Tensor,
    threads: cutlass.Constexpr[int],
    eps: cutlass.Constexpr[float],
    stream: CUstream,
):
    params = EnvelopeSoftmaxFwdParams(
        threads=threads,
        logits=logits,
        edge_gate=edge_gate,
        dst_ptr=dst_ptr,
        z_bias_raw=z_bias_raw,
        out=out,
        group_max=group_max,
        denom=denom,
    )
    n_nodes, groups = denom.shape
    EnvelopeSoftmaxForward(threads).kernel(params, eps).launch(
        grid=[n_nodes, groups, 1],
        block=[threads, 1, 1],
        stream=stream,
    )


@device_aware_lru_cache(maxsize=16)
def compile_envelope_softmax_forward(threads: int, eps: float = 1.0e-7) -> Callable:
    e = cute.sym_int64()
    n = cute.sym_int64()
    g = cute.sym_int64()
    fake_logits = make_fake_compact_tensor(cutlass.Float32, (e, g), stride_order=(1, 0))
    fake_gate = make_fake_compact_tensor(cutlass.Float32, (e,), stride_order=(0,))
    fake_dst_ptr = make_fake_compact_tensor(
        cutlass.Int32, (cute.sym_int64(),), stride_order=(0,)
    )
    fake_z = make_fake_compact_tensor(cutlass.Float32, (g,), stride_order=(0,))
    fake_out = make_fake_compact_tensor(cutlass.Float32, (e, g), stride_order=(1, 0))
    fake_group_max = make_fake_compact_tensor(
        cutlass.Float32, (n, g), stride_order=(1, 0)
    )
    fake_denom = make_fake_compact_tensor(cutlass.Float32, (n, g), stride_order=(1, 0))
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        envelope_softmax_forward_jit,
        fake_logits,
        fake_gate,
        fake_dst_ptr,
        fake_z,
        fake_out,
        fake_group_max,
        fake_denom,
        threads,
        eps,
        fake_stream,
        options="--enable-tvm-ffi",
    )
