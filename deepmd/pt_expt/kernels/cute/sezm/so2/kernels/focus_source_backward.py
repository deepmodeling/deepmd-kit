# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
# ruff: noqa: ANN001, ANN201, ANN202, TC002, UP035
"""CuTe forward for Neo attention-prelude source features.

The forward kernel fuses two independent PyTorch producer chains in one launch:

    optional focus RMSNorm -> two-focus logits -> softmax -> label smoothing
    scalar Q/K RMSNorm -> Q projection + K projection

The kernel is specialized to the Neo SO2 shape `(E, F=2, C=32)`.
"""

from __future__ import (
    annotations,
)

from functools import (
    lru_cache,
)
from typing import (
    Callable,
)

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import (
    CUstream,
)
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


@cute.jit
def _warp_sum(value):
    return cute.arch.warp_reduction_sum(value)


@cute.jit
def neo_attention_prelude_forward_jit(
    focus: cute.Tensor,
    x_l0: cute.Tensor,
    focus_weight: cute.Tensor,
    focus_scale: cute.Tensor,
    q_weight: cute.Tensor,
    k_weight: cute.Tensor,
    qk_scale: cute.Tensor,
    focus_alpha: cute.Tensor,
    q_node: cute.Tensor,
    k_node: cute.Tensor,
    stream: CUstream,
    focus_eps: cutlass.Float32,
    qk_eps: cutlass.Float32,
    tau: cutlass.Float32,
    label_smoothing: cutlass.Float32,
    use_focus_norm: cutlass.Constexpr[bool],
):
    edges, _ = focus.shape
    nodes, _, _ = x_l0.shape
    neo_attention_prelude_forward_kernel(
        focus,
        x_l0,
        focus_weight,
        focus_scale,
        q_weight,
        k_weight,
        qk_scale,
        focus_alpha,
        q_node,
        k_node,
        focus_eps,
        qk_eps,
        tau,
        label_smoothing,
        use_focus_norm,
    ).launch(
        # The launch covers two independent domains.  Using their sum avoids a
        # host-side branch on symbolic E/N while guaranteeing coverage of both.
        grid=[cute.ceil_div(edges + nodes, 8), 1, 1],
        block=[256, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_attention_prelude_forward_kernel(
    focus: cute.Tensor,
    x_l0: cute.Tensor,
    focus_weight: cute.Tensor,
    focus_scale: cute.Tensor,
    q_weight: cute.Tensor,
    k_weight: cute.Tensor,
    qk_scale: cute.Tensor,
    focus_alpha: cute.Tensor,
    q_node: cute.Tensor,
    k_node: cute.Tensor,
    focus_eps: cutlass.Float32,
    qk_eps: cutlass.Float32,
    tau: cutlass.Float32,
    label_smoothing: cutlass.Float32,
    use_focus_norm: cutlass.Constexpr[bool],
):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = tid % 32
    edge = block * 8 + tid // 32
    edges, _ = focus.shape

    if edge < edges:
        x0 = focus[edge, lane].to(cutlass.Float32)
        x1 = focus[edge, 32 + lane].to(cutlass.Float32)
        if cutlass.const_expr(use_focus_norm):
            inv0 = cute.rsqrt(_warp_sum(x0 * x0) / cutlass.Float32(32.0) + focus_eps)
            inv1 = cute.rsqrt(_warp_sum(x1 * x1) / cutlass.Float32(32.0) + focus_eps)
            norm0 = x0 * inv0 * focus_scale[0, lane].to(cutlass.Float32)
            norm1 = x1 * inv1 * focus_scale[1, lane].to(cutlass.Float32)
        else:
            norm0 = x0
            norm1 = x1
        logit0 = _warp_sum(norm0 * focus_weight[lane, 0].to(cutlass.Float32))
        logit1 = _warp_sum(norm1 * focus_weight[lane, 1].to(cutlass.Float32))

        if lane == 0:
            z0 = logit0 / tau
            z1 = logit1 / tau
            zmax = z0
            if z1 > zmax:
                zmax = z1
            e0 = cute.exp(z0 - zmax)
            e1 = cute.exp(z1 - zmax)
            denom = e0 + e1
            keep = cutlass.Float32(1.0) - label_smoothing
            smooth = label_smoothing / cutlass.Float32(2.0)
            focus_alpha[edge, 0] = (e0 / denom * keep + smooth).to(
                focus_alpha.element_type
            )
            focus_alpha[edge, 1] = (e1 / denom * keep + smooth).to(
                focus_alpha.element_type
            )

    # Q/K is a per-node chain, not a child of the per-edge focus chain.  Keep
    # this guard independent so every node row is initialized even when E < N.
    nodes, _, _ = x_l0.shape
    node = block * 8 + tid // 32
    if node < nodes:
        qk_x0 = x_l0[node, 0, lane].to(cutlass.Float32)
        qk_x1 = x_l0[node, 1, lane].to(cutlass.Float32)
        qk_norm0 = (
            qk_x0
            * cute.rsqrt(_warp_sum(qk_x0 * qk_x0) / cutlass.Float32(32.0) + qk_eps)
            * qk_scale[0, lane].to(cutlass.Float32)
        )
        qk_norm1 = (
            qk_x1
            * cute.rsqrt(_warp_sum(qk_x1 * qk_x1) / cutlass.Float32(32.0) + qk_eps)
            * qk_scale[1, lane].to(cutlass.Float32)
        )
        q0 = cutlass.Float32(0.0)
        k0 = cutlass.Float32(0.0)
        q1 = cutlass.Float32(0.0)
        so2 = cutlass.Float32(0.0)
        for input_channel in cutlass.range_constexpr(32):
            value0 = cute.arch.shuffle_sync(qk_norm0, input_channel)
            value1 = cute.arch.shuffle_sync(qk_norm1, input_channel)
            q0 += value0 * q_weight[input_channel, 0, lane].to(cutlass.Float32)
            k0 += value0 * k_weight[input_channel, 0, lane].to(cutlass.Float32)
            q1 += value1 * q_weight[input_channel, 1, lane].to(cutlass.Float32)
            so2 += value1 * k_weight[input_channel, 1, lane].to(cutlass.Float32)
        q_node[node, 0, lane] = q0.to(q_node.element_type)
        k_node[node, 0, lane] = k0.to(k_node.element_type)
        q_node[node, 1, lane] = q1.to(q_node.element_type)
        k_node[node, 1, lane] = so2.to(k_node.element_type)


@lru_cache(maxsize=8)
def compile_neo_attention_prelude_forward(
    focus_eps: float,
    qk_eps: float,
    tau: float,
    label_smoothing: float,
    compile_identity: tuple[int, int, int] | None = None,
    *,
    use_focus_norm: bool = True,
) -> Callable:
    # The identity keeps independently compiled device/architecture binaries in
    # distinct cache entries. Compilation itself runs under the runner's device.
    del compile_identity
    edges = cute.sym_int64()
    nodes = cute.sym_int64()
    fake_focus = make_fake_compact_tensor(
        cutlass.Float32, (edges, 64), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    fake_x_l0 = make_fake_compact_tensor(
        cutlass.Float32, (nodes, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    fake_focus_weight = make_fake_compact_tensor(
        cutlass.Float32, (32, 2), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    fake_focus_scale = make_fake_compact_tensor(
        cutlass.Float32, (2, 32), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    fake_q_weight = make_fake_compact_tensor(
        cutlass.Float32, (32, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    fake_k_weight = make_fake_compact_tensor(
        cutlass.Float32, (32, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    fake_qk_scale = make_fake_compact_tensor(
        cutlass.Float32, (2, 32), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    fake_focus_alpha = make_fake_compact_tensor(
        cutlass.Float32, (edges, 2), stride_order=(1, 0), **FAKE_TENSOR_KW
    )
    fake_q_node = make_fake_compact_tensor(
        cutlass.Float32, (nodes, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    fake_k_node = make_fake_compact_tensor(
        cutlass.Float32, (nodes, 2, 32), stride_order=(2, 1, 0), **FAKE_TENSOR_KW
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        neo_attention_prelude_forward_jit,
        fake_focus,
        fake_x_l0,
        fake_focus_weight,
        fake_focus_scale,
        fake_q_weight,
        fake_k_weight,
        fake_qk_scale,
        fake_focus_alpha,
        fake_q_node,
        fake_k_node,
        fake_stream,
        cutlass.Float32(focus_eps),
        cutlass.Float32(qk_eps),
        cutlass.Float32(tau),
        cutlass.Float32(label_smoothing),
        bool(use_focus_norm),
        options="--enable-tvm-ffi",
    )

    def run(
        focus,
        x_l0,
        focus_weight,
        focus_scale,
        q_weight,
        k_weight,
        qk_scale,
        focus_alpha,
        q_node,
        k_node,
    ):
        return compiled(
            focus,
            x_l0,
            focus_weight,
            focus_scale,
            q_weight,
            k_weight,
            qk_scale,
            focus_alpha,
            q_node,
            k_node,
            cutlass.Float32(focus_eps),
            cutlass.Float32(qk_eps),
            cutlass.Float32(tau),
            cutlass.Float32(label_smoothing),
        )

    return run
