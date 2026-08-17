# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Manual backward for the fused Neo attention output gate.

Only input gradients are produced.  The gate and RMSNorm parameter gradients
are intentionally omitted for the E/F/S path.  ``grad_phase`` may alias
``grad_gated``; ``grad_x_wide`` is an existing accumulation buffer and only its
first 64 values per node are updated.

The forward gate is recomputed.  Its logit gradient uses
``sum(grad_gated * gated_out) * (1 - gate)``, so backward does not need either
an ungated Phase-C aggregate or a saved gate tensor.
"""

from __future__ import (
    annotations,
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

# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN201, ANN202, TC002, UP035

FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}
DEGREE_COUNT = 16
FOCUS_COUNT = 2
CHANNELS = 32
HIDDEN = FOCUS_COUNT * CHANNELS
OUTPUT_WIDTH = DEGREE_COUNT * HIDDEN


@cute.jit
def _warp_sum(value):
    return cute.arch.warp_reduction_sum(value)


@cute.jit
def _sigmoid(value):
    one = cutlass.Float32(1.0)
    return one / (one + cute.exp(-value))


@cute.jit
def compute_neo_inv_rms(x, eps: cutlass.Constexpr[float]):
    square_sum = _warp_sum(x * x)
    return cute.rsqrt(square_sum / cutlass.Float32(CHANNELS) + cutlass.Float32(eps))


@cute.jit
def compute_neo_output_gate(
    x,
    norm_scale: cute.Tensor,
    gate_weight: cute.Tensor,
    inv_rms,
    focus,
    channel,
):
    logit_part = (
        x
        * inv_rms
        * norm_scale[focus, channel].to(cutlass.Float32)
        * gate_weight[channel, focus, 0].to(cutlass.Float32)
    )
    return _sigmoid(_warp_sum(logit_part))


@cute.jit
def neo_output_gate_backward_jit(
    grad_gated: cute.Tensor,
    gated_out: cute.Tensor,
    x_wide: cute.Tensor,
    norm_scale: cute.Tensor,
    gate_weight: cute.Tensor,
    grad_phase: cute.Tensor,
    grad_x_wide: cute.Tensor,
    stream: CUstream,
    eps: cutlass.Constexpr[float],
):
    nodes, _ = gated_out.shape
    neo_output_gate_backward_kernel(
        grad_gated,
        gated_out,
        x_wide,
        norm_scale,
        gate_weight,
        grad_phase,
        grad_x_wide,
        eps,
    ).launch(
        grid=[nodes, 1, 1],
        block=[HIDDEN, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_output_gate_backward_kernel(
    grad_gated: cute.Tensor,
    gated_out: cute.Tensor,
    x_wide: cute.Tensor,
    norm_scale: cute.Tensor,
    gate_weight: cute.Tensor,
    grad_phase: cute.Tensor,
    grad_x_wide: cute.Tensor,
    eps: cutlass.Constexpr[float],
):
    tid, _, _ = cute.arch.thread_idx()
    node, _, _ = cute.arch.block_idx()
    focus = tid // CHANNELS
    channel = tid - focus * CHANNELS

    x = x_wide[node, tid].to(cutlass.Float32)
    inv_rms = compute_neo_inv_rms(x, eps)
    gate_value = compute_neo_output_gate(
        x,
        norm_scale,
        gate_weight,
        inv_rms,
        focus,
        channel,
    )

    gate_dot = cutlass.Float32(0.0)
    for degree in cutlass.range_constexpr(DEGREE_COUNT):
        idx = degree * HIDDEN + tid
        grad = grad_gated[node, idx].to(cutlass.Float32)
        gated = gated_out[node, idx].to(cutlass.Float32)
        grad_phase[node, idx] = (grad * gate_value).to(grad_phase.element_type)
        gate_dot += grad * gated

    gate_dot = _warp_sum(gate_dot)
    grad_logit = gate_dot * (cutlass.Float32(1.0) - gate_value)

    scale = norm_scale[focus, channel].to(cutlass.Float32)
    weight = gate_weight[channel, focus, 0].to(cutlass.Float32)
    grad_scaled = grad_logit * weight * scale
    rms_coeff = _warp_sum(grad_scaled * x) / cutlass.Float32(CHANNELS)

    grad_x = grad_scaled * inv_rms
    grad_x -= x * inv_rms * inv_rms * inv_rms * rms_coeff
    previous = grad_x_wide[node, tid].to(cutlass.Float32)
    grad_x_wide[node, tid] = (previous + grad_x).to(grad_x_wide.element_type)


def compile_neo_output_gate_backward(eps: float) -> Callable:
    """Compile input-only backward.

    Runtime order is ``grad_gated, gated_out, x_wide, norm_scale, gate_weight,
    grad_phase, grad_x_wide``.  ``grad_phase`` may alias ``grad_gated`` and the
    kernel adds only the scalar row into the preinitialized ``grad_x_wide``.
    """
    nodes = cute.sym_int64()

    def fake_node_tensor():
        return make_fake_compact_tensor(
            cutlass.Float32,
            (nodes, OUTPUT_WIDTH),
            stride_order=(1, 0),
            **FAKE_TENSOR_KW,
        )

    grad_gated = fake_node_tensor()
    gated_out = fake_node_tensor()
    x_wide = fake_node_tensor()
    norm_scale = make_fake_compact_tensor(
        cutlass.Float32,
        (FOCUS_COUNT, CHANNELS),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    gate_weight = make_fake_compact_tensor(
        cutlass.Float32,
        (CHANNELS, FOCUS_COUNT, 1),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    grad_phase = fake_node_tensor()
    grad_x_wide = fake_node_tensor()
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        neo_output_gate_backward_jit,
        grad_gated,
        gated_out,
        x_wide,
        norm_scale,
        gate_weight,
        grad_phase,
        grad_x_wide,
        stream,
        eps,
        options="--enable-tvm-ffi",
    )
