# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Strict-FP32 one-pass Neo Phase-C forward with output gating.

This module specializes the supported Neo shape ``D=16, Dm=10, F=2,
C=32``.  One 64-thread CTA owns one destination CSR row.  Lane
``focus * 32 + channel`` sweeps every edge in that row and retains its 16
output-degree accumulators in a CuTe register fragment.  Wigner values are
cooperatively staged through a double-buffered shared-memory panel, so no
``node * chunks`` partial tensor or second reduction launch is required.

The final stores fuse the output-side attention gate:

``sigmoid(project(RMSNorm(x_wide[:, 0]))) * rotate_inv_rescale * aggregate``.

Dense Wigner, packed Wigner through the generic loader, and the 46-value
packed-direct path share the same runtime tensor signature.  Every floating
tensor and every arithmetic operation is FP32 by construction.
"""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    TYPE_CHECKING,
    Any,
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

from ..compile_cache import (
    device_aware_lru_cache,
)
from ..k1_wigner_layout import PACKED_VALUE_COUNT as PACKED_WIGNER_VALUES

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN201, ANN202, TC002

DEGREE_COUNT = 16
REDUCED_COUNT = 10
FOCUS_COUNT = 2
CHANNELS = 32
HIDDEN = FOCUS_COUNT * CHANNELS
PHASE_WIDTH = REDUCED_COUNT * HIDDEN
OUTPUT_WIDTH = DEGREE_COUNT * HIDDEN
THREADS = HIDDEN
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


@dataclass(frozen=True)
class NeoPhaseCOnePassParams:
    """Runtime tensors for the warp-private packed kernel."""

    x_local: cute.Tensor
    wigner_dt: cute.Tensor
    alpha: cute.Tensor
    focus_alpha: cute.Tensor
    out: cute.Tensor
    x_wide: cute.Tensor
    norm_scale: cute.Tensor
    gate_weight: cute.Tensor
    dst_ptr: cute.Tensor
    rotate_inv_rescale: cute.Tensor


@cute.jit
def _sigmoid(value):
    one = cutlass.Float32(1.0)
    return one / (one + cute.exp(-value))


@cute.jit
def _weighted_input(
    params: NeoPhaseCOnePassParams,
    edge,
    tid,
    focus,
    channel,
    reduced: cutlass.Constexpr[int],
    alpha,
    focus_scale,
):
    """Load one reduced coefficient and apply both edge attention weights."""
    load_idx = focus * REDUCED_COUNT * CHANNELS + reduced * CHANNELS + channel
    value = params.x_local[edge, load_idx].to(cutlass.Float32)
    # Match the two-launch path's FP32 association exactly.
    return value * alpha * focus_scale


@cute.jit
def _focus_scale(
    params: NeoPhaseCOnePassParams,
    edge,
    focus,
):
    return params.focus_alpha[edge, focus].to(cutlass.Float32)


@cute.jit
def _output_gate(
    params: NeoPhaseCOnePassParams,
    node,
    tid,
    focus,
    channel,
    eps: cutlass.Constexpr[float],
):
    """Compute one gate per focus; each warp owns one 32-channel focus."""
    x = params.x_wide[node, tid].to(cutlass.Float32)
    square_sum = cute.arch.warp_reduction_sum(x * x)
    inv_rms = cute.rsqrt(square_sum / cutlass.Float32(CHANNELS) + cutlass.Float32(eps))
    logit_part = (
        x
        * inv_rms
        * params.norm_scale[focus, channel].to(cutlass.Float32)
        * params.gate_weight[channel, focus, 0].to(cutlass.Float32)
    )
    logit = cute.arch.warp_reduction_sum(logit_part)
    return _sigmoid(logit)


@cute.jit
def _store_gated_output(
    params: NeoPhaseCOnePassParams,
    accumulator: cute.Tensor,
    node,
    tid,
    gate,
):
    for degree in cutlass.range_constexpr(DEGREE_COUNT):
        value = accumulator[degree]
        value *= params.rotate_inv_rescale[degree].to(cutlass.Float32)
        value *= gate
        params.out[node, degree * HIDDEN + tid] = value


@cute.jit
def _warp_private_packed_wigner(
    panel,
    focus,
    panel_index: cutlass.Constexpr[int],
):
    """Load one scalar from a focus warp's private shared Wigner panel."""
    return panel[focus, panel_index]


@cute.jit
def neo_phase_c_onepass_output_gate_packed_direct_warp_private_jit(
    x_local: cute.Tensor,
    wigner_dt: cute.Tensor,
    alpha: cute.Tensor,
    focus_alpha: cute.Tensor,
    out: cute.Tensor,
    x_wide: cute.Tensor,
    norm_scale: cute.Tensor,
    gate_weight: cute.Tensor,
    dst_ptr: cute.Tensor,
    rotate_inv_rescale: cute.Tensor,
    eps: cutlass.Constexpr[float],
    stream: CUstream,
):
    """Launch the packed-direct kernel with a warp-local Wigner epilogue."""
    params = NeoPhaseCOnePassParams(
        x_local=x_local,
        wigner_dt=wigner_dt,
        alpha=alpha,
        focus_alpha=focus_alpha,
        out=out,
        x_wide=x_wide,
        norm_scale=norm_scale,
        gate_weight=gate_weight,
        dst_ptr=dst_ptr,
        rotate_inv_rescale=rotate_inv_rescale,
    )
    accumulator_layout = cute.make_layout((DEGREE_COUNT,), stride=(1,))
    node_count, _ = out.shape
    neo_phase_c_onepass_output_gate_packed_direct_warp_private_kernel(
        params,
        accumulator_layout,
        eps,
    ).launch(
        grid=[node_count, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_phase_c_onepass_output_gate_packed_direct_warp_private_kernel(
    params: NeoPhaseCOnePassParams,
    accumulator_layout: cute.Layout,
    eps: cutlass.Constexpr[float],
):
    """Store the gated output without a Wigner shared-memory round trip."""
    tid, _, _ = cute.arch.thread_idx()
    node, _, _ = cute.arch.block_idx()
    focus = tid // CHANNELS
    channel = tid - focus * CHANNELS
    lane = cute.arch.lane_idx()

    # Each focus is exactly one warp.  The gate reduction already broadcasts
    # its result within that warp, so it does not need shared memory either.
    smem = cutlass.utils.SmemAllocator()
    panel_storage = smem.allocate_tensor(
        cutlass.Float32,
        FOCUS_COUNT * PACKED_WIGNER_VALUES,
    )
    panel_layout = cute.make_layout(
        (FOCUS_COUNT, PACKED_WIGNER_VALUES),
        stride=(PACKED_WIGNER_VALUES, 1),
    )
    panel = cute.make_tensor(panel_storage.iterator, panel_layout)
    gate = _output_gate(params, node, tid, focus, channel, eps)
    accumulator = cute.make_rmem_tensor(accumulator_layout, cutlass.Float32)
    accumulator.fill(0.0)

    lo = params.dst_ptr[node]
    hi = params.dst_ptr[node + 1]
    for edge in cutlass.range(lo, hi, 1, unroll=1):
        panel[focus, lane] = params.wigner_dt[edge, lane].to(cutlass.Float32)
        if lane < PACKED_WIGNER_VALUES - CHANNELS:
            panel[focus, lane + CHANNELS] = params.wigner_dt[edge, lane + CHANNELS].to(
                cutlass.Float32
            )
        cute.arch.sync_warp()

        alpha = params.alpha[edge, focus].to(cutlass.Float32)
        focus_scale = _focus_scale(params, edge, focus)
        value0 = _weighted_input(
            params,
            edge,
            tid,
            focus,
            channel,
            0,
            alpha,
            focus_scale,
        )
        accumulator[0] += _warp_private_packed_wigner(panel, focus, 0) * value0

        for row_slot in cutlass.range_constexpr(3):
            value1 = _weighted_input(
                params,
                edge,
                tid,
                focus,
                channel,
                1 + row_slot * 3,
                alpha,
                focus_scale,
            )
            panel_start1 = 1 + row_slot * 3
            for local_row in cutlass.range_constexpr(3):
                accumulator[1 + local_row] += (
                    _warp_private_packed_wigner(
                        panel,
                        focus,
                        panel_start1 + local_row,
                    )
                    * value1
                )

        for row_slot in cutlass.range_constexpr(3):
            value2 = _weighted_input(
                params,
                edge,
                tid,
                focus,
                channel,
                2 + row_slot * 3,
                alpha,
                focus_scale,
            )
            panel_start2 = 10 + row_slot * 5
            for local_row in cutlass.range_constexpr(5):
                accumulator[4 + local_row] += (
                    _warp_private_packed_wigner(
                        panel,
                        focus,
                        panel_start2 + local_row,
                    )
                    * value2
                )

        for row_slot in cutlass.range_constexpr(3):
            value3 = _weighted_input(
                params,
                edge,
                tid,
                focus,
                channel,
                3 + row_slot * 3,
                alpha,
                focus_scale,
            )
            panel_start3 = 25 + row_slot * 7
            for local_row in cutlass.range_constexpr(7):
                accumulator[9 + local_row] += (
                    _warp_private_packed_wigner(
                        panel,
                        focus,
                        panel_start3 + local_row,
                    )
                    * value3
                )

        # Do not overwrite this warp's panel until every lane has consumed it.
        cute.arch.sync_warp()

    _store_gated_output(params, accumulator, node, tid, gate)


def _fake_common_tensors():
    edges = cute.sym_int64()
    nodes = cute.sym_int64()
    x_local = make_fake_compact_tensor(
        cutlass.Float32,
        (edges, PHASE_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    wigner_dt = make_fake_compact_tensor(
        cutlass.Float32,
        (edges, PACKED_WIGNER_VALUES),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    alpha = make_fake_compact_tensor(
        cutlass.Float32,
        (edges, FOCUS_COUNT),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    focus_alpha = make_fake_compact_tensor(
        cutlass.Float32,
        (edges, FOCUS_COUNT),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    out = make_fake_compact_tensor(
        cutlass.Float32,
        (nodes, OUTPUT_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    x_wide = make_fake_compact_tensor(
        cutlass.Float32,
        (nodes, OUTPUT_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
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
    dst_ptr = make_fake_compact_tensor(
        cutlass.Int32,
        (cute.sym_int64(),),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    rotate_inv_rescale = make_fake_compact_tensor(
        cutlass.Float32,
        (DEGREE_COUNT,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    return (
        x_local,
        wigner_dt,
        alpha,
        focus_alpha,
        out,
        x_wide,
        norm_scale,
        gate_weight,
        dst_ptr,
        rotate_inv_rescale,
    )


def compile_neo_phase_c_onepass_output_gate(
    eps: float,
) -> Callable:
    """Compile the packed, focus-major, warp-private specialization."""
    if not (eps > 0.0 and eps < float("inf")):
        raise ValueError("output-gate RMSNorm eps must be finite and positive")
    common_args = _fake_common_tensors()
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        neo_phase_c_onepass_output_gate_packed_direct_warp_private_jit,
        *common_args,
        eps,
        stream,
        options="--enable-tvm-ffi",
    )


@device_aware_lru_cache(maxsize=8)
def _compiled_neo_phase_c_onepass_output_gate(eps: float) -> Callable:
    return compile_neo_phase_c_onepass_output_gate(eps)


def _expect_shape(name: str, tensor: Any, expected: tuple[int, ...]) -> None:
    actual = tuple(tensor.shape)
    if actual != expected:
        raise ValueError(f"expected {name} shape {expected}, got {actual}")


def _expect_fp32_cuda(name: str, tensor: Any, *, torch: Any, device: Any) -> None:
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be strict float32, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def run_neo_phase_c_onepass_output_gate(
    *,
    x_local_flat: Any,
    Dt_full: Any,
    alpha_focus: Any,
    focus_compete_alpha: Any,
    dst_ptr: Any,
    rotate_inv_rescale: Any,
    x_wide: Any,
    output_gate_norm_scale: Any,
    output_gate_weight: Any,
    output_gate_eps: float,
    out: Any | None = None,
) -> Any:
    """Validate and launch fused Phase C and output gating."""
    import torch

    if not x_local_flat.is_cuda:
        raise ValueError("x_local_flat must be a CUDA tensor")

    device = x_local_flat.device
    edge_count = x_local_flat.shape[0]
    node_count = x_wide.shape[0]
    _expect_shape(
        "x_local_flat",
        x_local_flat,
        (edge_count, FOCUS_COUNT, REDUCED_COUNT, CHANNELS),
    )
    _expect_shape("x_wide", x_wide, (node_count, DEGREE_COUNT, HIDDEN))
    _expect_shape("alpha_focus", alpha_focus, (edge_count, FOCUS_COUNT))
    _expect_shape(
        "output_gate_norm_scale",
        output_gate_norm_scale,
        (FOCUS_COUNT, CHANNELS),
    )
    _expect_shape(
        "output_gate_weight",
        output_gate_weight,
        (CHANNELS, FOCUS_COUNT, 1),
    )
    _expect_shape("rotate_inv_rescale", rotate_inv_rescale, (DEGREE_COUNT,))
    _expect_shape("dst_ptr", dst_ptr, (node_count + 1,))
    _expect_shape(
        "focus_compete_alpha",
        focus_compete_alpha,
        (edge_count, FOCUS_COUNT),
    )
    _expect_shape("Dt_full", Dt_full, (edge_count, PACKED_WIGNER_VALUES))

    floating_tensors = {
        "x_local_flat": x_local_flat,
        "Dt_full": Dt_full,
        "alpha_focus": alpha_focus,
        "focus_compete_alpha": focus_compete_alpha,
        "rotate_inv_rescale": rotate_inv_rescale,
        "x_wide": x_wide,
        "output_gate_norm_scale": output_gate_norm_scale,
        "output_gate_weight": output_gate_weight,
    }
    for name, tensor in floating_tensors.items():
        _expect_fp32_cuda(name, tensor, torch=torch, device=device)

    if dst_ptr.device != device:
        raise ValueError("dst_ptr must be on the input CUDA device")
    if dst_ptr.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"dst_ptr must be int32 or int64, got {dst_ptr.dtype}")

    if out is None:
        out = torch.empty(
            node_count,
            DEGREE_COUNT,
            HIDDEN,
            device=device,
            dtype=torch.float32,
        )
    else:
        _expect_shape("out", out, (node_count, DEGREE_COUNT, HIDDEN))
        _expect_fp32_cuda("out", out, torch=torch, device=device)
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")

    with torch.cuda.device(device):
        kernel = _compiled_neo_phase_c_onepass_output_gate(float(output_gate_eps))
        kernel(
            x_local_flat.contiguous().view(edge_count, REDUCED_COUNT * HIDDEN),
            Dt_full.contiguous(),
            alpha_focus.contiguous(),
            focus_compete_alpha.contiguous(),
            out.view(node_count, DEGREE_COUNT * HIDDEN),
            x_wide.contiguous().view(node_count, DEGREE_COUNT * HIDDEN),
            output_gate_norm_scale.contiguous(),
            output_gate_weight.contiguous(),
            dst_ptr.to(dtype=torch.int32).contiguous(),
            rotate_inv_rescale.contiguous(),
        )
    return out


__all__ = [
    "compile_neo_phase_c_onepass_output_gate",
    "neo_phase_c_onepass_output_gate_packed_direct_warp_private_jit",
    "neo_phase_c_onepass_output_gate_packed_direct_warp_private_kernel",
    "run_neo_phase_c_onepass_output_gate",
]
