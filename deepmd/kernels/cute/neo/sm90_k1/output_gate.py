# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""SM90 CuTe epilogue for chunked final-SO2Linear sufficient statistics."""

from __future__ import (
    annotations,
)

import math
from typing import (
    TYPE_CHECKING,
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

from ..compile_cache import (
    device_aware_lru_cache,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN202, TC002

FOCUS_COUNT = 2
DEGREE_COUNT = 16
CHANNELS = 32
HIDDEN = FOCUS_COUNT * CHANNELS
THREADS = HIDDEN
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


@cute.jit
def _sigmoid(value):
    one = cutlass.Float32(1.0)
    return one / (one + cute.exp(-value))


@cute.jit
def _chunked_final_output_gate_jit(
    raw: cute.Tensor,
    x_wide: cute.Tensor,
    norm_scale: cute.Tensor,
    gate_weight: cute.Tensor,
    rotate_inv_rescale: cute.Tensor,
    out: cute.Tensor,
    eps: cutlass.Constexpr[float],
    stream: CUstream,
):
    _chunked_final_output_gate_kernel(
        raw,
        x_wide,
        norm_scale,
        gate_weight,
        rotate_inv_rescale,
        out,
        eps,
    ).launch(
        grid=[raw.shape[0], 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _chunked_final_output_gate_kernel(
    raw: cute.Tensor,
    x_wide: cute.Tensor,
    norm_scale: cute.Tensor,
    gate_weight: cute.Tensor,
    rotate_inv_rescale: cute.Tensor,
    out: cute.Tensor,
    eps: cutlass.Constexpr[float],
):
    tidx, _, _ = cute.arch.thread_idx()
    node, _, _ = cute.arch.block_idx()
    focus = tidx // CHANNELS
    channel = tidx - focus * CHANNELS

    x = x_wide[node, 0, tidx].to(cutlass.Float32)
    square_sum = cute.arch.warp_reduction_sum(x * x)
    inv_rms = cute.rsqrt(square_sum / cutlass.Float32(CHANNELS) + cutlass.Float32(eps))
    logit_part = (
        x
        * inv_rms
        * norm_scale[focus, channel].to(cutlass.Float32)
        * gate_weight[channel, focus, 0].to(cutlass.Float32)
    )
    gate = _sigmoid(cute.arch.warp_reduction_sum(logit_part))
    for degree in cutlass.range_constexpr(DEGREE_COUNT):
        value = raw[node, focus, degree, channel].to(cutlass.Float32)
        value *= rotate_inv_rescale[degree].to(cutlass.Float32)
        out[node, degree, tidx] = value * gate


def _fake_float(shape: tuple[object, ...], stride_order: tuple[int, ...]):
    return make_fake_compact_tensor(
        cutlass.Float32,
        shape,
        stride_order=stride_order,
        **FAKE_TENSOR_KW,
    )


@device_aware_lru_cache(maxsize=8)
def _compiled_chunked_final_output_gate(eps: float) -> Callable:
    nodes = cute.sym_int64()
    return cute.compile(
        _chunked_final_output_gate_jit,
        _fake_float(
            (nodes, FOCUS_COUNT, DEGREE_COUNT, CHANNELS),
            (3, 2, 1, 0),
        ),
        _fake_float((nodes, DEGREE_COUNT, HIDDEN), (2, 1, 0)),
        _fake_float((FOCUS_COUNT, CHANNELS), (1, 0)),
        _fake_float((CHANNELS, FOCUS_COUNT, 1), (2, 1, 0)),
        _fake_float((DEGREE_COUNT,), (0,)),
        _fake_float((nodes, DEGREE_COUNT, HIDDEN), (2, 1, 0)),
        eps,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.float32 or tensor.device != device:
        raise ValueError(f"{name} must be FP32 on {device}")
    if not tensor.is_contiguous() or tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be contiguous and 16-byte aligned")


def run_chunked_final_output_gate(
    *,
    raw: torch.Tensor,
    x_wide: torch.Tensor,
    norm_scale: torch.Tensor,
    gate_weight: torch.Tensor,
    rotate_inv_rescale: torch.Tensor,
    eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the real Neo rotate-rescale and output gate to node statistics."""
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("output-gate epsilon must be finite and positive")
    device = raw.device
    if device.type != "cuda" or tuple(torch.cuda.get_device_capability(device)) != (
        9,
        0,
    ):
        raise RuntimeError("final output gate requires SM90")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("strict FP32 requires allow_tf32=False")
    node_count = int(raw.shape[0])
    _require_tensor(
        "raw",
        raw,
        (node_count, FOCUS_COUNT, DEGREE_COUNT, CHANNELS),
        device,
    )
    _require_tensor("x_wide", x_wide, (node_count, DEGREE_COUNT, HIDDEN), device)
    _require_tensor("norm_scale", norm_scale, (FOCUS_COUNT, CHANNELS), device)
    _require_tensor("gate_weight", gate_weight, (CHANNELS, FOCUS_COUNT, 1), device)
    _require_tensor("rotate_inv_rescale", rotate_inv_rescale, (DEGREE_COUNT,), device)
    if out is None:
        out = torch.empty(
            (node_count, DEGREE_COUNT, HIDDEN),
            device=device,
            dtype=torch.float32,
        )
    else:
        _require_tensor("out", out, (node_count, DEGREE_COUNT, HIDDEN), device)
    with torch.cuda.device(device):
        _compiled_chunked_final_output_gate(float(eps))(
            raw,
            x_wide,
            norm_scale,
            gate_weight,
            rotate_inv_rescale,
            out,
        )
    return out


__all__ = ["run_chunked_final_output_gate"]
