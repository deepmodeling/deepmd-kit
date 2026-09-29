# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compact radial state for the SM90 persistent-complex Neo SO2 path.

The fused Phase-A kernel computes three logically separate values in one
edge CTA: the packed-Wigner rotation, the 25-value compact radial map, and the
64-value degree-zero radial attention feature.  A persistent-complex Phase A
must not call that kernel merely to recover the latter two values because doing
so would also allocate and write the discarded ``(E,2,10,32)`` real stack.

This module retains the same FP32 reduction order for the two
radial projections while omitting the dense SO2 output.  The resulting compact
state is consumed directly by the split-complex Phase A and its adjoint.
"""

from __future__ import (
    annotations,
)

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

from ...compile_cache import (
    device_aware_lru_cache,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN202, TC002

RADIAL_WIDTH = 4 * 32
COMPACT_WIDTH = 25
ATTENTION_WIDTH = 64
THREADS = 64
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}

__all__ = [
    "project_neo_radial_input_adjoint_fp32",
    "run_neo_radial_state_forward_fp32",
]


@cute.jit
def _radial_state_forward_jit(
    radial,
    combined_weight,
    hidden_weight,
    compact_out,
    attention_out,
    stream: CUstream,
):
    edge_count, _ = radial.shape
    _radial_state_forward_kernel(
        radial,
        combined_weight,
        hidden_weight,
        compact_out,
        attention_out,
    ).launch(
        grid=[edge_count, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
    )


@cute.kernel
def _radial_state_forward_kernel(
    radial,
    combined_weight,
    hidden_weight,
    compact_out,
    attention_out,
):
    channel, _, _ = cute.arch.thread_idx()
    edge, _, _ = cute.arch.block_idx()

    if channel < COMPACT_WIDTH:
        compact = cutlass.Float32(0.0)
        for radial_channel in cutlass.range_constexpr(RADIAL_WIDTH):
            compact += radial[edge, radial_channel].to(
                cutlass.Float32
            ) * combined_weight[radial_channel, channel].to(cutlass.Float32)
        compact_out[edge, channel] = compact.to(compact_out.element_type)

    attention = cutlass.Float32(0.0)
    for radial_channel in cutlass.range_constexpr(32):
        attention += radial[edge, radial_channel].to(cutlass.Float32) * hidden_weight[
            radial_channel, channel
        ].to(cutlass.Float32)
    attention_out[edge, channel] = attention.to(attention_out.element_type)


@device_aware_lru_cache(maxsize=4)
def _compiled_radial_state_forward() -> Callable:
    edge_count = cute.sym_int64()
    fake_radial = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, RADIAL_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_combined = make_fake_compact_tensor(
        cutlass.Float32,
        (RADIAL_WIDTH, COMPACT_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_hidden = make_fake_compact_tensor(
        cutlass.Float32,
        (32, ATTENTION_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_compact = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, COMPACT_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_attention = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, ATTENTION_WIDTH),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    return cute.compile(
        _radial_state_forward_jit,
        fake_radial,
        fake_combined,
        fake_hidden,
        fake_compact,
        fake_attention,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _expect_fp32_cuda(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if (
        tensor.dtype != torch.float32
        or tensor.device != device
        or not tensor.is_cuda
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous CUDA float32 on {device}")


def run_neo_radial_state_forward_fp32(
    *,
    radial_feat: torch.Tensor,
    combined_weight: torch.Tensor,
    hidden_weight: torch.Tensor,
    compact_out: torch.Tensor | None = None,
    attention_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the 25-value Phase-A map and 64-value attention feature."""
    if radial_feat.ndim != 3 or tuple(radial_feat.shape[1:]) != (4, 32):
        raise ValueError(
            f"radial_feat must have shape (E,4,32), got {tuple(radial_feat.shape)}"
        )
    if not radial_feat.is_cuda:
        raise ValueError("radial_feat must be a CUDA tensor")
    device = radial_feat.device
    edge_count = radial_feat.shape[0]
    _expect_fp32_cuda(
        "radial_feat",
        radial_feat,
        (edge_count, 4, 32),
        device,
    )
    _expect_fp32_cuda(
        "combined_weight",
        combined_weight,
        (RADIAL_WIDTH, COMPACT_WIDTH),
        device,
    )
    _expect_fp32_cuda(
        "hidden_weight",
        hidden_weight,
        (32, ATTENTION_WIDTH),
        device,
    )
    if edge_count <= 0:
        raise ValueError("persistent-complex SO2 requires E > 0")
    if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
        raise RuntimeError("persistent-complex SO2 requires SM90")
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("strict FP32 requires allow_tf32=False")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("strict FP32 requires float32 matmul precision 'highest'")

    if compact_out is None:
        compact_out = torch.empty(
            (edge_count, COMPACT_WIDTH),
            dtype=torch.float32,
            device=device,
        )
    if attention_out is None:
        attention_out = torch.empty(
            (edge_count, ATTENTION_WIDTH),
            dtype=torch.float32,
            device=device,
        )
    _expect_fp32_cuda(
        "compact_out",
        compact_out,
        (edge_count, COMPACT_WIDTH),
        device,
    )
    _expect_fp32_cuda(
        "attention_out",
        attention_out,
        (edge_count, ATTENTION_WIDTH),
        device,
    )

    with torch.cuda.device(device):
        _compiled_radial_state_forward()(
            radial_feat.view(edge_count, RADIAL_WIDTH),
            combined_weight,
            hidden_weight,
            compact_out,
            attention_out,
        )
    return compact_out, attention_out


def project_neo_radial_input_adjoint_fp32(
    *,
    grad_compact: torch.Tensor,
    grad_logits: torch.Tensor,
    combined_weight: torch.Tensor,
    combined_attention_weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project compact Phase-A and attention adjoints to ``(E,4,32)``.

    This intentionally follows the strict-FP32 cuBLAS route. No approximation
    is introduced: the first matrix product is the compact radial adjoint and
    the degree-zero slice receives the independent attention-logit adjoint.
    """
    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("strict FP32 requires allow_tf32=False")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("strict FP32 requires float32 matmul precision 'highest'")
    edge_count = grad_compact.shape[0]
    device = grad_compact.device
    _expect_fp32_cuda(
        "grad_compact",
        grad_compact,
        (edge_count, COMPACT_WIDTH),
        device,
    )
    _expect_fp32_cuda(
        "grad_logits",
        grad_logits,
        (edge_count, 2),
        device,
    )
    _expect_fp32_cuda(
        "combined_weight",
        combined_weight,
        (RADIAL_WIDTH, COMPACT_WIDTH),
        device,
    )
    _expect_fp32_cuda(
        "combined_attention_weight",
        combined_attention_weight,
        (32, 2),
        device,
    )
    if out is None:
        out = torch.empty(
            (edge_count, 4, 32),
            dtype=torch.float32,
            device=device,
        )
    _expect_fp32_cuda("out", out, (edge_count, 4, 32), device)

    out_flat = out.view(edge_count, RADIAL_WIDTH)
    torch.mm(grad_compact, combined_weight.transpose(0, 1), out=out_flat)
    out_flat[:, :32].addmm_(
        grad_logits,
        combined_attention_weight.transpose(0, 1),
    )
    return out
