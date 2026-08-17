# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""PyTorch/cuBLAS glue for the structural Neo K1 split-gate path.

The two focus panels stay contiguous across the gate-linear boundary. Forward
uses direct cuBLAS matrix products, while backward accumulates into the scalar
slice of ``grad_y`` through the GEMM beta epilogue.
"""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch
from torch import (
    Tensor,
)

FOCUS_COUNT = 2
CHANNELS = 32
GATE_WIDTH = 3 * CHANNELS
VEC4_ALIGNMENT_BYTES = 16
VEC4_STORAGE_OFFSET_MULTIPLE = 4


def _dispatch_aligned_vec4_kernel(
    kernel: Any,
    tensor_names: tuple[str, ...],
    *tensors: Tensor,
) -> Any:
    """Validate the float4 load/store contract, then dispatch the kernel."""
    if len(tensor_names) != len(tensors):
        raise ValueError("vec4 dispatch tensor names and arguments must match")
    if not tensors or tensors[0].numel() == 0:
        return None

    for name, tensor in zip(tensor_names, tensors, strict=True):
        if tensor.dtype != torch.float32:
            raise TypeError(
                "SM80 vec4 structural gate requires float32 tensors; "
                f"{name} has dtype={tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                "SM80 vec4 structural gate requires compact tensors; "
                f"{name} has shape={tuple(tensor.shape)} and stride={tensor.stride()}"
            )
        if tensor.storage_offset() % VEC4_STORAGE_OFFSET_MULTIPLE:
            raise ValueError(
                "SM80 vec4 structural gate requires storage offsets divisible "
                f"by {VEC4_STORAGE_OFFSET_MULTIPLE} float32 elements; "
                f"{name} has storage_offset={tensor.storage_offset()}"
            )
        pointer_remainder = tensor.data_ptr() % VEC4_ALIGNMENT_BYTES
        if pointer_remainder:
            raise ValueError(
                "SM80 vec4 structural gate requires 16-byte-aligned tensors; "
                f"{name} has data_ptr modulo 16={pointer_remainder}"
            )

    return kernel(*tensors)


def focus_major_gate_linear_forward(
    gate_src: Tensor,
    gate_weight: Tensor,
) -> Tensor:
    """Project ``(E, 2, 32)`` into contiguous ``(2, E, 96)`` panels."""
    edge_count = gate_src.shape[0]
    weight = gate_weight.view(CHANNELS, FOCUS_COUNT, GATE_WIDTH)
    logits = torch.empty(
        FOCUS_COUNT,
        edge_count,
        GATE_WIDTH,
        dtype=gate_src.dtype,
        device=gate_src.device,
    )
    for focus in range(FOCUS_COUNT):
        torch.mm(gate_src[:, focus, :], weight[:, focus, :], out=logits[focus])
    return logits


def focus_major_gate_linear_backward_add_(
    grad_y: Tensor,
    grad_logits: Tensor,
    gate_weight: Tensor,
) -> Tensor:
    """Accumulate the gate-linear adjoint into ``grad_y`` in place."""
    weight = gate_weight.view(CHANNELS, FOCUS_COUNT, GATE_WIDTH)
    for focus in range(FOCUS_COUNT):
        grad_y[:, focus, 0, :].addmm_(
            grad_logits[focus],
            weight[:, focus, :].T,
        )
    return grad_y


def focus_major_so2_backward_with_folded_residual_out(
    grad_out: Tensor,
    w0_folded_t: Tensor,
    wpair_folded_t: Tensor,
    *,
    out: Tensor,
) -> Tensor:
    """Write ``grad_out @ (W.T + I)`` without seeding output copies."""
    if out.shape != grad_out.shape:
        raise ValueError("SO2 backward tensors must have identical shapes")
    if not out.is_contiguous():
        raise ValueError("SO2 backward output storage must be contiguous")
    if out.untyped_storage()._cdata == grad_out.untyped_storage()._cdata:
        raise ValueError("SO2 backward output must not alias its input")
    edge_count = grad_out.shape[0]
    grad_flat = grad_out.view(edge_count, FOCUS_COUNT, 10 * CHANNELS)
    out_flat = out.view(edge_count, FOCUS_COUNT, 10 * CHANNELS)
    split = 4 * CHANNELS
    for focus in range(FOCUS_COUNT):
        torch.mm(
            grad_flat[:, focus, :split],
            w0_folded_t[focus],
            out=out_flat[:, focus, :split],
        )
        torch.mm(
            grad_flat[:, focus, split:],
            wpair_folded_t[focus],
            out=out_flat[:, focus, split:],
        )
    return out


def run_structural_gate_forward(
    kernel: Any,
    residual: Tensor,
    y: Tensor,
    logits: Tensor,
    *,
    out: Tensor,
) -> Tensor:
    """Run the alias-safe CuTe forward into caller-owned storage."""
    rows = residual.shape[0] * residual.shape[1]
    kernel(
        residual.view(rows, 10 * CHANNELS),
        y.view(rows, 10 * CHANNELS),
        logits,
        out.view(rows, 10 * CHANNELS),
    )
    return out


def run_structural_gate_backward(
    kernel: Any,
    grad_out: Tensor,
    y: Tensor,
    logits: Tensor,
    grad_y: Tensor,
    *,
    grad_logits: Tensor | None,
    overwrite_logits: bool,
) -> Tensor:
    """Run gate backward, optionally replacing consumed logits with adjoints."""
    grad_logits_out = logits if overwrite_logits else grad_logits
    if grad_logits_out is None:
        raise ValueError("grad_logits storage is required when logits are preserved")
    rows = y.shape[0] * y.shape[1]
    kernel(
        grad_out.view(rows, 10 * CHANNELS),
        y.view(rows, 10 * CHANNELS),
        logits,
        grad_y.view(rows, 10 * CHANNELS),
        grad_logits_out,
    )
    return grad_logits_out
