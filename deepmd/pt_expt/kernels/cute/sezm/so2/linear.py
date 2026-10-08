# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""SO2Linear helpers for the Neo CuTe SO2 path.

The in-place residual adjoint enforces FP32 operand dtypes. Strict FP32 GEMM
also requires the caller to select highest float32 matmul precision and disable
TF32; these helpers do not mutate process-wide backend settings.
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
REDUCED_COUNT = 10
CHANNELS = 32
M0_WIDTH = 4 * 32
PAIR_WIDTH = 6 * CHANNELS
FULL_WIDTH = REDUCED_COUNT * CHANNELS


def _validate_neo_so2_linear(so2_linear: Any) -> None:
    if (
        so2_linear.lmax != 3
        or so2_linear.mmax != 1
        or so2_linear.in_channels != 32
        or so2_linear.out_channels != 32
        or so2_linear.n_focus != 2
        or so2_linear.mlp_bias
    ):
        raise NotImplementedError("Neo CuTe SO2Linear expects lmax=3,mmax=1,F=2,C=32")


def run_neo_so2_linear_manual(
    so2_linear: Any,
    x_local: Any,
    *,
    add_residual: bool = False,
    per_focus_pair: bool = False,
) -> Any:
    """Run the fixed Neo SO2Linear block with two dense focus batched GEMMs."""
    import torch

    _validate_neo_so2_linear(so2_linear)
    if add_residual:
        w0, wpair = cached_neo_so2_linear_residual_weights(so2_linear)
    else:
        w0, wpair = cached_neo_so2_linear_weights(so2_linear)
    x_flat = x_local.reshape(x_local.shape[0], 2, 10 * 32).transpose(0, 1)
    out = x_local.new_empty(x_local.shape[0], 2, 10 * 32)
    out_t = out.transpose(0, 1)
    torch.bmm(x_flat[:, :, : 4 * 32], w0, out=out_t[:, :, : 4 * 32])
    if per_focus_pair:
        for focus in range(FOCUS_COUNT):
            torch.mm(
                x_flat[focus, :, M0_WIDTH:],
                wpair[focus],
                out=out_t[focus, :, M0_WIDTH:],
            )
    else:
        torch.bmm(x_flat[:, :, M0_WIDTH:], wpair, out=out_t[:, :, M0_WIDTH:])
    return out.reshape(x_local.shape[0], 2, 10, 32)


def cached_neo_so2_linear_residual_weights(so2_linear: Any) -> tuple[Any, Any]:
    """Return dense weights with the fixed SO2 residual folded in."""
    w0, wpair = cached_neo_so2_linear_weights(so2_linear)
    cache = getattr(so2_linear, "_deepmd_cute_neo_manual_residual_weights", None)
    cache_key = (w0.data_ptr(), wpair.data_ptr(), w0.dtype, w0.device)
    if (
        cache is not None
        and cache[0] is w0
        and cache[1] is wpair
        and cache[2] == cache_key
    ):
        return cache[3], cache[4]

    w0_residual = w0.clone()
    wpair_residual = wpair.clone()
    w0_residual.diagonal(dim1=-2, dim2=-1).add_(1.0)
    wpair_residual.diagonal(dim1=-2, dim2=-1).add_(1.0)
    so2_linear._deepmd_cute_neo_manual_residual_weights = (
        w0,
        wpair,
        cache_key,
        w0_residual,
        wpair_residual,
    )
    return w0_residual, wpair_residual


def cached_neo_so2_linear_weights(so2_linear: Any) -> tuple[Any, Any]:
    """Return cached dense block weights for Neo's fixed SO2Linear layout."""
    import torch

    cache = getattr(so2_linear, "_deepmd_cute_neo_manual_weights", None)
    cache_key = (
        so2_linear.weight_m0.data_ptr(),
        so2_linear.weight_m[0].data_ptr(),
        so2_linear.weight_m0._version,
        so2_linear.weight_m[0]._version,
        so2_linear.weight_m0.dtype,
        so2_linear.weight_m0.device,
    )
    if (
        cache is not None
        and cache[0] is so2_linear.weight_m0
        and cache[1] is so2_linear.weight_m[0]
        and cache[2] == cache_key
    ):
        return cache[3], cache[4]

    w0 = so2_linear.weight_m0.detach().view(4 * 32, 2, 4 * 32)
    w0 = w0.permute(1, 0, 2).contiguous()
    raw_pair = so2_linear.weight_m[0].detach().view(3 * 32, 2, 2 * 3 * 32)
    w_u = raw_pair[:, :, : 3 * 32]
    w_v = raw_pair[:, :, 3 * 32 :]
    wpair = torch.empty(
        2,
        2 * 3 * 32,
        2 * 3 * 32,
        device=raw_pair.device,
        dtype=raw_pair.dtype,
    )
    wpair[:, : 3 * 32, : 3 * 32] = w_u.permute(1, 0, 2)
    wpair[:, : 3 * 32, 3 * 32 :] = w_v.permute(1, 0, 2)
    wpair[:, 3 * 32 :, : 3 * 32] = -w_v.permute(1, 0, 2)
    wpair[:, 3 * 32 :, 3 * 32 :] = w_u.permute(1, 0, 2)
    wpair = wpair.contiguous()
    so2_linear._deepmd_cute_neo_manual_weights = (
        so2_linear.weight_m0,
        so2_linear.weight_m[0],
        cache_key,
        w0,
        wpair,
    )
    return w0, wpair


def _has_direct_cublas_layout(tensor: Tensor) -> bool:
    """Check only the direct-cuBLAS stride layouts used by PyTorch 2.10.

    This is a stride-layout predicate, not a complete cuBLAS eligibility test;
    dtype/device requirements are validated separately.
    """
    if tensor.ndim != 2:
        return False
    rows, columns = tensor.shape
    stride0, stride1 = tensor.stride()
    return (stride0 == 1 and stride1 >= max(1, rows)) or (
        stride1 == 1 and stride0 >= max(1, columns)
    )


def _contiguous_tensors_overlap(lhs: Tensor, rhs: Tensor) -> bool:
    """Return exact byte overlap for validated, nonempty contiguous tensors."""
    if lhs.device != rhs.device:
        return False
    if lhs.device.type == "meta":
        return torch._C._overlaps(lhs, rhs)

    lhs_start = lhs.data_ptr()
    rhs_start = rhs.data_ptr()
    lhs_stop = lhs_start + lhs.numel() * lhs.element_size()
    rhs_stop = rhs_start + rhs.numel() * rhs.element_size()
    return lhs_start < rhs_stop and rhs_start < lhs_stop


def _validate_edge_focus(tensor: Tensor, name: str) -> None:
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {tensor.dtype}")
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (
        FOCUS_COUNT,
        REDUCED_COUNT,
        CHANNELS,
    ):
        raise ValueError(
            f"{name} must have shape (E,2,10,32), got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] <= 0:
        raise ValueError(f"{name} requires E > 0")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must use the canonical contiguous SO2 layout")


def _validate_weight(
    weight: Tensor,
    name: str,
    width: int,
    device: torch.device,
) -> None:
    if weight.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {weight.dtype}")
    expected_shape = (FOCUS_COUNT, width, width)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {tuple(weight.shape)}"
        )
    if weight.device != device:
        raise ValueError(f"{name} must be on {device}, got {weight.device}")
    if not weight.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def neo_so2_linear_backward_residual_inplace(
    residual: Tensor,
    grad_out: Tensor,
    w0_t: Tensor,
    wpair_t: Tensor,
) -> Tensor:
    """Overwrite a dead residual with ``grad_out @ W.T + residual``."""
    _validate_edge_focus(residual, "residual")
    _validate_edge_focus(grad_out, "grad_out")
    if residual.shape != grad_out.shape:
        raise ValueError("residual and grad_out shapes must match")
    if residual.device != grad_out.device:
        raise ValueError("residual and grad_out devices must match")
    if _contiguous_tensors_overlap(residual, grad_out):
        raise ValueError(
            "residual and grad_out must not alias; keep the final layer out-of-place"
        )

    _validate_weight(w0_t, "w0_t", M0_WIDTH, residual.device)
    _validate_weight(wpair_t, "wpair_t", PAIR_WIDTH, residual.device)
    for name, weight in (("w0_t", w0_t), ("wpair_t", wpair_t)):
        if _contiguous_tensors_overlap(residual, weight):
            raise ValueError(f"residual and {name} must not alias")

    edge_count = residual.shape[0]
    residual_flat = residual.view(edge_count, FOCUS_COUNT, FULL_WIDTH)
    grad_flat = grad_out.view(edge_count, FOCUS_COUNT, FULL_WIDTH)
    for focus in range(FOCUS_COUNT):
        for start, stop, weight in (
            (0, M0_WIDTH, w0_t[focus]),
            (M0_WIDTH, FULL_WIDTH, wpair_t[focus]),
        ):
            residual_block = residual_flat[:, focus, start:stop]
            grad_block = grad_flat[:, focus, start:stop]
            if not _has_direct_cublas_layout(residual_block) or not (
                _has_direct_cublas_layout(grad_block)
                and _has_direct_cublas_layout(weight)
            ):
                raise ValueError("SO2 block layout would require cuBLAS staging")
            residual_block.addmm_(
                grad_block,
                weight,
                beta=1.0,
                alpha=1.0,
            )
    return residual
