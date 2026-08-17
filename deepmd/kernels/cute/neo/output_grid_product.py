# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Strict-FP32 CuTe middle contractions for supported Neo grid MLPs."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
)

import torch

from . import (
    runtime_policy,
)
from .runtime_policy import (
    PORTABLE_TILED_BACKEND,
    SUPPORTED_HIDDEN_CHANNELS,
    select_output_grid_backend,
)

COEFF_DIM = 16
N_FRAMES = 3
PACKED_COEFF_DIM = COEFF_DIM * N_FRAMES
GRID_SIZE = 152


def _exact_hidden_channels(
    left: torch.Tensor,
    n_frames: int,
) -> int | None:
    if left.ndim != 4 or left.shape[0] <= 0 or int(n_frames) != N_FRAMES:
        return None
    for hidden_channels in SUPPORTED_HIDDEN_CHANNELS:
        if tuple(left.shape[1:]) == (
            COEFF_DIM,
            1,
            N_FRAMES * hidden_channels,
        ):
            return hidden_channels
    return None


def _has_exact_contract(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> bool:
    hidden_channels = _exact_hidden_channels(left, n_frames)
    if (
        hidden_channels is None
        or not left.is_cuda
        or left.dtype != torch.float32
        or right.device != left.device
        or right.dtype != left.dtype
        or to_grid.device != left.device
        or from_grid.device != left.device
        or to_grid.dtype != left.dtype
        or from_grid.dtype != left.dtype
        or right.shape != left.shape
        or tuple(to_grid.shape) != (GRID_SIZE, PACKED_COEFF_DIM)
        or tuple(from_grid.shape) != (PACKED_COEFF_DIM, GRID_SIZE)
        or not left.is_contiguous()
        or not right.is_contiguous()
        or not to_grid.is_contiguous()
        or not from_grid.is_contiguous()
        or to_grid.requires_grad
        or from_grid.requires_grad
        or torch.is_autocast_enabled("cuda")
        or not runtime_policy.uses_strict_fp32_matmul()
    ):
        return False
    max_intermediate_values = left.shape[0] * GRID_SIZE * hidden_channels
    if max_intermediate_values > runtime_policy.INT32_MAX:
        return False
    compute_capability = tuple(torch.cuda.get_device_capability(left.device))
    return (
        select_output_grid_backend(compute_capability, hidden_channels)
        == PORTABLE_TILED_BACKEND
    )


def _validate_exact_contract(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> int:
    hidden_channels = _exact_hidden_channels(left, n_frames)
    if hidden_channels is None or not _has_exact_contract(
        left,
        right,
        to_grid,
        from_grid,
        n_frames,
    ):
        raise ValueError(
            "the fused output GridMLP kernel requires contiguous CUDA FP32 "
            "tensors with Neo's (D=16, F=1, frames=3, G=152, "
            "C in {96, 192}) contract"
        )
    return hidden_channels


def _output_grid_product_impl(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> torch.Tensor:
    hidden_channels = _validate_exact_contract(
        left,
        right,
        to_grid,
        from_grid,
        n_frames,
    )
    from .output_grid_kernels.cute_tiled_grid_product import (
        run_tiled_output_grid_product,
    )

    nodes = left.shape[0]
    left_flat = left.detach().view(nodes, PACKED_COEFF_DIM, hidden_channels)
    right_flat = right.detach().view(nodes, PACKED_COEFF_DIM, hidden_channels)
    compute_capability = tuple(torch.cuda.get_device_capability(left.device))
    use_sm80_c96_n48 = (
        hidden_channels == 96
        and compute_capability in runtime_policy.SM80_PROFILE_CAPABILITIES
        and runtime_policy.is_output_grid_fwd_sm80_c96_n48_enabled(compute_capability)
    )
    use_sm90_c96_asymmetric_panels = (
        hidden_channels == 96
        and compute_capability == (9, 0)
        and runtime_policy.is_output_grid_sm90_c96_asymmetric_panels_enabled(
            compute_capability
        )
    )
    out = run_tiled_output_grid_product(
        left_flat,
        right_flat,
        to_grid.detach(),
        from_grid.detach(),
        use_sm80_c96_n48=use_sm80_c96_n48,
        use_sm90_c96_asymmetric_panels=use_sm90_c96_asymmetric_panels,
    )
    return out.view_as(left)


def _output_grid_product_bwd_impl(
    grad_out: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_channels = _validate_exact_contract(
        left,
        right,
        to_grid,
        from_grid,
        n_frames,
    )
    if (
        grad_out.shape != left.shape
        or grad_out.dtype != left.dtype
        or grad_out.device != left.device
    ):
        raise ValueError("grad_out must match the fused output GridMLP output")
    from .output_grid_kernels.cute_tiled_grid_product import (
        run_tiled_output_grid_product_backward,
    )

    nodes = left.shape[0]
    compute_capability = tuple(torch.cuda.get_device_capability(left.device))
    use_sm80_c96_n48_panel = (
        hidden_channels == 96
        and compute_capability in runtime_policy.SM80_PROFILE_CAPABILITIES
        and runtime_policy.is_output_grid_bwd_sm80_c96_n48_panel_enabled(
            compute_capability
        )
    )
    use_sm90_c96_asymmetric_panels = (
        hidden_channels == 96
        and compute_capability == (9, 0)
        and runtime_policy.is_output_grid_sm90_c96_asymmetric_panels_enabled(
            compute_capability
        )
    )
    grad_left, grad_right = run_tiled_output_grid_product_backward(
        grad_out.detach()
        .contiguous()
        .view(
            nodes,
            PACKED_COEFF_DIM,
            hidden_channels,
        ),
        left.detach().view(nodes, PACKED_COEFF_DIM, hidden_channels),
        right.detach().view(nodes, PACKED_COEFF_DIM, hidden_channels),
        to_grid.detach(),
        from_grid.detach(),
        use_sm80_c96_n48_panel=use_sm80_c96_n48_panel,
        use_sm90_c96_asymmetric_panels=use_sm90_c96_asymmetric_panels,
    )
    return grad_left.view_as(left), grad_right.view_as(right)


_output_grid_product_op = torch.library.custom_op(
    "sezm_cute::output_grid_product",
    mutates_args=(),
)(_output_grid_product_impl)
_output_grid_product_bwd_op = torch.library.custom_op(
    "sezm_cute::output_grid_product_bwd",
    mutates_args=(),
)(_output_grid_product_bwd_impl)


@_output_grid_product_op.register_fake
def _output_grid_product_fake(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> torch.Tensor:
    del right, to_grid, from_grid, n_frames
    return torch.empty(left.shape, dtype=left.dtype, device=left.device)


@_output_grid_product_bwd_op.register_fake
def _output_grid_product_bwd_fake(
    grad_out: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    n_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del grad_out, to_grid, from_grid, n_frames
    return (
        torch.empty(left.shape, dtype=left.dtype, device=left.device),
        torch.empty(right.shape, dtype=right.dtype, device=right.device),
    )


def _setup_context(
    ctx: Any,
    inputs: tuple,
    output: torch.Tensor,
) -> None:
    del output
    left, right, to_grid, from_grid, n_frames = inputs
    ctx.save_for_backward(left, right, to_grid, from_grid)
    ctx.n_frames = int(n_frames)


def _backward(ctx: Any, grad_out: torch.Tensor) -> tuple:
    left, right, to_grid, from_grid = ctx.saved_tensors
    grad_left, grad_right = _output_grid_product_bwd_op(
        grad_out,
        left,
        right,
        to_grid,
        from_grid,
        ctx.n_frames,
    )
    return grad_left, grad_right, None, None, None


_output_grid_product_op.register_autograd(
    _backward,
    setup_context=_setup_context,
)


def output_grid_product_cute(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    *,
    n_frames: int,
) -> torch.Tensor:
    """Run one supported exact-shape fused grid contraction."""
    return _output_grid_product_op(
        left,
        right,
        to_grid,
        from_grid,
        int(n_frames),
    )


def maybe_run_cute_output_grid_product(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
    *,
    n_frames: int,
) -> torch.Tensor | None:
    """Return ``None`` unless the master gate and exact Neo contract match."""
    if not runtime_policy.is_cute_infer_enabled() or not _has_exact_contract(
        left,
        right,
        to_grid,
        from_grid,
        n_frames,
    ):
        return None
    return output_grid_product_cute(
        left,
        right,
        to_grid,
        from_grid,
        n_frames=n_frames,
    )


__all__ = [
    "maybe_run_cute_output_grid_product",
    "output_grid_product_cute",
]
