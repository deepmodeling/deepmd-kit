# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Packed strict-FP32 CuTe grid product for Neo's F=2 K1 branch."""

from __future__ import (
    annotations,
)

from collections.abc import (
    Callable,
)
from functools import (
    lru_cache,
)
from typing import (
    Any,
)

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.runtime import (
    make_fake_compact_tensor,
    make_fake_stream,
)

from .. import (
    runtime_policy,
)

# CuTe JIT functions use DSL-inferred argument and return types.
# ruff: noqa: ANN001, ANN201, TC003


PACKED_COEFF_DIM = 48
HIDDEN_CHANNELS = 64
GRID_SIZE = 152
FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}
_SUPPORTED_CAPABILITIES = runtime_policy.SM80_PROFILE_CAPABILITIES | {
    runtime_policy.SM90_CAPABILITY
}


def _make_grid_operation(
    operation_type: type[Any],
    *,
    panel_adjoint: bool = False,
) -> Any:
    from ..output_grid_kernels import cute_tiled_grid_product as tiled

    # F=2 and C=32 form one complete 64-channel panel. Bypass only the
    # shared readout policy; the tiled implementation itself is unchanged.
    operation = operation_type.__new__(operation_type)
    operation.hidden_channels = HIDDEN_CHANNELS
    operation.tile_k = tiled.TILE_K
    operation.sm80_c96_n48_panel = bool(panel_adjoint)
    operation.cta_tiler = (tiled.TILE_M, tiled.TILE_N, operation.tile_k)
    operation.channel_tile_start = 0
    operation.channel_tiles = 1
    operation.has_channel_residue = False
    operation.cta_sync_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=tiled.THREADS,
    )
    return operation


def _validate_compile_target(
    device_index: int,
    compute_capability: tuple[int, int],
) -> None:
    import torch

    actual = tuple(torch.cuda.get_device_capability(device_index))
    if actual != tuple(compute_capability):
        raise ValueError("compile target does not match the selected CUDA device")
    if actual not in _SUPPORTED_CAPABILITIES:
        raise ValueError(
            "packed Neo message-grid product requires the SM80-family profile or sm90"
        )


def _fake_inputs() -> tuple[Any, Any, Any]:
    nodes = cute.sym_int64()
    coeff = make_fake_compact_tensor(
        cutlass.Float32,
        (nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS),
        stride_order=(2, 1, 0),
        **FAKE_TENSOR_KW,
    )
    to_grid = make_fake_compact_tensor(
        cutlass.Float32,
        (GRID_SIZE, PACKED_COEFF_DIM),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    from_grid = make_fake_compact_tensor(
        cutlass.Float32,
        (PACKED_COEFF_DIM, GRID_SIZE),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    return coeff, to_grid, from_grid


def compile_message_grid_product_forward(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    """Compile the F=2 packed product with a symbolic node count."""
    import torch

    _validate_compile_target(device_index, compute_capability)
    from ..output_grid_kernels.cute_tiled_grid_product import (
        TiledOutputGridProductForward,
    )

    coeff, to_grid, from_grid = _fake_inputs()
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    with torch.cuda.device(device_index):
        return cute.compile(
            _make_grid_operation(TiledOutputGridProductForward),
            coeff,
            coeff,
            to_grid,
            from_grid,
            coeff,
            stream,
            options="--enable-tvm-ffi",
        )


def compile_message_grid_product_backward(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    """Compile both packed input adjoints with a symbolic node count."""
    import torch

    _validate_compile_target(device_index, compute_capability)
    from ..output_grid_kernels.cute_tiled_grid_product import (
        TiledOutputGridProductBackward,
    )

    coeff, to_grid, from_grid = _fake_inputs()
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    with torch.cuda.device(device_index):
        return cute.compile(
            _make_grid_operation(
                TiledOutputGridProductBackward,
                panel_adjoint=(
                    compute_capability in runtime_policy.SM80_PROFILE_CAPABILITIES
                ),
            ),
            coeff,
            coeff,
            coeff,
            to_grid,
            from_grid,
            coeff,
            coeff,
            stream,
            options="--enable-tvm-ffi",
        )


@lru_cache(maxsize=8)
def _compiled_forward(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    return compile_message_grid_product_forward(device_index, compute_capability)


@lru_cache(maxsize=8)
def _compiled_backward(
    device_index: int,
    compute_capability: tuple[int, int],
) -> Callable:
    return compile_message_grid_product_backward(device_index, compute_capability)


def _compile_identity(tensor) -> tuple[int, tuple[int, int]]:
    import torch

    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return int(device_index), tuple(torch.cuda.get_device_capability(device_index))


def _validate_tensors(left, right, to_grid, from_grid, grad_out=None) -> None:
    import torch

    floating = (left, right, to_grid, from_grid)
    if (
        tuple(left.shape[1:]) != (PACKED_COEFF_DIM, HIDDEN_CHANNELS)
        or left.shape[0] <= 0
        or right.shape != left.shape
        or tuple(to_grid.shape) != (GRID_SIZE, PACKED_COEFF_DIM)
        or tuple(from_grid.shape) != (PACKED_COEFF_DIM, GRID_SIZE)
        or any(not tensor.is_cuda for tensor in floating)
        or any(tensor.device != left.device for tensor in floating)
        or any(tensor.dtype != torch.float32 for tensor in floating)
        or any(not tensor.is_contiguous() for tensor in floating)
        or any(tensor.data_ptr() % 16 != 0 for tensor in floating)
        or tuple(torch.cuda.get_device_capability(left.device))
        not in _SUPPORTED_CAPABILITIES
        or not runtime_policy.uses_strict_fp32_matmul()
    ):
        raise ValueError(
            "packed message-grid product requires contiguous SM80-family/SM90 FP32 "
            "left/right=(N,48,64), to_grid=(152,48), and from_grid=(48,152)"
        )
    if grad_out is not None and (
        grad_out.shape != left.shape
        or grad_out.device != left.device
        or grad_out.dtype != torch.float32
        or not grad_out.is_contiguous()
        or grad_out.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "packed message-grid grad_out must be contiguous and match left"
        )


def run_message_grid_product(left, right, to_grid, from_grid):
    """Run the F=2 product without materializing projection-layout clones."""
    import torch

    _validate_tensors(left, right, to_grid, from_grid)
    out = torch.empty_like(left)
    with torch.cuda.device(left.device):
        _compiled_forward(*_compile_identity(left))(
            left,
            right,
            to_grid,
            from_grid,
            out,
        )
    return out


def run_message_grid_product_backward(
    grad_out,
    left,
    right,
    to_grid,
    from_grid,
):
    """Run both F=2 product input adjoints in the packed layout."""
    import torch

    _validate_tensors(left, right, to_grid, from_grid, grad_out)
    grad_left = torch.empty_like(left)
    grad_right = torch.empty_like(right)
    with torch.cuda.device(left.device):
        _compiled_backward(*_compile_identity(left))(
            grad_out,
            left,
            right,
            to_grid,
            from_grid,
            grad_left,
            grad_right,
        )
    return grad_left, grad_right


__all__ = [
    "compile_message_grid_product_backward",
    "compile_message_grid_product_forward",
    "run_message_grid_product",
    "run_message_grid_product_backward",
]
