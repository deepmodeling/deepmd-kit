# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runtime wrapper for the source-CSR node-tiled radial/Phase-A backward."""

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

from .compile_cache import (
    device_aware_lru_cache,
)
from .k1_wigner_layout import (
    PACKED_VALUE_COUNT,
)

if TYPE_CHECKING:
    from torch import (
        Tensor,
    )


DEGREE_COUNT = 16
REDUCED_COUNT = 10
HIDDEN = 64
FOCUS_COUNT = 2
FOCUS_HIDDEN = 32
PACKED_WIGNER_VALUES = PACKED_VALUE_COUNT
RADIAL_WIDTH = 4 * FOCUS_HIDDEN
COMPACT_WIDTH = 25
PROJECTION_INPUT_WIDTH = COMPACT_WIDTH + FOCUS_COUNT


@dataclass(frozen=True)
class NeoSourceCSR:
    """Indirect source CSR over the unchanged physical edge order."""

    source_order: Tensor
    source_ptr: Tensor


@dataclass(frozen=True)
class NeoRadialPhaseABackwardNodeResult:
    """Output buffers populated by the node-tiled backward."""

    grad_x_wide: Tensor
    grad_d_full: Tensor
    grad_radial_m0: Tensor


def build_source_csr(
    src: Tensor,
    node_count: int,
    *,
    validate_sources: bool = False,
) -> NeoSourceCSR:
    """Build indirect source CSR without changing the physical edge order.

    Source bounds are always checked before constructing the CSR. Setting
    ``validate_sources=True`` reports an eager ``ValueError`` and therefore
    synchronizes when ``src`` is CUDA; the default uses an asynchronous device
    assertion. Callers should build this once with the edge cache and retain
    both tensors.
    """
    import torch

    if src.dim() != 1:
        raise ValueError("src must be one-dimensional")
    if src.dtype not in (torch.int32, torch.int64):
        raise TypeError("src must have dtype int32 or int64")
    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    if src.numel() > 2**31 - 1:
        raise ValueError("source CSR int32 indexing requires E <= 2**31 - 1")

    src = src.contiguous()
    if src.numel() != 0:
        valid = torch.all((src >= 0) & (src < node_count))
        message = "source-CSR backward requires source indices in [0, node_count)"
        if validate_sources:
            if not bool(valid):
                raise ValueError(message)
        else:
            torch._assert_async(
                valid,
                message,
            )

    source_order_i64 = torch.argsort(src, stable=True)
    sorted_src = src.index_select(0, source_order_i64)
    boundaries = torch.arange(
        node_count + 1,
        device=src.device,
        dtype=src.dtype,
    )
    source_ptr = torch.searchsorted(
        sorted_src,
        boundaries,
        out_int32=True,
    ).contiguous()
    source_order = source_order_i64.to(dtype=torch.int32).contiguous()
    return NeoSourceCSR(source_order=source_order, source_ptr=source_ptr)


@device_aware_lru_cache(maxsize=4)
def _compile_node_tiled() -> Any:
    from .k1_kernels.cute_neo_radial_phase_a_backward_node import (
        compile_neo_radial_phase_a_backward_node_tiled,
    )

    return compile_neo_radial_phase_a_backward_node_tiled()


def _expect_tensor(
    name: str,
    tensor: Tensor,
    shape: tuple[int, ...],
    *,
    device: Any,
    dtype: Any,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def prepare_batched_radial_projection_weight(
    combined_weight: Tensor,
    attention_radial_weight: Tensor,
) -> Tensor:
    """Precombine the compact and attention adjoints for one FP32 GEMM."""
    import torch

    device = combined_weight.device
    _expect_tensor(
        "combined_weight",
        combined_weight,
        (RADIAL_WIDTH, COMPACT_WIDTH),
        device=device,
        dtype=torch.float32,
    )
    _expect_tensor(
        "attention_radial_weight",
        attention_radial_weight,
        (FOCUS_HIDDEN, FOCUS_COUNT),
        device=device,
        dtype=torch.float32,
    )

    projection_weight = combined_weight.new_zeros(
        (PROJECTION_INPUT_WIDTH, RADIAL_WIDTH)
    )
    projection_weight[:COMPACT_WIDTH].copy_(combined_weight.transpose(0, 1))
    projection_weight[COMPACT_WIDTH:, :FOCUS_HIDDEN].copy_(
        attention_radial_weight.transpose(0, 1)
    )
    return projection_weight


def _project_batched_radial_adjoint(
    projection_weight: Tensor,
    grad_radial_m0: Tensor,
    consumed_workspace: Tensor,
) -> None:
    """Project the 27-column adjoint packed in consumed edge scratch."""
    import torch

    if torch.backends.cuda.matmul.allow_tf32:
        raise RuntimeError("strict FP32 requires allow_tf32=False")
    if torch.get_float32_matmul_precision() != "highest":
        raise RuntimeError("strict FP32 requires float32 matmul precision 'highest'")
    edge_count = consumed_workspace.shape[0]
    device = consumed_workspace.device
    tensors = (
        (
            "projection_weight",
            projection_weight,
            (PROJECTION_INPUT_WIDTH, RADIAL_WIDTH),
        ),
        ("grad_radial_m0", grad_radial_m0, (edge_count, RADIAL_WIDTH)),
        (
            "consumed_workspace",
            consumed_workspace,
            (edge_count, FOCUS_COUNT * REDUCED_COUNT * FOCUS_HIDDEN),
        ),
    )
    for name, tensor, shape in tensors:
        _expect_tensor(
            name,
            tensor,
            shape,
            device=device,
            dtype=torch.float32,
        )

    projection_input = consumed_workspace[:, :PROJECTION_INPUT_WIDTH]
    torch.mm(projection_input, projection_weight, out=grad_radial_m0)


def _validate_csr_values(
    source_order: Tensor,
    source_ptr: Tensor,
    edge_count: int,
) -> None:
    import torch

    valid = (source_ptr[0] == 0) & (source_ptr[-1] == edge_count)
    if source_ptr.numel() > 1:
        valid = valid & torch.all(source_ptr[1:] >= source_ptr[:-1])
    if not bool(valid):
        raise ValueError(
            "source_ptr must be nondecreasing, begin at zero, and end at E"
        )
    expected = torch.arange(
        edge_count,
        device=source_order.device,
        dtype=source_order.dtype,
    )
    if not torch.equal(torch.sort(source_order).values, expected):
        raise ValueError("source_order must be a permutation of [0, E)")


def run_neo_radial_phase_a_backward_node_tiled(
    grad_out_focus: Tensor,
    grad_logits: Tensor,
    radial_state: Tensor,
    channel_basis: Tensor,
    x_wide: Tensor,
    source_order: Tensor,
    source_ptr: Tensor,
    d_full: Tensor,
    *,
    grad_focus_src_focus: Tensor,
    batched_radial_projection_weight: Tensor,
    grad_x_wide: Tensor | None = None,
    grad_d_full: Tensor | None = None,
    grad_radial_m0: Tensor | None = None,
    validate_csr: bool = False,
) -> NeoRadialPhaseABackwardNodeResult:
    """Run the node-owned backward over indirect source CSR.

    The kernel recomputes Phase A, fuses the focus-source adjoint, uses
    four-lane warp reductions with a 68-float shared row pitch, and packs the
    27-column radial projection input for one strict-FP32 matrix call.
    ``grad_out_focus`` is repacked in place as projection workspace and must
    not be reused after this function returns.
    """
    import torch

    if not x_wide.is_cuda:
        raise ValueError("node-tiled radial Phase-A backward requires CUDA tensors")
    if x_wide.dtype != torch.float32:
        raise TypeError("node-tiled radial Phase-A backward specializes float32")
    if not x_wide.is_contiguous() or x_wide.dim() != 2:
        raise ValueError("x_wide must be a contiguous two-dimensional tensor")

    device = x_wide.device
    dtype = x_wide.dtype
    node_count = x_wide.shape[0]
    edge_count = grad_out_focus.shape[0]
    _expect_tensor(
        "x_wide",
        x_wide,
        (node_count, DEGREE_COUNT * HIDDEN),
        device=device,
        dtype=dtype,
    )
    _expect_tensor(
        "grad_out_focus",
        grad_out_focus,
        (edge_count, FOCUS_COUNT * REDUCED_COUNT * FOCUS_HIDDEN),
        device=device,
        dtype=dtype,
    )
    _expect_tensor(
        "grad_focus_src_focus",
        grad_focus_src_focus,
        (FOCUS_COUNT, edge_count, FOCUS_HIDDEN),
        device=device,
        dtype=dtype,
    )
    _expect_tensor(
        "grad_logits",
        grad_logits,
        (edge_count, FOCUS_COUNT),
        device=device,
        dtype=dtype,
    )
    _expect_tensor(
        "radial_state",
        radial_state,
        (edge_count, COMPACT_WIDTH),
        device=device,
        dtype=dtype,
    )
    _expect_tensor(
        "batched_radial_projection_weight",
        batched_radial_projection_weight,
        (PROJECTION_INPUT_WIDTH, RADIAL_WIDTH),
        device=device,
        dtype=dtype,
    )
    _expect_tensor(
        "channel_basis",
        channel_basis,
        (HIDDEN,),
        device=device,
        dtype=dtype,
    )
    _expect_tensor(
        "source_order",
        source_order,
        (edge_count,),
        device=device,
        dtype=torch.int32,
    )
    _expect_tensor(
        "source_ptr",
        source_ptr,
        (node_count + 1,),
        device=device,
        dtype=torch.int32,
    )
    _expect_tensor(
        "d_full",
        d_full,
        (edge_count, PACKED_WIGNER_VALUES),
        device=device,
        dtype=dtype,
    )
    if node_count == 0 and edge_count != 0:
        raise ValueError("a non-empty edge list requires at least one source node")
    torch._assert_async(
        source_ptr[0] == 0,
        "source-CSR backward requires source_ptr[0] == 0",
    )
    torch._assert_async(
        source_ptr[-1] == edge_count,
        "source-CSR backward requires source_ptr[-1] == edge_count",
    )
    if source_ptr.numel() > 1:
        torch._assert_async(
            torch.all(source_ptr[1:] >= source_ptr[:-1]),
            "source-CSR backward requires nondecreasing source_ptr",
        )
    if source_order.numel() != 0:
        torch._assert_async(
            torch.all((source_order >= 0) & (source_order < edge_count)),
            "source-CSR backward requires source_order entries in [0, E)",
        )
    if validate_csr:
        _validate_csr_values(source_order, source_ptr, edge_count)

    if grad_x_wide is None:
        grad_x_wide = torch.empty_like(x_wide)
    else:
        _expect_tensor(
            "grad_x_wide",
            grad_x_wide,
            (node_count, DEGREE_COUNT * HIDDEN),
            device=device,
            dtype=dtype,
        )
    if grad_d_full is None:
        grad_d_full = torch.empty_like(d_full)
    else:
        _expect_tensor(
            "grad_d_full",
            grad_d_full,
            (edge_count, PACKED_WIGNER_VALUES),
            device=device,
            dtype=dtype,
        )
    if grad_radial_m0 is None:
        grad_radial_m0 = torch.empty(
            (edge_count, RADIAL_WIDTH),
            device=device,
            dtype=dtype,
        )
    else:
        _expect_tensor(
            "grad_radial_m0",
            grad_radial_m0,
            (edge_count, RADIAL_WIDTH),
            device=device,
            dtype=dtype,
        )

    result = NeoRadialPhaseABackwardNodeResult(
        grad_x_wide=grad_x_wide,
        grad_d_full=grad_d_full,
        grad_radial_m0=grad_radial_m0,
    )
    if edge_count == 0:
        grad_x_wide.zero_()
        return result

    with torch.cuda.device(device):
        kernel = _compile_node_tiled()
        kernel(
            grad_out_focus,
            grad_focus_src_focus,
            grad_logits,
            radial_state,
            channel_basis,
            x_wide,
            source_order,
            source_ptr,
            d_full,
            grad_x_wide,
            grad_d_full,
        )
        _project_batched_radial_adjoint(
            batched_radial_projection_weight,
            grad_radial_m0,
            grad_out_focus,
        )
    return result
