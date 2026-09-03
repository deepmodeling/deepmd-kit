# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Runtime contract for exact-shape Neo Phase-C backward."""

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    Any,
)

import torch

from ..compile_cache import (
    device_aware_lru_cache,
)
from .kernels.phase_c_backward import (
    DEGREE_COUNT,
    FOCUS_CHANNELS,
    HIDDEN,
    N_FOCUS,
    REDUCED_COUNT,
    compile_neo_phase_c_backward_layout,
)
from .wigner_layout import PACKED_VALUE_COUNT as PACKED_WIGNER_VALUES

REQUIRED_ALIGNMENT = 16


@dataclass(frozen=True)
class NeoPhaseCBackwardLayoutOutputs:
    """Caller-owned outputs and scratch for one fused Phase-C invocation."""

    grad_stack: torch.Tensor
    grad_wigner_dt: torch.Tensor
    grad_logits: torch.Tensor
    grad_edge: torch.Tensor
    grad_z_partial: torch.Tensor
    grad_z: torch.Tensor
    grad_focus_src: torch.Tensor


def _storage_id(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage()._cdata


def _is_exact_view(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Return whether two tensors name the same logical and physical view."""
    return (
        _storage_id(lhs) == _storage_id(rhs)
        and lhs.data_ptr() == rhs.data_ptr()
        and lhs.storage_offset() == rhs.storage_offset()
        and lhs.shape == rhs.shape
        and lhs.stride() == rhs.stride()
        and lhs.dtype == rhs.dtype
        and lhs.device == rhs.device
    )


def _tensor_byte_region(tensor: torch.Tensor) -> tuple[int, int] | None:
    """Return the physical byte range of a compact runtime tensor."""
    if tensor.numel() == 0 or tensor.device.type == "meta":
        return None
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _tensor_regions_overlap(
    lhs: torch.Tensor,
    lhs_region: tuple[int, int] | None,
    rhs: torch.Tensor,
    rhs_region: tuple[int, int] | None,
) -> bool:
    """Compare precomputed physical regions, including external storage views."""
    if lhs.device != rhs.device:
        return False
    if lhs.device.type == "meta":
        return torch._C._overlaps(lhs, rhs)
    if lhs_region is None or rhs_region is None:
        return False
    lhs_start, lhs_stop = lhs_region
    rhs_start, rhs_stop = rhs_region
    return lhs_start < rhs_stop and rhs_start < lhs_stop


def _require_alignment(
    name: str,
    tensor: torch.Tensor,
    alignment: int = REQUIRED_ALIGNMENT,
) -> None:
    """Enforce the alignment promised to CuTe by ``assumed_align``."""
    if tensor.device.type == "meta":
        return

    byte_offset = tensor.storage_offset() * tensor.element_size()
    pointer_remainder = tensor.data_ptr() % alignment
    storage_remainder = tensor.untyped_storage().data_ptr() % alignment
    offset_remainder = byte_offset % alignment
    if pointer_remainder or storage_remainder or offset_remainder:
        raise ValueError(
            f"{name} must be {alignment}-byte aligned; got data pointer "
            f"remainder {pointer_remainder}, storage pointer remainder "
            f"{storage_remainder}, and byte storage-offset remainder "
            f"{offset_remainder}"
        )


def _require_tensor(
    name: str,
    tensor: torch.Tensor,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> None:
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be compact")
    _require_alignment(name, tensor)


@device_aware_lru_cache(maxsize=32)
def _compile_layout_boundary(
    focus_eps: float,
    focus_tau: float,
    focus_label_smoothing: float,
) -> Any:
    return compile_neo_phase_c_backward_layout(
        focus_eps=focus_eps,
        focus_tau=focus_tau,
        focus_label_smoothing=focus_label_smoothing,
    )


class CuteNeoPhaseCBackwardLayout:
    """Callable for the fused node-owned Phase-C boundary.

    One invocation replaces Phase-C backward, envelope-softmax backward plus
    its z reduction, and focus-source backward.  The Phase-C residual adjoint
    is the unmodified ``grad_out`` input and remains owned by the caller.

    Inputs retain their model-native ranks. ``grad_stack`` aliases compact
    edge-major ``stack`` storage and is written only after the complete source
    fragment has been consumed. ``grad_focus_src`` is compact ``(F,E,C)``.
    """

    def __init__(
        self,
        *,
        focus_eps: float,
        focus_tau: float,
        focus_label_smoothing: float,
    ) -> None:
        self._compiled = _compile_layout_boundary(
            float(focus_eps),
            float(focus_tau),
            float(focus_label_smoothing),
        )

    def __call__(
        self,
        grad_out: torch.Tensor,
        stack: torch.Tensor,
        wigner_dt: torch.Tensor,
        alpha: torch.Tensor,
        focus_alpha: torch.Tensor,
        dst_ptr: torch.Tensor,
        rotate_inv_rescale: torch.Tensor,
        edge_gate: torch.Tensor,
        z_bias_raw: torch.Tensor,
        group_max: torch.Tensor,
        denom: torch.Tensor,
        focus_src: torch.Tensor,
        focus_weight: torch.Tensor,
        focus_scale: torch.Tensor,
        outputs: NeoPhaseCBackwardLayoutOutputs,
    ) -> NeoPhaseCBackwardLayoutOutputs:
        edge_count = stack.shape[0]
        node_count = grad_out.shape[0]
        device = stack.device
        stack_shape = (
            edge_count,
            N_FOCUS,
            REDUCED_COUNT,
            FOCUS_CHANNELS,
        )

        _require_tensor(
            "grad_out",
            grad_out,
            (node_count, DEGREE_COUNT, HIDDEN),
            device=device,
        )
        _require_tensor("stack", stack, stack_shape, device=device)
        _require_tensor(
            "wigner_dt",
            wigner_dt,
            (edge_count, PACKED_WIGNER_VALUES),
            device=device,
        )
        _require_tensor("alpha", alpha, (edge_count, N_FOCUS), device=device)
        _require_tensor(
            "focus_alpha", focus_alpha, (edge_count, N_FOCUS), device=device
        )
        _require_tensor(
            "dst_ptr",
            dst_ptr,
            (node_count + 1,),
            device=device,
            dtype=torch.int32,
        )
        _require_tensor(
            "rotate_inv_rescale",
            rotate_inv_rescale,
            (DEGREE_COUNT,),
            device=device,
        )
        _require_tensor("edge_gate", edge_gate, (edge_count,), device=device)
        _require_tensor("z_bias_raw", z_bias_raw, (N_FOCUS,), device=device)
        _require_tensor("group_max", group_max, (node_count, N_FOCUS), device=device)
        _require_tensor("denom", denom, (node_count, N_FOCUS), device=device)
        _require_tensor(
            "focus_src",
            focus_src,
            (edge_count, N_FOCUS, FOCUS_CHANNELS),
            device=device,
        )
        _require_tensor(
            "focus_weight",
            focus_weight,
            (FOCUS_CHANNELS, N_FOCUS),
            device=device,
        )
        _require_tensor(
            "focus_scale",
            focus_scale,
            (N_FOCUS, FOCUS_CHANNELS),
            device=device,
        )

        _require_tensor(
            "outputs.grad_stack",
            outputs.grad_stack,
            stack_shape,
            device=device,
        )
        if not _is_exact_view(outputs.grad_stack, stack):
            raise ValueError("outputs.grad_stack must be the exact in-place stack view")
        _require_tensor(
            "outputs.grad_wigner_dt",
            outputs.grad_wigner_dt,
            (edge_count, PACKED_WIGNER_VALUES),
            device=device,
        )
        _require_tensor(
            "outputs.grad_logits",
            outputs.grad_logits,
            (edge_count, N_FOCUS),
            device=device,
        )
        _require_tensor(
            "outputs.grad_edge", outputs.grad_edge, (edge_count,), device=device
        )
        _require_tensor(
            "outputs.grad_z_partial",
            outputs.grad_z_partial,
            (node_count, N_FOCUS),
            device=device,
        )
        _require_tensor("outputs.grad_z", outputs.grad_z, (N_FOCUS,), device=device)
        _require_tensor(
            "outputs.grad_focus_src",
            outputs.grad_focus_src,
            (N_FOCUS, edge_count, FOCUS_CHANNELS),
            device=device,
        )

        input_tensors = tuple(
            (name, tensor, _tensor_byte_region(tensor))
            for name, tensor in (
                ("grad_out", grad_out),
                ("stack", stack),
                ("wigner_dt", wigner_dt),
                ("alpha", alpha),
                ("focus_alpha", focus_alpha),
                ("dst_ptr", dst_ptr),
                ("rotate_inv_rescale", rotate_inv_rescale),
                ("edge_gate", edge_gate),
                ("z_bias_raw", z_bias_raw),
                ("group_max", group_max),
                ("denom", denom),
                ("focus_src", focus_src),
                ("focus_weight", focus_weight),
                ("focus_scale", focus_scale),
            )
        )
        output_tensors = tuple(
            (
                f"outputs.{field_name}",
                getattr(outputs, field_name),
                _tensor_byte_region(getattr(outputs, field_name)),
            )
            for field_name in (
                "grad_stack",
                "grad_wigner_dt",
                "grad_logits",
                "grad_edge",
                "grad_z_partial",
                "grad_z",
                "grad_focus_src",
            )
        )

        for output_index, (
            output_name,
            output,
            output_region,
        ) in enumerate(output_tensors):
            for other_name, other, other_region in output_tensors[output_index + 1 :]:
                if _tensor_regions_overlap(
                    output,
                    output_region,
                    other,
                    other_region,
                ):
                    raise ValueError(
                        f"{output_name} must not overlap output {other_name}"
                    )
            for input_name, input_tensor, input_region in input_tensors:
                if not _tensor_regions_overlap(
                    output,
                    output_region,
                    input_tensor,
                    input_region,
                ):
                    continue
                is_stack_adjoint = (
                    output_name == "outputs.grad_stack" and input_name == "stack"
                )
                if is_stack_adjoint:
                    continue
                raise ValueError(f"{output_name} must not overlap input {input_name}")

        self._compiled(
            grad_out,
            stack,
            wigner_dt,
            alpha,
            focus_alpha,
            dst_ptr,
            rotate_inv_rescale,
            edge_gate,
            z_bias_raw,
            group_max,
            denom,
            focus_src,
            focus_weight,
            focus_scale,
            outputs.grad_stack,
            outputs.grad_wigner_dt,
            outputs.grad_logits,
            outputs.grad_edge,
            outputs.grad_z_partial,
            outputs.grad_z,
            outputs.grad_focus_src,
        )
        return outputs
