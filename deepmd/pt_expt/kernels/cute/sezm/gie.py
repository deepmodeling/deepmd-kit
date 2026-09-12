# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
# pyright: reportMissingImports=false
# ruff: noqa: ANN001
"""Opt-in CuTe fusion for the DPA4 geometric initial embedding.

The eager implementation materializes both ``radial_value_for_row`` and
``non_scalar_message`` with shape ``(E, D - 1, C)`` before reducing by
destination. This module computes the same strict-FP32 expression directly
into ``(N, D, C)`` using the destination-sorted edge list. Its first backward
produces radial, zonal/Wigner, source-gate, and degree-normalization gradients
without an edge-by-row-by-channel temporary.

``DP_CUTE_INFER`` is the master opt-in. The SM80/SM86 path enables this
fusion by default; ``DP_CUTE_GIE=0`` disables it explicitly.
"""

from __future__ import (
    annotations,
)

import threading
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch
from torch import (
    Tensor,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    from cuda.bindings.driver import CUstream  # noqa: TC002
    from cutlass.cute.runtime import (
        from_dlpack,
    )

    SEZM_CUTE_GIE_AVAILABLE = True
except Exception:  # pragma: no cover - import guard for non-CuTe environments
    SEZM_CUTE_GIE_AVAILABLE = False


def is_cute_gie_enabled(device: torch.device | None = None) -> bool:
    """Return whether the architecture-selected GIE path is enabled."""
    from .runtime_policy import (
        is_gie_enabled,
    )

    if device is None:
        if not torch.cuda.is_available():
            return False
        device_index = torch.cuda.current_device()
    else:
        if device.type != "cuda":
            return False
        device_index = device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
    return is_gie_enabled(tuple(torch.cuda.get_device_capability(device_index)))


def _backward_compile_key(
    device_identity: tuple[int, int, int],
    lmax: int,
    channels: int,
    has_gate: bool,
    radial_stride: tuple[int, ...],
    dst_dtype: torch.dtype,
) -> tuple[Any, ...]:
    """Build an ABI-complete GIE backward compilation key."""
    return (
        "gie_bwd",
        *device_identity,
        lmax,
        channels,
        has_gate,
        tuple(radial_stride),
        dst_dtype,
    )


def _degree_slots(lmax: int, *, device: torch.device) -> Tensor:
    degrees = torch.arange(1, lmax + 1, device=device, dtype=torch.long)
    return torch.repeat_interleave(degrees - 1, 2 * degrees + 1)


def _standard_index_contract(module: Any, lmax: int, row_count: int) -> bool:
    rows = getattr(module, "non_scalar_row_index", None)
    slots = getattr(module, "radial_slot_index_for_row", None)
    if not isinstance(rows, Tensor) or not isinstance(slots, Tensor):
        return False
    if rows.numel() != row_count or slots.numel() != row_count:
        return False
    if rows.dtype != torch.long or slots.dtype != torch.long:
        return False

    # These buffers are constructor-owned and immutable. Validate their values
    # when host-resident without introducing a CUDA synchronization.
    if rows.device.type == "cpu" and slots.device.type == "cpu":
        expected_rows = torch.arange(
            1, row_count + 1, dtype=torch.long, device=torch.device("cpu")
        )
        expected_slots = _degree_slots(lmax, device=torch.device("cpu"))
        return bool(
            torch.equal(rows, expected_rows) and torch.equal(slots, expected_slots)
        )
    return True


def validate_gie_contract(
    module: Any,
    n_nodes: int,
    edge_cache: Any,
    radial: Tensor,
    zonal: Tensor,
) -> bool:
    """Validate the shape/layout contract without inspecting dynamic edge values."""
    lmax = int(getattr(module, "lmax", -1))
    channels = int(getattr(module, "channels", -1))
    if lmax <= 0 or channels <= 0 or n_nodes <= 0:
        return False
    row_count = (lmax + 1) ** 2 - 1
    dst = getattr(edge_cache, "dst", None)
    inv_sqrt_deg = getattr(edge_cache, "inv_sqrt_deg", None)
    gate = getattr(edge_cache, "edge_src_gate", None)
    if not bool(getattr(edge_cache, "destinations_sorted", False)):
        return False
    if not isinstance(dst, Tensor) or not isinstance(inv_sqrt_deg, Tensor):
        return False
    if radial.dim() != 3 or zonal.dim() != 2 or dst.dim() != 1:
        return False
    edge_count = radial.shape[0]
    if edge_count == 0:
        return False
    if (
        zonal.shape != (edge_count, row_count)
        or radial.shape[1:] != (lmax, channels)
        or dst.shape[0] != edge_count
        or inv_sqrt_deg.shape != (n_nodes, 1, 1)
    ):
        return False
    if radial.dtype != torch.float32 or zonal.dtype != torch.float32:
        return False
    if inv_sqrt_deg.dtype != torch.float32:
        return False
    if dst.dtype not in (torch.int32, torch.int64):
        return False
    if not (radial.device == zonal.device == dst.device == inv_sqrt_deg.device):
        return False
    if radial.stride(-1) != 1 or radial.stride(-2) != channels:
        return False
    if (
        not zonal.is_contiguous()
        or not dst.is_contiguous()
        or not inv_sqrt_deg.is_contiguous()
    ):
        return False
    if gate is not None:
        if not isinstance(gate, Tensor):
            return False
        if gate.shape not in ((edge_count,), (edge_count, 1)):
            return False
        if gate.dtype != torch.float32 or gate.device != radial.device:
            return False
        if not gate.is_contiguous():
            return False
    return _standard_index_contract(module, lmax, row_count)


if SEZM_CUTE_GIE_AVAILABLE:
    _F32 = cutlass.Float32
    _I32 = cutlass.Int32
    _WARPS_PER_BLOCK = 4
    _LANES = 32

    def _build_forward(lmax: int, channels: int, has_gate: bool) -> Callable:
        row_count = (lmax + 1) ** 2 - 1

        @cute.kernel
        def kernel(m_radial, m_zonal, m_inv, m_dst_ptr, m_gate, m_out) -> None:
            node, _, _ = cute.arch.block_idx()
            lane, warp, _ = cute.arch.thread_idx()

            if warp == 0:
                for channel in cutlass.range(lane, channels, _LANES, unroll=1):
                    m_out[node, channel] = _F32(0.0)

            lo = m_dst_ptr[node].to(_I32)
            hi = m_dst_ptr[node + 1].to(_I32)
            for row in cutlass.range(warp, row_count, _WARPS_PER_BLOCK, unroll=1):
                radial_slot = _I32(0)
                for degree in cutlass.range_constexpr(lmax):
                    start = (degree + 1) * (degree + 1) - 1
                    stop = (degree + 2) * (degree + 2) - 1
                    if row >= start and row < stop:
                        radial_slot = degree
                for channel in cutlass.range(lane, channels, _LANES, unroll=1):
                    acc = _F32(0.0)
                    for edge in cutlass.range(lo, hi, 1, unroll=1):
                        scale = _F32(1.0)
                        if has_gate:
                            scale = m_gate[edge].to(_F32)
                        acc += (
                            m_zonal[edge, row].to(_F32)
                            * m_radial[edge, radial_slot * channels + channel].to(_F32)
                            * scale
                        )
                    m_out[node, (row + 1) * channels + channel] = acc * m_inv[
                        node, 0
                    ].to(_F32)

        @cute.jit
        def host(
            m_radial,
            m_zonal,
            m_inv,
            m_dst_ptr,
            m_gate,
            m_out,
            stream: CUstream,
        ) -> None:
            nodes, _ = m_out.shape
            kernel(m_radial, m_zonal, m_inv, m_dst_ptr, m_gate, m_out).launch(
                grid=[nodes, 1, 1],
                block=[_LANES, _WARPS_PER_BLOCK, 1],
                stream=stream,
            )

        return host

    def _build_backward(lmax: int, channels: int, has_gate: bool) -> Callable:
        row_count = (lmax + 1) ** 2 - 1
        full_width = (row_count + 1) * channels

        @cute.kernel
        def edge_kernel(
            m_grad_out,
            m_radial,
            m_zonal,
            m_inv,
            m_dst,
            m_gate,
            m_grad_radial,
            m_grad_zonal,
            m_grad_gate,
        ) -> None:
            block_edge, _, _ = cute.arch.block_idx()
            lane, warp, _ = cute.arch.thread_idx()
            edge = block_edge * _WARPS_PER_BLOCK + warp
            edge_count, _ = m_radial.shape
            load_edge = edge
            if edge >= edge_count:
                load_edge = 0
            node = m_dst[load_edge].to(_I32)
            norm = m_inv[node, 0].to(_F32)
            gate_value = _F32(1.0)
            if has_gate:
                gate_value = m_gate[load_edge].to(_F32)
            grad_gate_acc = _F32(0.0)

            if channels <= _LANES:
                active_channel = lane < channels
                for degree in cutlass.range_constexpr(lmax):
                    start = (degree + 1) * (degree + 1) - 1
                    stop = (degree + 2) * (degree + 2) - 1
                    grad_radial_acc = _F32(0.0)
                    for row in cutlass.range_constexpr(start, stop, 1):
                        grad_value = _F32(0.0)
                        radial_value = _F32(0.0)
                        zonal_value = _F32(0.0)
                        if active_channel:
                            grad_value = (
                                m_grad_out[node, (row + 1) * channels + lane].to(_F32)
                                * norm
                            )
                            radial_value = m_radial[
                                load_edge, degree * channels + lane
                            ].to(_F32)
                            zonal_value = m_zonal[load_edge, row].to(_F32)
                            grad_radial_acc += grad_value * zonal_value * gate_value
                            grad_gate_acc += grad_value * zonal_value * radial_value
                        grad_zonal_value = cute.arch.warp_reduction_sum(
                            grad_value * radial_value * gate_value
                        )
                        if lane == 0 and edge < edge_count:
                            m_grad_zonal[edge, row] = grad_zonal_value
                    if active_channel and edge < edge_count:
                        m_grad_radial[edge, degree * channels + lane] = grad_radial_acc
            else:
                for degree in cutlass.range_constexpr(lmax):
                    start = (degree + 1) * (degree + 1) - 1
                    stop = (degree + 2) * (degree + 2) - 1
                    for channel in cutlass.range(lane, channels, _LANES, unroll=1):
                        grad_radial_acc = _F32(0.0)
                        for row in cutlass.range_constexpr(start, stop, 1):
                            grad_value = (
                                m_grad_out[node, (row + 1) * channels + channel].to(
                                    _F32
                                )
                                * norm
                            )
                            radial_value = m_radial[
                                load_edge, degree * channels + channel
                            ].to(_F32)
                            zonal_value = m_zonal[load_edge, row].to(_F32)
                            grad_radial_acc += grad_value * zonal_value * gate_value
                            grad_gate_acc += grad_value * zonal_value * radial_value
                        if edge < edge_count:
                            m_grad_radial[edge, degree * channels + channel] = (
                                grad_radial_acc
                            )
                    for row in cutlass.range_constexpr(start, stop, 1):
                        grad_zonal_acc = _F32(0.0)
                        for channel in cutlass.range(lane, channels, _LANES, unroll=1):
                            grad_value = (
                                m_grad_out[node, (row + 1) * channels + channel].to(
                                    _F32
                                )
                                * norm
                            )
                            radial_value = m_radial[
                                load_edge, degree * channels + channel
                            ].to(_F32)
                            grad_zonal_acc += grad_value * radial_value * gate_value
                        grad_zonal_value = cute.arch.warp_reduction_sum(grad_zonal_acc)
                        if lane == 0 and edge < edge_count:
                            m_grad_zonal[edge, row] = grad_zonal_value

            if has_gate:
                grad_gate_value = cute.arch.warp_reduction_sum(grad_gate_acc)
                if lane == 0 and edge < edge_count:
                    m_grad_gate[edge] = grad_gate_value

        @cute.kernel
        def inv_kernel(m_grad_out, m_out, m_inv, m_grad_inv) -> None:
            block_node, _, _ = cute.arch.block_idx()
            lane, warp, _ = cute.arch.thread_idx()
            node = block_node * _WARPS_PER_BLOCK + warp
            node_count, _ = m_out.shape
            load_node = node
            if node >= node_count:
                load_node = 0
            acc = _F32(0.0)
            for idx in cutlass.range(channels + lane, full_width, _LANES, unroll=1):
                acc += m_grad_out[load_node, idx].to(_F32) * m_out[load_node, idx].to(
                    _F32
                )
            acc = cute.arch.warp_reduction_sum(acc)
            if lane == 0 and node < node_count:
                m_grad_inv[node, 0] = acc / m_inv[node, 0].to(_F32)

        @cute.jit
        def host(
            m_grad_out,
            m_radial,
            m_zonal,
            m_inv,
            m_dst,
            m_gate,
            m_out,
            m_grad_radial,
            m_grad_zonal,
            m_grad_inv,
            m_grad_gate,
            stream: CUstream,
        ) -> None:
            edge_count, _ = m_radial.shape
            node_count, _ = m_out.shape
            edge_kernel(
                m_grad_out,
                m_radial,
                m_zonal,
                m_inv,
                m_dst,
                m_gate,
                m_grad_radial,
                m_grad_zonal,
                m_grad_gate,
            ).launch(
                grid=[cute.ceil_div(edge_count, _WARPS_PER_BLOCK), 1, 1],
                block=[_LANES, _WARPS_PER_BLOCK, 1],
                stream=stream,
            )
            inv_kernel(m_grad_out, m_out, m_inv, m_grad_inv).launch(
                grid=[cute.ceil_div(node_count, _WARPS_PER_BLOCK), 1, 1],
                block=[_LANES, _WARPS_PER_BLOCK, 1],
                stream=stream,
            )

        return host

    _compile_lock = threading.Lock()
    _compiled: dict[tuple[Any, ...], Any] = {}

    def _as_cute(tensor: Tensor) -> Any:
        value = from_dlpack(tensor)
        if tensor.dim() <= 1:
            return value.mark_layout_dynamic()
        return value.mark_layout_dynamic(leading_dim=tensor.dim() - 1)

    def _device_key(tensor: Tensor) -> tuple[int, int, int]:
        index = tensor.device.index
        if index is None:
            index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(index)
        return index, major, minor

    def _get_compiled(
        key: tuple[Any, ...],
        builder: Callable[[], Callable],
        example_args: tuple[Any, ...],
    ) -> Any:
        compiled = _compiled.get(key)
        if compiled is not None:
            return compiled
        with _compile_lock:
            compiled = _compiled.get(key)
            if compiled is None:
                compiled = cute.compile(builder(), *example_args)
                _compiled[key] = compiled
        return compiled

    def _flat_radial(radial: Tensor) -> Tensor:
        return radial.view(radial.shape[0], radial.shape[1] * radial.shape[2])

    def _flat_node(tensor: Tensor) -> Tensor:
        return tensor.view(tensor.shape[0], -1)

    def _launch_forward(
        radial: Tensor,
        zonal: Tensor,
        inv_sqrt_deg: Tensor,
        dst_ptr: Tensor,
        gate: Tensor,
        lmax: int,
        has_gate: bool,
    ) -> Tensor:
        channels = radial.shape[2]
        out = torch.empty(
            inv_sqrt_deg.shape[0],
            (lmax + 1) ** 2,
            channels,
            device=radial.device,
            dtype=radial.dtype,
        )
        radial_flat = _flat_radial(radial.detach())
        inv_flat = _flat_node(inv_sqrt_deg.detach())
        gate_flat = gate.detach().view(-1)
        out_flat = _flat_node(out)
        args = tuple(
            _as_cute(value)
            for value in (
                radial_flat,
                zonal.detach(),
                inv_flat,
                dst_ptr.detach(),
                gate_flat,
                out_flat,
            )
        )
        device_identity = _device_key(radial)
        with torch.cuda.device(device_identity[0]):
            stream = cutlass_torch.current_stream()
            key = (
                "gie_fwd",
                *device_identity,
                lmax,
                channels,
                has_gate,
                tuple(radial_flat.stride()),
                dst_ptr.dtype,
            )
            compiled = _get_compiled(
                key,
                lambda: _build_forward(lmax, channels, has_gate),
                (*args, stream),
            )
            compiled(*args, stream)
        return out

    def _launch_backward(
        grad_out: Tensor,
        radial: Tensor,
        zonal: Tensor,
        inv_sqrt_deg: Tensor,
        dst: Tensor,
        gate: Tensor,
        out: Tensor,
        lmax: int,
        has_gate: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        channels = radial.shape[2]
        grad_out_flat = _flat_node(grad_out.detach().contiguous())
        radial_flat = _flat_radial(radial.detach())
        inv_flat = _flat_node(inv_sqrt_deg.detach())
        gate_flat = gate.detach().view(-1)
        out_flat = _flat_node(out.detach())
        grad_radial = torch.empty(
            radial.shape,
            device=radial.device,
            dtype=radial.dtype,
            memory_format=torch.contiguous_format,
        )
        grad_zonal = torch.empty_like(zonal, memory_format=torch.contiguous_format)
        grad_inv = torch.empty_like(inv_sqrt_deg, memory_format=torch.contiguous_format)
        grad_gate = torch.empty_like(gate, memory_format=torch.contiguous_format)
        grad_radial_flat = _flat_radial(grad_radial)
        grad_inv_flat = _flat_node(grad_inv)
        grad_gate_flat = grad_gate.view(-1)
        args = tuple(
            _as_cute(value)
            for value in (
                grad_out_flat,
                radial_flat,
                zonal.detach(),
                inv_flat,
                dst.detach(),
                gate_flat,
                out_flat,
                grad_radial_flat,
                grad_zonal,
                grad_inv_flat,
                grad_gate_flat,
            )
        )
        device_identity = _device_key(radial)
        with torch.cuda.device(device_identity[0]):
            stream = cutlass_torch.current_stream()
            key = _backward_compile_key(
                device_identity,
                lmax,
                channels,
                has_gate,
                tuple(radial_flat.stride()),
                dst.dtype,
            )
            compiled = _get_compiled(
                key,
                lambda: _build_backward(lmax, channels, has_gate),
                (*args, stream),
            )
            compiled(*args, stream)
        return grad_radial, grad_zonal, grad_inv, grad_gate

    @torch.library.custom_op(
        "sezm_cute::gie_fused", mutates_args=(), device_types="cuda"
    )
    def _gie_op(
        radial: Tensor,
        zonal: Tensor,
        inv_sqrt_deg: Tensor,
        dst: Tensor,
        dst_ptr: Tensor,
        gate: Tensor,
        lmax: int,
        has_gate: bool,
    ) -> Tensor:
        del dst
        return _launch_forward(
            radial,
            zonal,
            inv_sqrt_deg,
            dst_ptr,
            gate,
            int(lmax),
            bool(has_gate),
        )

    @_gie_op.register_fake
    def _(
        radial: Tensor,
        zonal: Tensor,
        inv_sqrt_deg: Tensor,
        dst: Tensor,
        dst_ptr: Tensor,
        gate: Tensor,
        lmax: int,
        has_gate: bool,
    ) -> Tensor:
        del zonal, dst, dst_ptr, gate, has_gate
        return radial.new_empty(
            (inv_sqrt_deg.shape[0], (int(lmax) + 1) ** 2, radial.shape[2])
        )

    @torch.library.custom_op(
        "sezm_cute::gie_fused_bwd", mutates_args=(), device_types="cuda"
    )
    def _gie_bwd_op(
        grad_out: Tensor,
        radial: Tensor,
        zonal: Tensor,
        inv_sqrt_deg: Tensor,
        dst: Tensor,
        gate: Tensor,
        out: Tensor,
        lmax: int,
        has_gate: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return _launch_backward(
            grad_out,
            radial,
            zonal,
            inv_sqrt_deg,
            dst,
            gate,
            out,
            int(lmax),
            bool(has_gate),
        )

    @_gie_bwd_op.register_fake
    def _(
        grad_out: Tensor,
        radial: Tensor,
        zonal: Tensor,
        inv_sqrt_deg: Tensor,
        dst: Tensor,
        gate: Tensor,
        out: Tensor,
        lmax: int,
        has_gate: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        del grad_out, dst, out, lmax, has_gate
        return (
            torch.empty_like(radial, memory_format=torch.contiguous_format),
            torch.empty_like(zonal, memory_format=torch.contiguous_format),
            torch.empty_like(inv_sqrt_deg, memory_format=torch.contiguous_format),
            torch.empty_like(gate, memory_format=torch.contiguous_format),
        )

    def _gie_setup_context(
        ctx: Any,
        inputs: tuple[Any, ...],
        output: Tensor,
    ) -> None:
        radial, zonal, inv_sqrt_deg, dst, _dst_ptr, gate, lmax, has_gate = inputs
        ctx.save_for_backward(radial, zonal, inv_sqrt_deg, dst, gate, output)
        ctx.lmax = int(lmax)
        ctx.has_gate = bool(has_gate)

    def _gie_backward(ctx: Any, grad_out: Tensor) -> tuple[Any, ...]:
        radial, zonal, inv_sqrt_deg, dst, gate, out = ctx.saved_tensors
        grad_radial, grad_zonal, grad_inv, grad_gate = _gie_bwd_op(
            grad_out,
            radial,
            zonal,
            inv_sqrt_deg,
            dst,
            gate,
            out,
            ctx.lmax,
            ctx.has_gate,
        )
        return grad_radial, grad_zonal, grad_inv, None, None, grad_gate, None, None

    _gie_op.register_autograd(_gie_backward, setup_context=_gie_setup_context)


def gie_fused_cuda(
    radial: Tensor,
    zonal: Tensor,
    inv_sqrt_deg: Tensor,
    dst: Tensor,
    gate: Tensor,
    *,
    n_nodes: int,
    lmax: int,
) -> Tensor:
    """Run the fused CUDA path after the caller has validated its contract."""
    if not SEZM_CUTE_GIE_AVAILABLE:
        raise RuntimeError("CuTe DSL is unavailable")
    boundaries = torch.arange(
        n_nodes + 1,
        device=dst.device,
        dtype=dst.dtype,
    )
    dst_ptr = torch.searchsorted(dst, boundaries)
    has_gate = gate.numel() != 0
    kernel_gate = gate if has_gate else radial.new_ones((1,))
    return _gie_op(
        radial,
        zonal,
        inv_sqrt_deg,
        dst,
        dst_ptr,
        kernel_gate,
        int(lmax),
        has_gate,
    )


def maybe_run_cute_gie(
    module: Any,
    *,
    n_nodes: int,
    edge_cache: Any,
    radial_feat: Tensor,
    zonal_coupling: Tensor,
) -> Tensor | None:
    """Run the opt-in path or return ``None`` for the eager fallback."""
    if (
        not is_cute_gie_enabled(radial_feat.device)
        or not SEZM_CUTE_GIE_AVAILABLE
        or bool(getattr(module, "training", True))
        or not radial_feat.is_cuda
        or not validate_gie_contract(
            module, n_nodes, edge_cache, radial_feat, zonal_coupling
        )
    ):
        return None
    gate = getattr(edge_cache, "edge_src_gate", None)
    if gate is None:
        gate = radial_feat.new_empty((0,))
    return gie_fused_cuda(
        radial_feat,
        zonal_coupling,
        edge_cache.inv_sqrt_deg,
        edge_cache.dst,
        gate,
        n_nodes=n_nodes,
        lmax=int(module.lmax),
    )


__all__ = [
    "SEZM_CUTE_GIE_AVAILABLE",
    "gie_fused_cuda",
    "is_cute_gie_enabled",
    "maybe_run_cute_gie",
    "validate_gie_contract",
]
