# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Packed-direct Phase-A/radial forward without an ``x_rot`` boundary."""

# ruff: noqa: ANN001, ANN201, ANN202, TC002, UP035

from __future__ import (
    annotations,
)

from dataclasses import (
    dataclass,
)
from typing import (
    Callable,
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

from ...compile_cache import (
    device_aware_lru_cache,
)
from ..wigner_layout import PACKED_VALUE_COUNT as PACKED_WIGNER_VALUES

FAKE_TENSOR_KW = {"assumed_align": 16, "use_32bit_stride": True}


@dataclass(frozen=True)
class NeoPhaseARadialForwardParams:
    x_wide: cute.Tensor
    src: cute.Tensor
    d_full: cute.Tensor
    radial_m0: cute.Tensor
    combined_weight: cute.Tensor
    hidden_weight: cute.Tensor
    channel_basis: cute.Tensor
    out: cute.Tensor
    rad_l0: cute.Tensor


@cute.jit
def _store_packed_phase_a_value(
    params: NeoPhaseARadialForwardParams,
    x_local: cute.Tensor,
    edge,
    src_node,
    channel,
    reduced: cutlass.Constexpr[int],
    panel_start: cutlass.Constexpr[int],
    full_start: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
):
    acc = cutlass.Float32(0.0)
    for local_col in cutlass.range_constexpr(width):
        d_val = params.d_full[edge, panel_start + local_col].to(cutlass.Float32)
        x_val = params.x_wide[
            src_node,
            (full_start + local_col) * 64 + channel,
        ].to(cutlass.Float32)
        acc += d_val * x_val
    x_local[reduced * 64 + channel] = acc


@cute.jit
def neo_phase_a_radial_forward_packed_direct_saved_jit(
    x_wide: cute.Tensor,
    src: cute.Tensor,
    d_full: cute.Tensor,
    radial_m0: cute.Tensor,
    combined_weight: cute.Tensor,
    hidden_weight: cute.Tensor,
    channel_basis: cute.Tensor,
    out: cute.Tensor,
    rad_l0: cute.Tensor,
    compact_out: cute.Tensor,
    stream: CUstream,
):
    params = NeoPhaseARadialForwardParams(
        x_wide=x_wide,
        src=src,
        d_full=d_full,
        radial_m0=radial_m0,
        combined_weight=combined_weight,
        hidden_weight=hidden_weight,
        channel_basis=channel_basis,
        out=out,
        rad_l0=rad_l0,
    )
    edges, _ = out.shape
    neo_phase_a_radial_forward_packed_direct_saved_kernel(
        params,
        compact_out,
    ).launch(
        grid=[edges, 1, 1],
        block=[64, 1, 1],
        stream=stream,
    )


@cute.kernel
def neo_phase_a_radial_forward_packed_direct_saved_kernel(
    params: NeoPhaseARadialForwardParams,
    compact_out: cute.Tensor,
):
    channel, _, _ = cute.arch.thread_idx()
    edge, _, _ = cute.arch.block_idx()

    smem = cutlass.utils.SmemAllocator()
    x_local = smem.allocate_tensor(cutlass.Float32, 10 * 64)
    compact = smem.allocate_tensor(cutlass.Float32, 25)
    src_node = params.src[edge]

    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 0, 0, 0, 1)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 1, 1, 1, 3)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 2, 10, 4, 5)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 3, 25, 9, 7)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 4, 4, 1, 3)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 5, 15, 4, 5)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 6, 32, 9, 7)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 7, 7, 1, 3)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 8, 20, 4, 5)
    _store_packed_phase_a_value(params, x_local, edge, src_node, channel, 9, 39, 9, 7)

    if channel < 25:
        acc = cutlass.Float32(0.0)
        for radial_idx in cutlass.range_constexpr(4 * 32):
            radial_value = params.radial_m0[edge, radial_idx].to(cutlass.Float32)
            weight = params.combined_weight[radial_idx, channel].to(cutlass.Float32)
            acc += radial_value * weight
        compact[channel] = acc
        compact_out[edge, channel] = acc

    acc_l0 = cutlass.Float32(0.0)
    for radial_channel in cutlass.range_constexpr(32):
        radial_value = params.radial_m0[edge, radial_channel].to(cutlass.Float32)
        weight = params.hidden_weight[radial_channel, channel].to(cutlass.Float32)
        acc_l0 += radial_value * weight
    params.rad_l0[edge, channel] = acc_l0

    cute.arch.sync_threads()

    for coeff in cutlass.range_constexpr(10):
        acc = cutlass.Float32(0.0)
        if coeff < 4:
            out_coeff = coeff
            for in_coeff in cutlass.range_constexpr(4):
                kval = compact[in_coeff * 4 + out_coeff]
                acc += kval * x_local[in_coeff * 64 + channel]
        elif coeff < 7:
            out_coeff = coeff - 4
            for in_coeff in cutlass.range_constexpr(3):
                kval = compact[16 + in_coeff * 3 + out_coeff]
                acc += kval * x_local[(4 + in_coeff) * 64 + channel]
        else:
            out_coeff = coeff - 7
            for in_coeff in cutlass.range_constexpr(3):
                kval = compact[16 + in_coeff * 3 + out_coeff]
                acc += kval * x_local[(7 + in_coeff) * 64 + channel]
        acc *= params.channel_basis[channel].to(cutlass.Float32)
        focus = channel // 32
        focus_channel = channel - focus * 32
        out_idx = focus * 10 * 32 + coeff * 32 + focus_channel
        params.out[edge, out_idx] = acc


def compile_neo_phase_a_radial_forward_packed_direct() -> Callable:
    edge_count = cute.sym_int64()
    node_count = cute.sym_int64()
    fake_x_wide = make_fake_compact_tensor(
        cutlass.Float32,
        (node_count, 16 * 64),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_src = make_fake_compact_tensor(
        cutlass.Int32,
        (edge_count,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_d = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, PACKED_WIGNER_VALUES),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_radial = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, 4 * 32),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_combined = make_fake_compact_tensor(
        cutlass.Float32,
        (4 * 32, 25),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_hidden = make_fake_compact_tensor(
        cutlass.Float32,
        (32, 64),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_basis = make_fake_compact_tensor(
        cutlass.Float32,
        (64,),
        stride_order=(0,),
        **FAKE_TENSOR_KW,
    )
    fake_out = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, 10 * 64),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_rad_l0 = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, 64),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    fake_stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    fake_compact = make_fake_compact_tensor(
        cutlass.Float32,
        (edge_count, 25),
        stride_order=(1, 0),
        **FAKE_TENSOR_KW,
    )
    return cute.compile(
        neo_phase_a_radial_forward_packed_direct_saved_jit,
        fake_x_wide,
        fake_src,
        fake_d,
        fake_radial,
        fake_combined,
        fake_hidden,
        fake_basis,
        fake_out,
        fake_rad_l0,
        fake_compact,
        fake_stream,
        options="--enable-tvm-ffi",
    )


@device_aware_lru_cache(maxsize=4)
def _compiled_neo_phase_a_radial_forward_packed_direct() -> Callable:
    return compile_neo_phase_a_radial_forward_packed_direct()


def _combined_radial_weight(radial_hidden_proj, radial_degree_mixer):
    import torch

    hidden_weight = radial_hidden_proj.weight
    mixer_weight = radial_degree_mixer.weight
    cache = getattr(radial_hidden_proj, "_deepmd_cute_neo_radial_combined", None)
    key = (
        hidden_weight.data_ptr(),
        hidden_weight._version,
        hidden_weight.dtype,
        hidden_weight.device,
        tuple(hidden_weight.shape),
        tuple(hidden_weight.stride()),
        hidden_weight.storage_offset(),
        mixer_weight.data_ptr(),
        mixer_weight._version,
        mixer_weight.dtype,
        mixer_weight.device,
        tuple(mixer_weight.shape),
        tuple(mixer_weight.stride()),
        mixer_weight.storage_offset(),
    )
    if cache is not None and cache[0] == key:
        return cache[1]

    blocks = []
    for degree in range(4):
        mixer_block = mixer_weight.detach()[degree * 64 : (degree + 1) * 64, :]
        blocks.append(torch.mm(hidden_weight.detach(), mixer_block))
    combined = torch.cat(blocks, dim=0).contiguous()
    radial_hidden_proj._deepmd_cute_neo_radial_combined = (key, combined)
    return combined


def run_neo_phase_a_radial_forward_packed_direct(
    *,
    radial_hidden_proj,
    radial_degree_mixer,
    x_wide,
    src,
    D_full,
    radial_feat_m0,
):
    """Validate and launch the packed Phase-A/radial forward kernel."""
    import torch

    if x_wide.shape[1:] != (16, 64):
        raise ValueError(f"expected x_wide shape (N,16,64), got {x_wide.shape}")
    edge_count = src.numel()
    if tuple(D_full.shape) != (edge_count, PACKED_WIGNER_VALUES):
        raise ValueError(
            "expected packed Wigner shape "
            f"{(edge_count, PACKED_WIGNER_VALUES)}, got {tuple(D_full.shape)}"
        )
    if radial_feat_m0.shape != (edge_count, 4, 32):
        raise ValueError(
            f"expected radial_feat_m0 shape {(edge_count, 4, 32)}, "
            f"got {tuple(radial_feat_m0.shape)}"
        )
    device = x_wide.device
    if device.type != "cuda":
        raise ValueError("packed Phase-A/radial forward requires CUDA tensors")
    if src.device != device or src.dtype not in (torch.int32, torch.int64):
        raise ValueError("src must be an int32 or int64 tensor on the input device")
    if src.data_ptr() % 16:
        raise ValueError("src must be 16-byte aligned")
    source_tensors = (
        ("x_wide", x_wide),
        ("D_full", D_full),
        ("radial_feat_m0", radial_feat_m0),
        ("radial_hidden_proj.weight", radial_hidden_proj.weight),
        ("radial_degree_mixer.weight", radial_degree_mixer.weight),
        ("radial_degree_mixer.channel_basis", radial_degree_mixer.channel_basis),
    )
    for name, tensor in source_tensors:
        if tensor.device != device or tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be FP32 on {device}")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    if radial_hidden_proj.bias is not None:
        raise NotImplementedError("collapsed radial mixer expects no hidden bias")
    if tuple(radial_hidden_proj.weight.shape) != (32, 64):
        raise NotImplementedError("collapsed radial mixer expects a (32,64) projection")
    if radial_degree_mixer.mode != "degree_channel" or radial_degree_mixer.rank != 1:
        raise NotImplementedError(
            "collapsed radial mixer expects degree_channel rank=1"
        )
    if tuple(radial_degree_mixer.weight.shape) != (4 * 64, 25):
        raise NotImplementedError("collapsed radial mixer expects lmax=3,mmax=1,C=64")
    if tuple(radial_degree_mixer.channel_basis.shape) != (1, 64):
        raise NotImplementedError(
            "collapsed radial mixer expects a rank-1, 64-channel basis"
        )

    combined_weight = _combined_radial_weight(
        radial_hidden_proj,
        radial_degree_mixer,
    )
    if not combined_weight.is_contiguous() or combined_weight.data_ptr() % 16:
        raise ValueError("combined radial weight must be contiguous and aligned")
    kernel = _compiled_neo_phase_a_radial_forward_packed_direct()
    out = torch.empty(
        edge_count,
        10 * 64,
        device=x_wide.device,
        dtype=x_wide.dtype,
    )
    rad_l0 = torch.empty(
        edge_count,
        64,
        device=radial_feat_m0.device,
        dtype=radial_feat_m0.dtype,
    )
    compact_out = torch.empty(
        (edge_count, 25),
        device=radial_feat_m0.device,
        dtype=torch.float32,
    )
    kernel(
        x_wide.contiguous().view(x_wide.shape[0], 16 * 64),
        src.to(torch.int32).contiguous(),
        D_full.contiguous(),
        radial_feat_m0.contiguous().view(edge_count, 4 * 32),
        combined_weight,
        radial_hidden_proj.weight.detach().contiguous(),
        radial_degree_mixer.channel_basis.detach().view(64).contiguous(),
        out,
        rad_l0,
        compact_out,
    )
    return (
        out.view(edge_count, 2, 10, 32),
        rad_l0.view(edge_count, 2, 32),
        compact_out,
    )
