# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Strict-FP32 degree-zero readout for the exact Neo output GridMLP."""

from __future__ import (
    annotations,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from . import (
    runtime_policy,
)
from .runtime_policy import (
    PORTABLE_TILED_BACKEND,
    select_output_grid_backend,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
    )


COEFF_DIM = 16
N_FRAMES = 3
PACKED_COEFF_DIM = 48
GRID_SIZE = 152
HIDDEN_CHANNELS = 192
PACKED_WIDTH = N_FRAMES * HIDDEN_CHANNELS
_READOUT_INPUT_FOLD_CACHE = "_neo_sm80_readout_input_fold_cache"
_READOUT_INPUT_FOLD_HOOK = "_neo_sm80_readout_input_fold_hook"
_READOUT_INPUT_FOLD_LEFT = "_neo_sm80_readout_input_fold_left"
_READOUT_INPUT_FOLD_RIGHT = "_neo_sm80_readout_input_fold_right"
_READOUT_INPUT_FOLD_SCALAR = "_neo_sm80_readout_input_fold_scalar"
_READOUT_INPUT_FOLD_BUFFERS = (
    _READOUT_INPUT_FOLD_LEFT,
    _READOUT_INPUT_FOLD_RIGHT,
    _READOUT_INPUT_FOLD_SCALAR,
)
_READOUT_INPUT_FOLD_PREPARE_ERROR = (
    "the SM80 readout input fold is stale or missing; call "
    "prepare_sm80_readout_input_fold(output_ffn) after loading, replacing, "
    "mutating, or moving model state and before torch.compile"
)


def _has_exact_product_shape(left: torch.Tensor) -> bool:
    return (
        left.ndim == 4
        and left.shape[0] > 0
        and tuple(left.shape[1:]) == (COEFF_DIM, 1, PACKED_WIDTH)
    )


def _has_exact_product_contract(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> bool:
    tensors = (left, right, to_grid, from_grid)
    if (
        not _has_exact_product_shape(left)
        or right.shape != left.shape
        or tuple(to_grid.shape) != (GRID_SIZE, PACKED_COEFF_DIM)
        or tuple(from_grid.shape) != (PACKED_COEFF_DIM, GRID_SIZE)
        or any(not tensor.is_cuda for tensor in tensors)
        or any(tensor.dtype != torch.float32 for tensor in tensors)
        or any(tensor.device != left.device for tensor in tensors)
        or any(not tensor.is_contiguous() for tensor in tensors)
        or to_grid.requires_grad
        or from_grid.requires_grad
    ):
        return False
    compute_capability = tuple(torch.cuda.get_device_capability(left.device))
    return (
        select_output_grid_backend(compute_capability, HIDDEN_CHANNELS)
        == PORTABLE_TILED_BACKEND
    )


def _validate_product_contract(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> None:
    if not _has_exact_product_contract(left, right, to_grid, from_grid):
        raise ValueError(
            "the readout l=0 kernel requires contiguous CUDA FP32 tensors "
            "with Neo's left/right=(N,16,1,576), to_grid=(152,48), and "
            "from_grid=(48,152) contract"
        )


def build_readout_l0_gram(
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> torch.Tensor:
    """Collapse the frozen row-zero projector into a dense FP32 Gram matrix."""
    if (
        tuple(to_grid.shape) != (GRID_SIZE, PACKED_COEFF_DIM)
        or tuple(from_grid.shape) != (PACKED_COEFF_DIM, GRID_SIZE)
        or to_grid.dtype != torch.float32
        or from_grid.dtype != torch.float32
        or to_grid.device != from_grid.device
        or not to_grid.is_contiguous()
        or not from_grid.is_contiguous()
        or to_grid.requires_grad
        or from_grid.requires_grad
    ):
        raise ValueError(
            "readout l=0 Gram construction requires frozen contiguous FP32 "
            "to_grid=(152,48) and from_grid=(48,152) tensors"
        )
    with torch.no_grad():
        return torch.matmul(
            to_grid.T,
            from_grid[0, :, None] * to_grid,
        ).contiguous()


def _has_exact_neo_readout_structure(output_ffn: Any) -> bool:
    from deepmd.pt.model.descriptor.sezm_nn.ffn import (
        EquivariantFFN,
    )
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        GridMLP,
        SO3GridNet,
    )

    if type(output_ffn) is not EquivariantFFN:
        return False
    grid_net = output_ffn.act
    if type(grid_net) is not SO3GridNet or type(grid_net.grid_op) is not GridMLP:
        return False
    grid_op = grid_net.grid_op
    projector = grid_net.projector
    return (
        output_ffn.lmax == 3
        and output_ffn.channels == 32
        and output_ffn.hidden_channels == 96
        and output_ffn.kmax == 1
        and output_ffn.grid_n_frames == N_FRAMES
        and output_ffn.use_grid_net
        and output_ffn.use_grid_mlp
        and not output_ffn.use_grid_branch
        and output_ffn.ffn_so3_grid
        and not output_ffn.s2_activation
        and not output_ffn.mlp_bias
        and grid_net.lmax == 3
        and grid_net.channels == 96
        and grid_net.n_focus == 1
        and grid_net.n_frames == N_FRAMES
        and grid_net.mode == "self"
        and grid_net.op_type == "mlp"
        and grid_net.layout == "ndfc"
        and grid_net.frame_zero_index == 0
        and grid_net.frames == [0, -1, 1]
        and grid_net.frame_expand is None
        and grid_net.frame_contract is None
        and grid_net.residual_scale is None
        and grid_op.mode == "self"
        and grid_op.channels == 96
        and grid_op.hidden_channels == HIDDEN_CHANNELS
        and grid_op.n_frames == N_FRAMES
        and tuple(output_ffn.so3_linear_1.weight.shape) == (4, 32, PACKED_WIDTH)
        and tuple(grid_op.left_proj.weight.shape) == (HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        and tuple(grid_op.right_proj.weight.shape) == (HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        and tuple(grid_op.out_proj.weight.shape) == (HIDDEN_CHANNELS, 96)
        and tuple(grid_net.scalar_gate.weight.shape) == (HIDDEN_CHANNELS, 96)
        and tuple(output_ffn.so3_linear_2.weight.shape) == (4, 288, 32)
        and output_ffn.so3_linear_1.bias is None
        and grid_op.left_proj.bias is None
        and grid_op.right_proj.bias is None
        and grid_op.out_proj.bias is None
        and grid_net.scalar_gate.bias is None
        and output_ffn.so3_linear_2.bias is None
        and tuple(projector.to_grid_mat.shape) == (GRID_SIZE, PACKED_COEFF_DIM)
        and tuple(projector.from_grid_mat.shape) == (PACKED_COEFF_DIM, GRID_SIZE)
    )


def _state_uses_strict_fp32(output_ffn: Any, device: torch.device) -> bool:
    for tensor in (*output_ffn.parameters(), *output_ffn.buffers()):
        if tensor.is_floating_point() and (
            tensor.dtype != torch.float32
            or tensor.device != device
            or not tensor.is_contiguous()
        ):
            return False
    return True


def _inference_mode_is_frozen(output_ffn: Any) -> bool:
    return not output_ffn.training and not any(
        parameter.requires_grad for parameter in output_ffn.parameters()
    )


@torch.compiler.assume_constant_result
def _uses_strict_fp32_matmul() -> bool:
    """Preserve the tested private helper while sharing the runtime policy."""
    return runtime_policy.uses_strict_fp32_matmul()


def _has_exact_neo_readout_contract(
    output_ffn: Any,
    ffn_in: torch.Tensor,
) -> bool:
    if (
        not _has_exact_neo_readout_structure(output_ffn)
        or not _inference_mode_is_frozen(output_ffn)
        or ffn_in.ndim != 4
        or ffn_in.shape[0] <= 0
        or tuple(ffn_in.shape[1:]) != (COEFF_DIM, 1, 32)
        or not ffn_in.is_cuda
        or ffn_in.dtype != torch.float32
        or not ffn_in.is_contiguous()
        or not _state_uses_strict_fp32(output_ffn, ffn_in.device)
        or torch.is_autocast_enabled("cuda")
        or not _uses_strict_fp32_matmul()
    ):
        return False
    compute_capability = tuple(torch.cuda.get_device_capability(ffn_in.device))
    return (
        select_output_grid_backend(compute_capability, HIDDEN_CHANNELS)
        == PORTABLE_TILED_BACKEND
    )


def _can_use_sm80_readout_input_fold(
    output_ffn: Any,
    ffn_in: torch.Tensor,
) -> bool:
    """Fail closed unless the exact frozen strict-FP32 SM80 path is active."""
    if not _has_exact_neo_readout_contract(output_ffn, ffn_in):
        return False
    compute_capability = tuple(torch.cuda.get_device_capability(ffn_in.device))
    return runtime_policy.is_readout_input_fold_enabled(compute_capability)


def _readout_input_fold_sources(output_ffn: Any) -> tuple[torch.Tensor, ...]:
    grid_net = output_ffn.act
    grid_op = grid_net.grid_op
    return (
        output_ffn.so3_linear_1.weight,
        grid_op.left_proj.weight,
        grid_op.right_proj.weight,
        grid_net.scalar_gate.weight,
    )


def _readout_input_fold_cache_key(
    sources: tuple[torch.Tensor, ...],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            tensor.data_ptr(),
            tensor._version,
            tensor.dtype,
            tensor.device,
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tensor.storage_offset(),
        )
        for tensor in sources
    )


def _readout_input_fold_cache_matches(
    output_ffn: Any,
    sources: tuple[torch.Tensor, ...],
    cache_key: tuple[tuple[Any, ...], ...],
) -> bool:
    cache = getattr(output_ffn, _READOUT_INPUT_FOLD_CACHE, None)
    return (
        cache is not None
        and len(cache) == 2
        and len(cache[0]) == len(sources)
        and all(
            cached is current for cached, current in zip(cache[0], sources, strict=True)
        )
        and cache[1] == cache_key
        and all(
            isinstance(getattr(output_ffn, name, None), torch.Tensor)
            for name in _READOUT_INPUT_FOLD_BUFFERS
        )
    )


def _invalidate_sm80_readout_input_fold(output_ffn: Any) -> None:
    setattr(output_ffn, _READOUT_INPUT_FOLD_CACHE, None)
    for name in _READOUT_INPUT_FOLD_BUFFERS:
        if name in output_ffn._buffers:
            setattr(output_ffn, name, None)


def invalidate_neo_readout_input_fold(output_ffn: Any) -> None:
    """Invalidate frozen readout weights after parameter topology changes."""
    _invalidate_sm80_readout_input_fold(output_ffn)


def _invalidate_sm80_readout_input_fold_after_load(
    output_ffn: Any,
    incompatible_keys: Any,
) -> None:
    del incompatible_keys
    _invalidate_sm80_readout_input_fold(output_ffn)


def _ensure_sm80_readout_input_fold_load_hook(output_ffn: Any) -> None:
    if getattr(output_ffn, _READOUT_INPUT_FOLD_HOOK, False):
        return
    output_ffn.register_load_state_dict_post_hook(
        _invalidate_sm80_readout_input_fold_after_load
    )
    setattr(output_ffn, _READOUT_INPUT_FOLD_HOOK, True)


def _set_nonpersistent_buffer(
    module: Any,
    name: str,
    tensor: torch.Tensor,
) -> None:
    if name in module._buffers:
        setattr(module, name, tensor)
    else:
        module.register_buffer(name, tensor, persistent=False)


def _synchronize_sm80_readout_input_fold_build(
    folded_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    """Make a newly built CUDA cache safe for every later consumer stream."""
    device = folded_weights[0].device
    if device.type != "cuda":
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "prepare the SM80 readout input fold before CUDA graph capture"
        )
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(device))
    ready.synchronize()


def _build_sm80_readout_input_fold(
    output_ffn: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compose the frozen maps without crossing either readout nonlinearity."""
    input_weight, left_weight, right_weight, scalar_gate_weight = (
        _readout_input_fold_sources(output_ffn)
    )
    grid_net = output_ffn.act
    with torch.no_grad():
        split_weight = input_weight.reshape(
            output_ffn.lmax + 1,
            output_ffn.channels,
            2,
            N_FRAMES,
            96,
        )
        left_input = split_weight[:, :, 0]
        right_input = split_weight[:, :, 1]
        per_frame_input = torch.cat((left_input, right_input), dim=-1)
        left_fold = torch.matmul(per_frame_input, left_weight)
        right_fold = torch.matmul(per_frame_input, right_weight)
        projected_left_weight = left_fold.reshape(
            output_ffn.lmax + 1,
            output_ffn.channels,
            -1,
        ).contiguous()
        projected_right_weight = right_fold.reshape(
            output_ffn.lmax + 1,
            output_ffn.channels,
            -1,
        ).contiguous()

        frame_zero = grid_net.frame_zero_index
        scalar_pair_weight = torch.cat(
            (
                left_input[0, :, frame_zero],
                right_input[0, :, frame_zero],
            ),
            dim=-1,
        )
        scalar_gate_fold = torch.matmul(scalar_pair_weight, scalar_gate_weight)
        scalar_aux_weight = torch.cat(
            (scalar_pair_weight, scalar_gate_fold),
            dim=-1,
        ).contiguous()
    return (
        projected_left_weight.detach().clone(),
        projected_right_weight.detach().clone(),
        scalar_aux_weight.detach().clone(),
    )


@torch.compiler.disable
def prepare_sm80_readout_input_fold(
    output_ffn: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare immutable folded weights before compiling the frozen readout."""
    if torch.compiler.is_compiling():
        raise RuntimeError(_READOUT_INPUT_FOLD_PREPARE_ERROR)
    if not _has_exact_neo_readout_structure(
        output_ffn
    ) or not _inference_mode_is_frozen(output_ffn):
        raise ValueError("readout input folding requires the exact frozen Neo FFN")

    sources = _readout_input_fold_sources(output_ffn)
    device = sources[0].device
    if any(
        tensor.dtype != torch.float32
        or tensor.device != device
        or not tensor.is_contiguous()
        for tensor in sources
    ):
        raise ValueError(
            "readout input folding requires contiguous FP32 source weights "
            "on one device"
        )
    cache_key = _readout_input_fold_cache_key(sources)
    if _readout_input_fold_cache_matches(output_ffn, sources, cache_key):
        return (
            getattr(output_ffn, _READOUT_INPUT_FOLD_LEFT),
            getattr(output_ffn, _READOUT_INPUT_FOLD_RIGHT),
            getattr(output_ffn, _READOUT_INPUT_FOLD_SCALAR),
        )

    _invalidate_sm80_readout_input_fold(output_ffn)
    folded_weights = _build_sm80_readout_input_fold(output_ffn)
    _synchronize_sm80_readout_input_fold_build(folded_weights)
    _ensure_sm80_readout_input_fold_load_hook(output_ffn)
    for name, tensor in zip(
        _READOUT_INPUT_FOLD_BUFFERS,
        folded_weights,
        strict=True,
    ):
        _set_nonpersistent_buffer(output_ffn, name, tensor)
    setattr(
        output_ffn,
        _READOUT_INPUT_FOLD_CACHE,
        (sources, cache_key),
    )
    return folded_weights


@torch.compiler.disable
def maybe_prepare_sm80_readout_input_fold(
    output_ffn: Any,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Prepare only when a supported frozen Neo readout contract matches."""
    if not _has_exact_neo_readout_structure(
        output_ffn
    ) or not _inference_mode_is_frozen(output_ffn):
        return False
    sources = _readout_input_fold_sources(output_ffn)
    device = sources[0].device
    if device.type != "cuda":
        return False
    if compute_capability is None:
        compute_capability = tuple(torch.cuda.get_device_capability(device))
    if not runtime_policy.is_readout_input_fold_enabled(compute_capability):
        return False
    if any(
        tensor.dtype != torch.float32
        or tensor.device != device
        or not tensor.is_contiguous()
        for tensor in sources
    ):
        return False
    prepare_sm80_readout_input_fold(output_ffn)
    return True


def _get_prepared_sm80_readout_input_fold(
    output_ffn: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = getattr(output_ffn, _READOUT_INPUT_FOLD_CACHE, None)
    folded_weights = (
        getattr(output_ffn, _READOUT_INPUT_FOLD_LEFT, None),
        getattr(output_ffn, _READOUT_INPUT_FOLD_RIGHT, None),
        getattr(output_ffn, _READOUT_INPUT_FOLD_SCALAR, None),
    )
    if (
        cache is None
        or len(cache) != 2
        or len(cache[0]) != 4
        or any(not isinstance(weight, torch.Tensor) for weight in folded_weights)
    ):
        raise RuntimeError(_READOUT_INPUT_FOLD_PREPARE_ERROR)

    sources = _readout_input_fold_sources(output_ffn)
    cached_sources, cache_key = cache
    if (
        any(
            cached is not current
            for cached, current in zip(cached_sources, sources, strict=True)
        )
        or any(
            source.dtype != cached[2]
            or source.device != cached[3]
            or tuple(source.shape) != cached[4]
            for source, cached in zip(sources, cache_key, strict=True)
        )
        or any(
            weight.dtype != torch.float32
            or weight.device != sources[0].device
            or not weight.is_contiguous()
            for weight in folded_weights
        )
    ):
        raise RuntimeError(_READOUT_INPUT_FOLD_PREPARE_ERROR)

    # Do not emit source ``_version`` counters into the graph. AOTAutograd
    # represents them as unbacked symbolic integers and cannot lower the
    # resulting assertion. Eager preparation validates versions before trace;
    # SeZM's load-state hook invalidates both local and shared compiled graphs.
    return folded_weights


def _get_sm80_readout_input_fold(
    output_ffn: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return prepared degree weights for operands and scalar auxiliaries."""
    if torch.compiler.is_compiling():
        return _get_prepared_sm80_readout_input_fold(output_ffn)
    return prepare_sm80_readout_input_fold(output_ffn)


def _maybe_prepare_sm80_readout_input_fold(
    output_ffn: Any,
    ffn_in: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Project directly from the 32-channel input to both product operands."""
    if not _can_use_sm80_readout_input_fold(output_ffn, ffn_in):
        return None

    left_weight, right_weight, scalar_weight = _get_sm80_readout_input_fold(output_ffn)
    expanded_left_weight = left_weight.index_select(
        0,
        output_ffn.so3_linear_1.expand_index,
    )
    expanded_right_weight = right_weight.index_select(
        0,
        output_ffn.so3_linear_1.expand_index,
    )
    left = torch.einsum("ndfi,dio->ndfo", ffn_in, expanded_left_weight).contiguous()
    right = torch.einsum("ndfi,dio->ndfo", ffn_in, expanded_right_weight).contiguous()
    scalar_aux = torch.einsum("nfi,io->nfo", ffn_in[:, 0], scalar_weight)
    scalar_pair, scalar_gate_logits = torch.split(scalar_aux, (192, 96), dim=-1)
    return left, right, scalar_pair, scalar_gate_logits


def _run_neo_readout_l0(
    output_ffn: Any,
    ffn_in: torch.Tensor,
    grid_product: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ],
) -> torch.Tensor:
    """Complete the exact Neo readout around one row-zero grid product."""
    if not _has_exact_neo_readout_structure(output_ffn):
        raise ValueError("readout l=0 completion requires the exact Neo output FFN")
    from deepmd.pt.model.descriptor.sezm_nn.grid_net import (
        _project_frames,
    )

    grid_net = output_ffn.act
    grid_op = grid_net.grid_op

    prepared = _maybe_prepare_sm80_readout_input_fold(output_ffn, ffn_in)
    if prepared is None:
        projected = output_ffn.so3_linear_1(ffn_in)
        left, right, scalar_pair = grid_net._prepare_self_pair(projected)
        shape = (*left.shape[:-1], N_FRAMES, -1)
        fused = torch.cat(
            [left.reshape(shape), right.reshape(shape)],
            dim=-1,
        ).reshape(*left.shape[:-1], -1)
        left = _project_frames(fused, grid_op.left_proj, N_FRAMES)
        right = _project_frames(fused, grid_op.right_proj, N_FRAMES)
        scalar_gate_logits = None
    else:
        left, right, scalar_pair, scalar_gate_logits = prepared

    q0 = grid_product(
        left,
        right,
        grid_net.projector.to_grid_mat,
        grid_net.projector.from_grid_mat,
    )
    if tuple(q0.shape) != (ffn_in.shape[0], HIDDEN_CHANNELS):
        raise ValueError("readout l=0 grid product must return shape (N,192)")

    q0 = torch.matmul(q0, grid_op.out_proj.weight)
    scalar_out = grid_net.scalar_act(scalar_pair)[:, 0, :]
    if scalar_gate_logits is None:
        scalar_gate_logits = grid_net.scalar_gate(scalar_pair)
    scalar_gate = torch.sigmoid(scalar_gate_logits)[:, 0, :]
    scalar_coeff = q0 * scalar_gate + scalar_out
    output_weight = output_ffn.so3_linear_2.weight[0, :96, :]
    return ffn_in[:, 0, 0, :] + torch.matmul(scalar_coeff, output_weight)


def maybe_run_neo_readout_l0(
    output_ffn: Any,
    ffn_in: torch.Tensor,
) -> torch.Tensor | None:
    """Return the optimized final `[N,32]` readout or ``None`` for fallback."""
    if not runtime_policy.is_cute_infer_enabled():
        return None
    if not _inference_mode_is_frozen(output_ffn):
        return None
    if not _has_exact_neo_readout_contract(output_ffn, ffn_in):
        return None
    return _run_neo_readout_l0(output_ffn, ffn_in, readout_l0_product_cute)


def run_neo_output_readout(
    output_ffn: Any,
    ffn_in: torch.Tensor,
    *,
    parameters_frozen: bool = True,
) -> torch.Tensor:
    """Return the residual-inclusive `[N,32]` output with generic fallback."""
    if parameters_frozen:
        candidate = maybe_run_neo_readout_l0(output_ffn, ffn_in)
        if candidate is not None:
            return candidate
    return (ffn_in + output_ffn(ffn_in))[:, 0:1, :, :].reshape(
        ffn_in.shape[0], output_ffn.channels
    )


def _readout_l0_impl(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> torch.Tensor:
    _validate_product_contract(left, right, to_grid, from_grid)
    from .output_grid_kernels.cute_readout_l0 import (
        run_readout_l0,
    )

    nodes = left.shape[0]
    q0 = run_readout_l0(
        left.detach().view(nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS),
        right.detach().view(nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS),
        to_grid.detach(),
        from_grid.detach(),
    )
    return q0


def _readout_l0_bwd_impl(
    dq0: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _validate_product_contract(left, right, to_grid, from_grid)
    if (
        tuple(dq0.shape) != (left.shape[0], HIDDEN_CHANNELS)
        or dq0.dtype != left.dtype
        or dq0.device != left.device
        or not dq0.is_contiguous()
    ):
        raise ValueError("dq0 must be contiguous and have shape (N,192)")
    from .output_grid_kernels.cute_readout_l0 import (
        run_readout_l0_backward,
    )

    nodes = left.shape[0]
    grad_left, grad_right = run_readout_l0_backward(
        dq0.detach(),
        left.detach().view(nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS),
        right.detach().view(nodes, PACKED_COEFF_DIM, HIDDEN_CHANNELS),
        to_grid.detach(),
        from_grid.detach(),
    )
    return grad_left.view_as(left), grad_right.view_as(right)


_readout_l0_op = torch.library.custom_op(
    "sezm_cute::readout_l0",
    mutates_args=(),
)(_readout_l0_impl)
_readout_l0_bwd_op = torch.library.custom_op(
    "sezm_cute::readout_l0_bwd",
    mutates_args=(),
)(_readout_l0_bwd_impl)


@_readout_l0_op.register_fake
def _readout_l0_fake(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> torch.Tensor:
    del right, to_grid, from_grid
    return torch.empty(
        (left.shape[0], HIDDEN_CHANNELS),
        dtype=left.dtype,
        device=left.device,
    )


@_readout_l0_bwd_op.register_fake
def _readout_l0_bwd_fake(
    dq0: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del dq0, to_grid, from_grid
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
    left, right, to_grid, from_grid = inputs
    ctx.save_for_backward(left, right, to_grid, from_grid)


def _backward(ctx: Any, dq0: torch.Tensor) -> tuple:
    left, right, to_grid, from_grid = ctx.saved_tensors
    grad_left, grad_right = _readout_l0_bwd_op(
        dq0.contiguous(),
        left,
        right,
        to_grid,
        from_grid,
    )
    return grad_left, grad_right, None, None


_readout_l0_op.register_autograd(
    _backward,
    setup_context=_setup_context,
)


def readout_l0_product_cute(
    left: torch.Tensor,
    right: torch.Tensor,
    to_grid: torch.Tensor,
    from_grid: torch.Tensor,
) -> torch.Tensor:
    """Run the exact-shape C=192 degree-zero grid contraction."""
    return _readout_l0_op(left, right, to_grid, from_grid)


__all__ = [
    "build_readout_l0_gram",
    "invalidate_neo_readout_input_fold",
    "maybe_prepare_sm80_readout_input_fold",
    "maybe_run_neo_readout_l0",
    "prepare_sm80_readout_input_fold",
    "readout_l0_product_cute",
    "run_neo_output_readout",
]
