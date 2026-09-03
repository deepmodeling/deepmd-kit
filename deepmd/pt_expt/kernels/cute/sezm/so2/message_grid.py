# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Packed-layout Neo message-grid forward and first input adjoint."""

from __future__ import (
    annotations,
)

from typing import (
    Any,
    Protocol,
)

import torch
from torch import (
    Tensor,
)

COEFF_DIM = 16
N_FOCUS = 2
N_FRAMES = 3


class _Sm90MessageGridState(Protocol):
    """State subset consumed by the packed SM90 forward path."""

    schedule: Tensor


CHANNELS = 32
HIDDEN_CHANNELS = N_FOCUS * CHANNELS


def _validate_module_contract(net: Any) -> None:
    expected = {
        "layout": "flat",
        "mode": "cross",
        "op_type": "glu",
        "n_focus": N_FOCUS,
        "n_frames": N_FRAMES,
        "channels": CHANNELS,
        "dtype": torch.float32,
    }
    mismatches = {
        name: (getattr(net, name, None), value)
        for name, value in expected.items()
        if getattr(net, name, None) != value
    }
    frame_expand = getattr(net, "frame_expand", None)
    frame_contract = getattr(net, "frame_contract", None)
    if frame_expand is None or frame_contract is None:
        mismatches["frame_modules"] = (
            (frame_expand is not None, frame_contract is not None),
            (True, True),
        )
    else:
        expected_frame = ("packed", N_FRAMES, CHANNELS)
        for name, module in (
            ("frame_expand", frame_expand),
            ("frame_contract", frame_contract),
        ):
            actual_frame = (
                getattr(module, "coefficient_layout", None),
                getattr(module, "n_frames", None),
                getattr(module, "channels", None),
            )
            if actual_frame != expected_frame:
                mismatches[name] = (actual_frame, expected_frame)
    if mismatches:
        details = ", ".join(
            f"{name}={actual!r} (expected {wanted!r})"
            for name, (actual, wanted) in mismatches.items()
        )
        raise ValueError(f"packed message-grid module contract mismatch: {details}")


def is_supported_message_grid(net: Any) -> bool:
    """Return whether forward and manual backward share the same contract."""
    try:
        _validate_module_contract(net)
    except ValueError:
        return False
    return True


def _as_flat_ndfc(
    net: Any,
    name: str,
    value: Tensor,
    *,
    like: Tensor | None = None,
) -> Tensor:
    """Adapt GridNet's flat layout without materializing a contiguous copy."""
    expected_shape = (COEFF_DIM, HIDDEN_CHANNELS)
    if value.ndim != 3 or tuple(value.shape[1:]) != expected_shape:
        raise ValueError(
            f"packed message-grid {name} must have shape (N, {COEFF_DIM}, "
            f"{HIDDEN_CHANNELS}), got {tuple(value.shape)}"
        )
    if value.dtype != torch.float32:
        raise ValueError(f"packed message-grid {name} must be FP32, got {value.dtype}")
    if like is not None and (value.shape != like.shape or value.device != like.device):
        raise ValueError(
            f"packed message-grid {name} must match query shape/device; got "
            f"shape={tuple(value.shape)}, device={value.device}"
        )
    # SO3Linear's einsum returns the valid flat GridNet layout with stride
    # (64, N*64, 1), while Phase C returns compact (1024, 64, 1). Both split
    # their unit-stride final axis into (F=2, C=32) without a copy.
    if value.stride(-1) != 1:
        raise ValueError(
            f"packed message-grid {name} requires a unit-stride folded F*C "
            f"axis, got stride={tuple(value.stride())}"
        )
    value_ndfc, shape_info = net._to_ndfc(value)
    expected_ndfc = (value.shape[0], COEFF_DIM, N_FOCUS, CHANNELS)
    if (
        tuple(shape_info) != tuple(value.shape)
        or tuple(value_ndfc.shape) != expected_ndfc
    ):
        raise ValueError(
            f"packed message-grid {name} did not adapt to {expected_ndfc}; got "
            f"shape={tuple(value_ndfc.shape)}, stride={tuple(value_ndfc.stride())}"
        )
    return value_ndfc


def _validate_contract(
    net: Any,
    query: Tensor,
    context: Tensor,
) -> tuple[Tensor, Tensor]:
    _validate_module_contract(net)
    query_ndfc = _as_flat_ndfc(net, "query", query)
    context_ndfc = _as_flat_ndfc(net, "context", context, like=query)
    return query_ndfc, context_ndfc


def _expanded_frame_weight(module: Any) -> Tensor:
    return module.weight.index_select(0, module.degree_index)


def _frame_expand_packed(module: Any, coeff: Tensor) -> Tensor:
    weight = _expanded_frame_weight(module).view(
        COEFF_DIM,
        CHANNELS,
        N_FRAMES,
        CHANNELS,
    )
    # Output order D,K,F,C is the native CuTe grid-product contract. The
    # trailing F,C panel remains unit-stride and therefore coalesced.
    # PyTorch's degree-batched einsum naturally returns a degree-major stride.
    # Normalize once at this producer because the CuTe consumer's packed
    # contract is compact (N,D,K,F,C), not because the flat input was strided.
    return torch.einsum("ndfi,dikc->ndkfc", coeff, weight).contiguous()


def _frame_expand_packed_backward(
    module: Any,
    grad_packed: Tensor,
) -> Tensor:
    weight = _expanded_frame_weight(module).view(
        COEFF_DIM,
        CHANNELS,
        N_FRAMES,
        CHANNELS,
    )
    return torch.einsum("ndkfc,dikc->ndfi", grad_packed, weight)


def _frame_contract_packed(module: Any, coeff_packed: Tensor) -> Tensor:
    weight = _expanded_frame_weight(module).view(
        COEFF_DIM,
        N_FRAMES,
        CHANNELS,
        CHANNELS,
    )
    return torch.einsum("ndkfc,dkco->ndfo", coeff_packed, weight)


def _frame_contract_packed_backward(module: Any, grad_out: Tensor) -> Tensor:
    weight = _expanded_frame_weight(module).view(
        COEFF_DIM,
        N_FRAMES,
        CHANNELS,
        CHANNELS,
    )
    return torch.einsum("ndfo,dkco->ndkfc", grad_out, weight)


def _focus_linear_backward_input(linear: Any, grad_out: Tensor) -> Tensor:
    weight = linear.weight.view(linear.in_channels, linear.n_focus, linear.out_channels)
    return torch.einsum("bfo,ifo->bfi", grad_out, weight)


def _swiglu_backward_input(x: Tensor, grad_out: Tensor) -> Tensor:
    gate, value = torch.chunk(x, chunks=2, dim=-1)
    sigmoid = torch.sigmoid(gate)
    grad_gate = grad_out * value * (sigmoid + gate * sigmoid * (1.0 - sigmoid))
    grad_value = grad_out * gate * sigmoid
    return torch.cat([grad_gate, grad_value], dim=-1)


def run_packed_message_grid_forward(
    net: Any,
    query_flat: Tensor,
    context_flat: Tensor,
    *,
    return_product: bool = False,
    sm90_state: _Sm90MessageGridState | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Run only the message-grid module and return its canonical flat output."""
    query, context = _validate_contract(net, query_flat, context_flat)
    from .kernels.message_grid_product import (
        run_message_grid_product,
    )

    nodes = query_flat.shape[0]
    scalar_pair = torch.cat([query[:, 0], context[:, 0]], dim=-1).to(net.dtype)

    left_packed = _frame_expand_packed(net.frame_expand, query)
    right_packed = _frame_expand_packed(net.frame_expand, context)
    left = left_packed.view(nodes, COEFF_DIM * N_FRAMES, HIDDEN_CHANNELS)
    right = right_packed.view(nodes, COEFF_DIM * N_FRAMES, HIDDEN_CHANNELS)
    if sm90_state is None:
        product_flat = run_message_grid_product(
            left,
            right,
            net.projector.to_grid_mat,
            net.projector.from_grid_mat,
        )
    else:
        from .sm90.message_grid_gaunt import (
            run_sm90_gaunt_forward,
        )

        product_flat = run_sm90_gaunt_forward(
            left,
            right,
            sm90_state.schedule,
        )
    product = product_flat.view(
        nodes,
        COEFF_DIM,
        N_FRAMES,
        N_FOCUS,
        CHANNELS,
    )

    scalar_out = net.scalar_act(scalar_pair)
    scalar_gate = torch.sigmoid(net.scalar_gate(scalar_pair))
    coeff_packed = product * scalar_gate[:, None, None, :, :]
    coeff_packed[:, 0, net.frame_zero_index].add_(scalar_out)
    coeff = _frame_contract_packed(net.frame_contract, coeff_packed)
    if net.residual_scale is not None:
        coeff = coeff * net.residual_scale.view(1, 1, N_FOCUS, CHANNELS)
    output = coeff.reshape_as(query_flat)
    if return_product:
        return output, product_flat
    return output


def run_packed_message_grid_backward(
    net: Any,
    query_flat: Tensor,
    context_flat: Tensor,
    grad_out_flat: Tensor,
    *,
    product_flat: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return query/context adjoints while retaining packed grid intermediates."""
    query, context = _validate_contract(net, query_flat, context_flat)
    from .kernels.message_grid_product import (
        run_message_grid_product,
        run_message_grid_product_backward,
    )

    nodes = query_flat.shape[0]
    scalar_pair = torch.cat([query[:, 0], context[:, 0]], dim=-1).to(net.dtype)

    left_packed = _frame_expand_packed(net.frame_expand, query)
    right_packed = _frame_expand_packed(net.frame_expand, context)
    left = left_packed.view(nodes, COEFF_DIM * N_FRAMES, HIDDEN_CHANNELS)
    right = right_packed.view(nodes, COEFF_DIM * N_FRAMES, HIDDEN_CHANNELS)
    if product_flat is None:
        product_flat = run_message_grid_product(
            left,
            right,
            net.projector.to_grid_mat,
            net.projector.from_grid_mat,
        )
    elif (
        tuple(product_flat.shape) != (nodes, COEFF_DIM * N_FRAMES, HIDDEN_CHANNELS)
        or product_flat.dtype != torch.float32
        or product_flat.device != query_flat.device
        or not product_flat.is_contiguous()
    ):
        raise ValueError(
            "saved packed message-grid product must be contiguous FP32 with shape "
            f"({nodes}, {COEFF_DIM * N_FRAMES}, {HIDDEN_CHANNELS}) on "
            f"{query_flat.device}"
        )
    product = product_flat.view(
        nodes,
        COEFF_DIM,
        N_FRAMES,
        N_FOCUS,
        CHANNELS,
    )

    scalar_gate = torch.sigmoid(net.scalar_gate(scalar_pair))

    grad = _as_flat_ndfc(
        net,
        "grad_out",
        grad_out_flat,
        like=query_flat,
    ).to(net.dtype)
    if net.residual_scale is not None:
        grad = grad * net.residual_scale.view(1, 1, N_FOCUS, CHANNELS)
    grad_scalar_packed = _frame_contract_packed_backward(net.frame_contract, grad)

    grad_product = grad_scalar_packed * scalar_gate[:, None, None, :, :]
    grad_scalar_gate = (grad_scalar_packed * product).sum(dim=(1, 2))
    grad_scalar_out = grad_scalar_packed[:, 0, net.frame_zero_index]
    grad_scalar_logits = grad_scalar_gate * scalar_gate * (1.0 - scalar_gate)
    grad_scalar_pair = _focus_linear_backward_input(
        net.scalar_gate,
        grad_scalar_logits,
    ) + _swiglu_backward_input(scalar_pair, grad_scalar_out)

    # Broadcast multiplication preserves the degree-major einsum layout. The
    # packed CuTe adjoint requires coefficient-major compact storage.
    grad_product_flat = grad_product.contiguous().view(
        nodes,
        COEFF_DIM * N_FRAMES,
        HIDDEN_CHANNELS,
    )
    grad_left, grad_right = run_message_grid_product_backward(
        grad_product_flat,
        left,
        right,
        net.projector.to_grid_mat,
        net.projector.from_grid_mat,
    )
    grad_query = _frame_expand_packed_backward(
        net.frame_expand,
        grad_left.view(nodes, COEFF_DIM, N_FRAMES, N_FOCUS, CHANNELS),
    )
    grad_context = _frame_expand_packed_backward(
        net.frame_expand,
        grad_right.view(nodes, COEFF_DIM, N_FRAMES, N_FOCUS, CHANNELS),
    )
    grad_query[:, 0].add_(grad_scalar_pair[:, :, :CHANNELS])
    grad_context[:, 0].add_(grad_scalar_pair[:, :, CHANNELS:])
    return (
        grad_query.reshape_as(query_flat).to(dtype=query_flat.dtype),
        grad_context.reshape_as(context_flat).to(dtype=context_flat.dtype),
    )


__all__ = [
    "run_packed_message_grid_backward",
    "run_packed_message_grid_forward",
]
