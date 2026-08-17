# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Shared runtime policy for the opt-in Neo CuTe inference path."""

from __future__ import (
    annotations,
)

import os

import torch

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})
NEO_CUTE_INFER_ENV = "DP_NEO_CUTE_INFER"
SM80_PROFILE_CAPABILITIES = frozenset({(8, 0), (8, 6)})
SM90_CAPABILITY = (9, 0)
FUSED_SO2_GATE_CAPABILITIES = frozenset({(8, 9), (12, 0)})
OUTPUT_GRID_SM90_C96_ASYMMETRIC_PANELS_ENV = (
    "DP_CUTE_OUTPUT_GRID_SM90_C96_ASYMMETRIC_PANELS"
)
READOUT_INPUT_FOLD_SM90_ENV = "DP_CUTE_READOUT_INPUT_FOLD_SM90"
SUPPORTED_K1_CAPABILITIES = SM80_PROFILE_CAPABILITIES | frozenset(
    {(8, 9), (9, 0), (10, 0), (12, 0)}
)
_GIE_DEFAULT_CAPABILITIES = SM80_PROFILE_CAPABILITIES
INT32_MAX = (1 << 31) - 1
K1_VALUES_PER_EDGE = 10 * 64
K1_VALUES_PER_NODE = 16 * 64
PORTABLE_TILED_BACKEND = "portable_tiled"
PYTORCH_BACKEND = "pytorch"
SUPPORTED_HIDDEN_CHANNELS = (96, 192)
_OUTPUT_GRID_ARCH_BACKENDS = {
    "sm80": {
        96: PORTABLE_TILED_BACKEND,
        192: PORTABLE_TILED_BACKEND,
    },
    "sm90": {
        96: PORTABLE_TILED_BACKEND,
        192: PORTABLE_TILED_BACKEND,
    },
}


def _env_override(name: str) -> bool | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return False


def is_cute_infer_enabled() -> bool:
    """Return whether the process opted into the full Neo CuTe K1 path.

    ``DP_NEO_CUTE_INFER`` is deliberately separate from the
    ``DP_CUTE_INFER`` inner SO2 value-path selector. The full K1 replacement
    may therefore coexist with ``DP_TRITON_INFER=2``.
    """
    return _env_override(NEO_CUTE_INFER_ENV) is True


def _current_compute_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    try:
        return tuple(torch.cuda.get_device_capability())
    except RuntimeError:
        return None


def output_grid_arch_key(compute_capability: tuple[int, int]) -> str:
    """Return the architecture-family key used for output-grid dispatch."""
    if tuple(compute_capability) in SM80_PROFILE_CAPABILITIES:
        return "sm80"
    major, minor = compute_capability
    return f"sm{int(major)}{int(minor)}"


def select_output_grid_backend(
    compute_capability: tuple[int, int],
    hidden_channels: int,
) -> str:
    """Select a width-specific CuTe or PyTorch output-grid backend."""
    hidden_channels = int(hidden_channels)
    if hidden_channels not in SUPPORTED_HIDDEN_CHANNELS:
        return PYTORCH_BACKEND
    architecture_backends = _OUTPUT_GRID_ARCH_BACKENDS.get(
        output_grid_arch_key(compute_capability)
    )
    if architecture_backends is None:
        return PYTORCH_BACKEND
    return architecture_backends.get(hidden_channels, PYTORCH_BACKEND)


def is_sm80_profile_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Return whether the shared SM80/SM86 profile is selected."""
    if compute_capability is None:
        compute_capability = _current_compute_capability()
    return (
        is_cute_infer_enabled()
        and compute_capability is not None
        and tuple(compute_capability) in SM80_PROFILE_CAPABILITIES
    )


def _sm80_profile_feature(
    name: str,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Apply an SM80-family default with an explicit disable override."""
    if not is_sm80_profile_enabled(compute_capability):
        return False
    return _env_override(name) is not False


def _profile_or_explicit_feature(
    name: str,
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Default on for the SM80 profile; otherwise require explicit opt-in."""
    if is_sm80_profile_enabled(compute_capability):
        return _env_override(name) is not False
    return is_cute_infer_enabled() and _env_override(name) is True


@torch.compiler.assume_constant_result
def is_k1_thin_wrapper_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select compile-visible K1 dispatch for the SM80 profile."""
    return _profile_or_explicit_feature(
        "DP_CUTE_K1_THIN_WRAPPER",
        compute_capability,
    )


@torch.compiler.assume_constant_result
def is_cute_strict_enabled() -> bool:
    """Return whether expensive CuTe contract assertions are requested."""
    return _env_override("DP_CUTE_STRICT") is True


def is_output_grid_bwd_sm80_c96_n48_panel_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select the C=96, N=48 SM80 panel adjoint."""
    return _sm80_profile_feature(
        "DP_CUTE_OUTPUT_GRID_BWD_SM80_C96_N48_PANEL",
        compute_capability,
    )


def is_output_grid_fwd_sm80_c96_n48_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select the C=96, N=48 SM80 forward."""
    return _sm80_profile_feature(
        "DP_CUTE_OUTPUT_GRID_FWD_SM80_C96_N48",
        compute_capability,
    )


@torch.compiler.assume_constant_result
def is_output_grid_sm90_c96_asymmetric_panels_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select the C96 N64+N32 panels only on exact SM90."""
    if compute_capability is None:
        compute_capability = _current_compute_capability()
    return _master_gated_feature(
        OUTPUT_GRID_SM90_C96_ASYMMETRIC_PANELS_ENV,
        default=compute_capability is not None
        and tuple(compute_capability) == SM90_CAPABILITY,
    )


@torch.compiler.assume_constant_result
def is_readout_input_fold_sm80_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select the frozen C=192 Neo readout fold on SM80."""
    return _sm80_profile_feature(
        "DP_CUTE_READOUT_INPUT_FOLD_SM80",
        compute_capability,
    )


@torch.compiler.assume_constant_result
def is_readout_input_fold_sm90_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select the frozen C=192 Neo readout fold only on exact SM90."""
    if compute_capability is None:
        compute_capability = _current_compute_capability()
    return _master_gated_feature(
        READOUT_INPUT_FOLD_SM90_ENV,
        default=compute_capability is not None
        and tuple(compute_capability) == SM90_CAPABILITY,
    )


def is_readout_input_fold_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select the architecture-specific frozen readout fold."""
    return is_readout_input_fold_sm80_enabled(
        compute_capability
    ) or is_readout_input_fold_sm90_enabled(compute_capability)


def is_supported_k1_capability(compute_capability: tuple[int, int]) -> bool:
    """Return whether K1 supports this compute capability."""
    return tuple(compute_capability) in SUPPORTED_K1_CAPABILITIES


def k1_int32_indexing_is_safe(
    *,
    edge_count: int,
    node_count: int,
) -> bool:
    """Check every flattened K1 offset represented with signed Int32."""
    if edge_count < 0 or node_count < 0:
        return False
    return (
        edge_count <= INT32_MAX // K1_VALUES_PER_EDGE
        and node_count <= INT32_MAX // K1_VALUES_PER_NODE
    )


@torch.compiler.assume_constant_result
def uses_strict_fp32_matmul() -> bool:
    """Read CUDA matmul precision outside Dynamo and fail closed on TF32."""
    matmul = torch.backends.cuda.matmul
    try:
        precision = matmul.fp32_precision
    except AttributeError:
        precision = None
    except RuntimeError:
        return False
    if precision is not None and precision != "none":
        return precision == "ieee"
    try:
        return not matmul.allow_tf32
    except RuntimeError:
        return False


def _master_gated_feature(name: str, *, default: bool) -> bool:
    if not is_cute_infer_enabled() or not default:
        return False
    override = _env_override(name)
    return override is not False


def is_gie_enabled(compute_capability: tuple[int, int]) -> bool:
    """Select the optimized geometric-initial-embedding path."""
    return _master_gated_feature(
        "DP_CUTE_GIE",
        default=compute_capability in _GIE_DEFAULT_CAPABILITIES,
    )


def is_packed_wigner_enabled(compute_capability: tuple[int, int]) -> bool:
    """Select packed Wigner storage required by the optimized K1 profiles."""
    return _master_gated_feature(
        "DP_CUTE_K1_PACKED_WIGNER",
        default=compute_capability in SUPPORTED_K1_CAPABILITIES,
    )


@torch.compiler.assume_constant_result
def is_k1_eager_island_enabled(
    compute_capability: tuple[int, int] | None = None,
) -> bool:
    """Select the SM80 neighbor-list eager island for the Neo K1 path."""
    if not is_cute_infer_enabled():
        return False
    override = _env_override("DP_CUTE_K1_EAGER_ISLANDS")
    if override is not None:
        return override
    if compute_capability is None:
        compute_capability = _current_compute_capability()
    return compute_capability is not None and tuple(compute_capability) == (8, 0)
