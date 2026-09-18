# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""In-place forward and reverse recurrences for the SM90 persistent prefix.

The CuTe gate is elementwise alias-safe for ``out=residual``: every output
element reads only the corresponding residual element, while all gate values
are derived from the separate ``z`` tensors.  The input state may therefore
become the running state without changing the recurrence.

The reverse recurrence overwrites dead saved preactivations and aliases the
running residual through ``torch.baddbmm`` to minimize edge-sized storage.
"""

from __future__ import (
    annotations,
)

import torch

from .persistent import (
    NeoPersistentComplexSaved,
    NeoPersistentComplexState,
    NeoPersistentComplexWeights,
    _empty_state_like,
    _run_gate_adjoint,
    _run_gate_forward,
    validate_neo_persistent_complex_state,
)

GATED_LAYERS = 2

__all__ = [
    "run_persistent_prefix_forward_inplace",
    "run_persistent_prefix_input_adjoint_destructive_saved",
]


def _saved(z0: list[torch.Tensor], z1: list[torch.Tensor]) -> NeoPersistentComplexSaved:
    return NeoPersistentComplexSaved(
        z0=(z0[0], z0[1]),
        z1=(z1[0], z1[1]),
    )


def run_persistent_prefix_forward_inplace(
    state: NeoPersistentComplexState,
    weights: NeoPersistentComplexWeights,
) -> tuple[NeoPersistentComplexState, NeoPersistentComplexSaved]:
    """Overwrite the caller-owned running state after each separate GEMM."""
    validate_neo_persistent_complex_state(state)
    saved_m0: list[torch.Tensor] = []
    saved_m1: list[torch.Tensor] = []
    for layer in range(GATED_LAYERS):
        z = _empty_state_like(state)
        torch.bmm(state.m0, weights.w0[layer], out=z.m0)
        torch.bmm(state.m1, weights.wc[layer], out=z.m1)
        saved_m0.append(z.m0)
        saved_m1.append(z.m1)
        _run_gate_forward(state, z, weights.gate[layer], state)
    return state, _saved(saved_m0, saved_m1)


def run_persistent_prefix_input_adjoint_destructive_saved(
    grad_out: NeoPersistentComplexState,
    saved: NeoPersistentComplexSaved,
    weights: NeoPersistentComplexWeights,
) -> NeoPersistentComplexState:
    """Overwrite each dead saved preactivation with its exact gate adjoint.

    The gate kernel stages the scalar row before writing any output.  Every
    remaining preactivation element is read and then replaced by its own
    adjoint, so input/output aliasing is safe.  Backward visits layer 1 before
    layer 0; each saved state is therefore dead when it becomes ``grad_z``.
    """
    validate_neo_persistent_complex_state(grad_out, name="grad_out")
    running = grad_out
    for layer in range(GATED_LAYERS - 1, -1, -1):
        grad_z = NeoPersistentComplexState(saved.z0[layer], saved.z1[layer])
        _run_gate_adjoint(running, grad_z, weights.gate[layer], grad_z)
        torch.baddbmm(
            running.m0,
            grad_z.m0,
            weights.w0_h[layer],
            out=running.m0,
        )
        torch.baddbmm(
            running.m1,
            grad_z.m1,
            weights.wc_h[layer],
            out=running.m1,
        )
    return running
