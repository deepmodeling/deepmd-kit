# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Implementation differentials for the live packed Neo Wigner path."""

from __future__ import (
    annotations,
)

import importlib

import pytest
import torch

TOL = 5.0e-5


def _cute_runtime_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "packed implementation differentials require CUDA"
    try:
        importlib.import_module("cutlass.cute")
        importlib.import_module("cuda.bindings.driver")
    except Exception as exc:  # pragma: no cover - runtime dependent
        return f"packed implementation differentials require CuTe DSL: {exc}"
    return None


def _randn(
    *shape: int,
    generator: torch.Generator,
    scale: float = 0.1,
) -> torch.Tensor:
    return scale * torch.randn(
        *shape,
        generator=generator,
        device="cuda",
        dtype=torch.float32,
    )


def _dense_wigner(edge_count: int, generator: torch.Generator) -> torch.Tensor:
    from deepmd.kernels.cute.neo import k1_wigner_layout as layout

    dense = torch.zeros(
        edge_count,
        16,
        16,
        device="cuda",
        dtype=torch.float32,
    )
    for start, stop in zip(
        layout.FULL_BLOCK_OFFSETS[:-1],
        layout.FULL_BLOCK_OFFSETS[1:],
        strict=True,
    ):
        dense[:, start:stop, start:stop] = _randn(
            edge_count,
            stop - start,
            stop - start,
            generator=generator,
        )
    return dense


_CUTE_SKIP_REASON = _cute_runtime_skip_reason()


@pytest.mark.skipif(
    _CUTE_SKIP_REASON is not None,
    reason=_CUTE_SKIP_REASON or "CuTe runtime unavailable",
)
def test_panel_native_quaternion_backward_matches_dense_torch():
    from deepmd.kernels.cute.neo import k1_wigner_layout as layout
    from deepmd.kernels.cute.neo import (
        k4_wignerd,
    )
    from deepmd.pt.model.descriptor.sezm_nn.wignerd import (
        WignerDCalculator,
    )

    generator = torch.Generator(device="cuda").manual_seed(20260702)
    q_value = _randn(3, 4, generator=generator, scale=1.0)
    q_value = q_value / q_value.norm(dim=-1, keepdim=True)
    grad_panel = _randn(3, 46, generator=generator)

    q_panel = q_value.detach().requires_grad_(True)
    panel = k4_wignerd._wignerd_panel_op(q_panel)
    grad_q_panel = torch.autograd.grad((panel * grad_panel).sum(), q_panel)[0]

    q_dense = q_value.detach().requires_grad_(True)
    dense, dense_t = WignerDCalculator(lmax=3, dtype=q_dense.dtype).to("cuda")(q_dense)
    entries = tuple(layout.iter_packed_entries())
    rows = [entry.full_row for entry in entries]
    cols = [entry.full_col for entry in entries]
    dense_weight = torch.zeros_like(dense)
    dense_weight[:, rows, cols] = grad_panel
    dense_loss = (dense * dense_weight).sum() + 0.0 * dense_t.sum()
    grad_q_dense = torch.autograd.grad(dense_loss, q_dense)[0]

    torch.testing.assert_close(
        panel,
        dense[..., rows, cols],
        rtol=TOL,
        atol=TOL,
    )
    torch.testing.assert_close(grad_q_panel, grad_q_dense, rtol=TOL, atol=TOL)
