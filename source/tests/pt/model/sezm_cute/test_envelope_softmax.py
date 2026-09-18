# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Numerical acceptance for the CuTe envelope-gated softmax."""

from __future__ import (
    annotations,
)

import importlib

import pytest
import torch

from deepmd.pt.model.descriptor.sezm_nn.attention import (
    segment_envelope_gated_softmax,
)


def _runtime_skip_reason() -> str | None:
    if not torch.cuda.is_available():
        return "CuTe envelope-softmax acceptance requires CUDA"
    try:
        importlib.import_module("cutlass.cute")
        importlib.import_module("cuda.bindings.driver")
    except Exception as exc:  # pragma: no cover - runtime dependent
        return f"CuTe DSL runtime unavailable: {exc}"
    return None


_SKIP_REASON = _runtime_skip_reason()


@pytest.mark.skipif(_SKIP_REASON is not None, reason=_SKIP_REASON or "unavailable")
@pytest.mark.parametrize(
    ("logit_scale", "edge_gate"),
    [
        (1.0, 1.0),
        (15.0, 1.0e-2),
        (110.0, 1.0e-23),
        (-120.0, 1.0),
    ],
)
def test_cute_envelope_softmax_matches_eager_across_extreme_frames(
    logit_scale: float,
    edge_gate: float,
) -> None:
    from deepmd.pt_expt.kernels.cute.sezm.so2.kernels.envelope_softmax import (
        compile_envelope_softmax_forward,
    )

    logits = torch.tensor(
        [
            [logit_scale, logit_scale - 0.5],
            [logit_scale - 1.0, logit_scale - 1.5],
            [logit_scale - 0.25, logit_scale - 0.75],
            [logit_scale - 2.0, logit_scale - 2.5],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    gate = torch.full((4,), edge_gate, device="cuda", dtype=torch.float32)
    dst = torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.long)
    dst_ptr = torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32)
    z_bias_raw = torch.tensor([0.1, -0.3], device="cuda", dtype=torch.float32)
    eps = 1.0e-7

    expected = segment_envelope_gated_softmax(
        logits.view(4, 2, 1),
        gate,
        dst,
        2,
        z_bias_raw.view(2, 1),
        eps,
    ).view(4, 2)
    actual = torch.empty_like(logits)
    group_max = torch.empty((2, 2), device="cuda", dtype=torch.float32)
    denom = torch.empty_like(group_max)
    run = compile_envelope_softmax_forward(128, eps)
    run(logits, gate, dst_ptr, z_bias_raw, actual, group_max, denom)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual, expected, atol=5.0e-5, rtol=5.0e-5)
    assert torch.isfinite(actual).all()
