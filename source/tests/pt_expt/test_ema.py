# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for exponential moving-average model weights."""

import torch

from deepmd.pt_expt.train.ema import (
    ModelEMA,
)


def test_apply_shadow_restores_parameters_shared_across_models() -> None:
    left = torch.nn.Linear(1, 1, bias=False)
    right = torch.nn.Linear(1, 1, bias=False)
    right.weight = left.weight
    models = {"left": left, "right": right}

    with torch.no_grad():
        left.weight.fill_(1.0)
    ema = ModelEMA(models, decay=0.9)
    for shadow in ema.shadow_params.values():
        shadow.fill_(2.0)

    with ema.apply_shadow(models):
        torch.testing.assert_close(left.weight, torch.full_like(left.weight, 2.0))

    torch.testing.assert_close(left.weight, torch.ones_like(left.weight))
