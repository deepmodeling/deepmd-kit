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


def _naive_lerp_update(
    shadows: dict[str, torch.Tensor],
    model: torch.nn.Module,
    decay: float,
) -> dict[str, torch.Tensor]:
    weight = 1.0 - decay
    out = {name: tensor.clone() for name, tensor in shadows.items()}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if torch.is_floating_point(param):
                out[name].lerp_(param.detach(), weight=weight)
    return out


def test_foreach_update_matches_naive_lerp() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.Linear(8, 2),
    )
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-1.0, 1.0)

    decay = 0.875
    ema = ModelEMA(model, decay=decay)
    shadows_before = {
        name: tensor.clone() for name, tensor in ema.shadow_params.items()
    }

    with torch.no_grad():
        for param in model.parameters():
            param.add_(0.25)

    expected = _naive_lerp_update(shadows_before, model, decay)
    ema.update(model)

    assert ema._update_groups
    assert not ema._update_fallback
    for name, shadow in ema.shadow_params.items():
        torch.testing.assert_close(shadow, expected[name])


def test_update_groups_by_dtype() -> None:
    class MixedDtype(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fp32 = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
            self.fp64 = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))

    model = MixedDtype()
    ema = ModelEMA(model, decay=0.5)
    assert len(ema._update_groups) == 2
    assert not ema._update_fallback
    dtypes = {params[0].dtype for _, params in ema._update_groups}
    assert dtypes == {torch.float32, torch.float64}

    ema.update(model)
    for shadow in ema.shadow_params.values():
        torch.testing.assert_close(shadow, torch.ones_like(shadow))


def test_update_does_not_walk_named_parameters(monkeypatch) -> None:
    model = torch.nn.Linear(3, 3)
    ema = ModelEMA(model, decay=0.9)

    def _boom(*_args, **_kwargs):
        raise AssertionError("update must not walk named parameters")

    monkeypatch.setattr(ModelEMA, "_named_model_parameters", staticmethod(_boom))
    ema.update(model)


def test_rebind_rebuilds_plan_for_new_model() -> None:
    first = torch.nn.Linear(2, 2)
    second = torch.nn.Linear(2, 2)
    with torch.no_grad():
        first.weight.fill_(1.0)
        first.bias.fill_(1.0)
        second.weight.copy_(first.weight)
        second.bias.copy_(first.bias)

    ema = ModelEMA(first, decay=0.5)
    first_param_ids = {id(p) for _, params in ema._update_groups for p in params}

    ema.rebind(second)
    second_param_ids = {id(p) for _, params in ema._update_groups for p in params}
    assert first_param_ids.isdisjoint(second_param_ids)

    with torch.no_grad():
        second.weight.fill_(4.0)
        second.bias.fill_(4.0)
    ema.update(second)
    torch.testing.assert_close(
        ema.shadow_params["weight"],
        torch.full_like(ema.shadow_params["weight"], 2.5),
    )
