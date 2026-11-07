# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for exponential moving-average model weights."""

import pytest
import torch

from deepmd.pt_expt.train.ema import (
    ModelEMA,
)


def test_apply_shadow_restores_parameters_shared_across_models() -> None:
    """Restore shared live weights after temporary shadow application."""
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
    """Reference the original per-parameter EMA arithmetic."""
    weight = 1.0 - decay
    out = {name: tensor.clone() for name, tensor in shadows.items()}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if torch.is_floating_point(param):
                out[name].lerp_(param.detach(), weight=weight)
    return out


def test_foreach_update_matches_naive_lerp() -> None:
    """Compare grouped updates with the original lerp implementation."""
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
    ema.update()

    assert ema._update_groups
    assert not ema._update_fallback
    for name, shadow in ema.shadow_params.items():
        torch.testing.assert_close(shadow, expected[name])


def test_update_groups_by_dtype() -> None:
    """Separate float32 and float64 updates into their own groups."""

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

    ema.update()
    for shadow in ema.shadow_params.values():
        torch.testing.assert_close(shadow, torch.ones_like(shadow))


def test_update_does_not_walk_named_parameters(monkeypatch) -> None:
    """Keep name traversal out of the optimizer-step hot path."""
    model = torch.nn.Linear(3, 3)
    ema = ModelEMA(model, decay=0.9)

    def _boom(*_args, **_kwargs):
        raise AssertionError("update must not walk named parameters")

    monkeypatch.setattr(ModelEMA, "_named_model_parameters", staticmethod(_boom))
    ema.update()


def test_rebind_rebuilds_plan_for_new_model() -> None:
    """Update the explicitly rebound model while preserving EMA history."""
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
    ema.update()
    torch.testing.assert_close(
        ema.shadow_params["weight"],
        torch.full_like(ema.shadow_params["weight"], 2.5),
    )


@pytest.mark.parametrize("layout", ["transpose", "channels_last", "sliced"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_strided_layouts(layout, device) -> None:
    """Dense non-contiguous tensors stay grouped; gapped views stay isolated."""
    if layout == "transpose":
        value = torch.ones(3, 4, device=device).T
    elif layout == "channels_last":
        value = torch.ones(2, 3, 4, 5, device=device).to(
            memory_format=torch.channels_last
        )
    else:
        value = torch.ones(3, 8, device=device)[:, ::2]
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(value))
    ema = ModelEMA(model, decay=0.5)
    assert bool(ema._update_fallback) == (layout == "sliced")
    assert bool(ema._update_groups) == (layout != "sliced")
    expected = ema.shadow_params["weight"].clone()
    for _ in range(20):
        with torch.no_grad():
            model.weight.add_(0.25)
            expected.lerp_(model.weight, 0.5)
        ema.update()
        torch.testing.assert_close(ema.shadow_params["weight"], expected)


@pytest.mark.parametrize("sparse_layout", ["coo", "csr"])
def test_sparse_parameters_rejected_at_bind(sparse_layout) -> None:
    """Unsupported layouts fail clearly at construction and atomic rebind."""
    first = torch.nn.Linear(2, 2, bias=False)
    ema = ModelEMA(first, decay=0.5)
    second = torch.nn.Linear(2, 2, bias=False)
    value = torch.ones(2, 2)
    value = value.to_sparse() if sparse_layout == "coo" else value.to_sparse_csr()
    second.weight = torch.nn.Parameter(value)
    match = f"weight.*torch.sparse_{sparse_layout}"
    with pytest.raises(ValueError, match=match):
        ModelEMA(second, decay=0.5)
    old_plan = ema._update_groups
    with pytest.raises(ValueError, match=match):
        ema.rebind(second)
    assert ema._update_groups is old_plan
    with torch.no_grad():
        first.weight.fill_(5.0)
    expected = ema.shadow_params["weight"].lerp(first.weight.detach(), 0.5)
    ema.update()
    torch.testing.assert_close(ema.shadow_params["weight"], expected)


@pytest.mark.parametrize("mismatch", ["shape", "names", "dtype"])
def test_rejected_rebind_preserves_plan(mismatch) -> None:
    """Reject changed structure before committing any new parameter references."""
    first = torch.nn.Linear(2, 2, bias=False)
    ema = ModelEMA(first, decay=0.5)
    second = torch.nn.Linear(
        2, 1 if mismatch == "shape" else 2, bias=mismatch == "names"
    )
    if mismatch == "dtype":
        second.double()
    expected_error = KeyError if mismatch == "names" else ValueError
    with pytest.raises(expected_error, match=r"keys|weight"):
        ema.rebind(second)
    with torch.no_grad():
        first.weight.add_(2.0)
    expected = ema.shadow_params["weight"].lerp(first.weight.detach(), 0.5)
    ema.update()
    torch.testing.assert_close(ema.shadow_params["weight"], expected)


def test_multitask_shared_parameters_and_checkpoint() -> None:
    """Shared inputs, checkpoint reload and shadow application preserve binding."""
    left = torch.nn.Linear(2, 2)
    right = torch.nn.Linear(2, 2)
    right.weight = left.weight
    models = {"right": right, "left": left}
    ema = ModelEMA(models, decay=0.5)
    expected = {name: tensor.clone() for name, tensor in ema.shadow_params.items()}
    for _ in range(10):
        with torch.no_grad():
            left.weight.add_(0.5)
            left.bias.add_(0.25)
            right.bias.sub_(0.25)
            for key, model in models.items():
                for name, param in model.named_parameters():
                    expected[f"{key}.{name}"].lerp_(param, 0.5)
        ema.update()
    for name in expected:
        torch.testing.assert_close(ema.shadow_params[name], expected[name])
    state = ema.state_dict()
    ids = {name: id(tensor) for name, tensor in ema.shadow_params.items()}
    ema.update()
    ema.load_state_dict(state)
    assert ids == {name: id(tensor) for name, tensor in ema.shadow_params.items()}
    resumed = ModelEMA(models, decay=0.5, state=state)
    before = left.weight.detach().clone()
    with ema.apply_shadow(models):
        torch.testing.assert_close(left.weight, expected["left.weight"])
    torch.testing.assert_close(left.weight, before)
    ema.update()
    resumed.update()
    for name in expected:
        torch.testing.assert_close(ema.shadow_params[name], resumed.shadow_params[name])


def test_rebind_after_parameter_replacement() -> None:
    """Explicit rebinding follows new Parameter objects without resetting history."""
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model, decay=0.5)
    model.weight = torch.nn.Parameter(torch.full((2, 2), 5.0))
    ema.rebind(model)
    ema.update()
    torch.testing.assert_close(ema.shadow_params["weight"], torch.full((2, 2), 3.0))


def test_singleton_strides_do_not_disable_foreach() -> None:
    """Size-one strides do not affect dense tensor traversal in foreach."""
    model = torch.nn.Module()
    model.register_parameter("weight", torch.nn.Parameter(torch.ones(2, 1, 3)))
    ema = ModelEMA(model, decay=0.5)
    model.weight = torch.nn.Parameter(
        torch.full((2, 3), 5.0).as_strided((2, 1, 3), (3, 123, 1))
    )
    ema.rebind(model)
    assert not ema._update_fallback
    ema.update()
    torch.testing.assert_close(ema.shadow_params["weight"], torch.full((2, 1, 3), 3.0))
