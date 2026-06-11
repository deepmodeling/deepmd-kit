# SPDX-License-Identifier: LGPL-3.0-or-later
"""Training utility functions."""

from __future__ import (
    annotations,
)

import math
import os
from contextlib import (
    contextmanager,
)
from typing import (
    TYPE_CHECKING,
)

import torch

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
    )


def clip_grad_norm_with_stable_fallback(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    use_stable_fallback: bool = True,
    named_parameters: Callable[[], Iterable[tuple[str, torch.nn.Parameter]]]
    | None = None,
) -> torch.Tensor:
    """Clip gradients, falling back to a scaled norm if the global norm overflows.

    The normal path returns PyTorch's native norm tensor. The stable fallback
    returns a float64 scalar tensor on the first gradient device so very large
    finite norms do not collapse back to inf when reported.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0, dtype=torch.float64, device="cpu")

    if not use_stable_fallback:
        total_norm = torch.nn.utils.clip_grad_norm_(
            params,
            max_norm,
        )
        if not torch.isfinite(total_norm):
            raise_nonfinite_gradient_norm(
                collect_named_grads(params, named_parameters), total_norm
            )
        return total_norm

    try:
        return torch.nn.utils.clip_grad_norm_(
            params,
            max_norm,
            error_if_nonfinite=True,
        )
    except RuntimeError as err:
        message = str(err).lower()
        if "non-finite" not in message and "nonfinite" not in message:
            raise
        return stable_clip_grad_norm(
            collect_named_grads(params, named_parameters), max_norm
        )


def collect_named_grads(
    parameters: list[torch.nn.Parameter],
    named_parameters: Callable[[], Iterable[tuple[str, torch.nn.Parameter]]] | None,
) -> list[tuple[str, torch.nn.Parameter]]:
    if named_parameters is None:
        return [(f"param_{idx}", param) for idx, param in enumerate(parameters)]
    return [
        (name, param) for name, param in named_parameters() if param.grad is not None
    ]


def raise_nonfinite_gradient_norm(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    total_norm: torch.Tensor,
) -> None:
    bad_params = []
    for name, param in named_parameters:
        grad_norm = param.grad.data.norm()
        if not torch.isfinite(grad_norm):
            bad_params.append(
                f"  {name}: grad_norm={grad_norm}, shape={list(param.shape)}"
            )
    detail = (
        "\n".join(bad_params)
        if bad_params
        else "  (all individual grads finite, overflow in norm reduction)"
    )
    raise RuntimeError(
        f"Non-finite gradient norm: {total_norm}\n"
        f"Parameters with non-finite gradients:\n{detail}"
    )


def stable_clip_grad_norm(
    named_parameters: list[tuple[str, torch.nn.Parameter]],
    max_norm: float,
) -> torch.Tensor:
    """Clip finite gradients with a scaled L2 norm to avoid overflow."""
    bad_params = []
    scale = 0.0
    first_device = named_parameters[0][1].grad.device

    # === Step 1. Find the largest finite gradient magnitude ===
    for name, param in named_parameters:
        grad = param.grad.detach()
        values = grad.coalesce().values() if grad.is_sparse else grad
        if not bool(torch.isfinite(values).all().item()):
            grad_norm = grad.norm()
            bad_params.append(
                f"  {name}: grad_norm={grad_norm}, shape={list(param.shape)}"
            )
            continue
        if values.numel() > 0:
            scale = max(scale, float(values.abs().max().item()))

    if bad_params:
        detail = "\n".join(bad_params)
        raise RuntimeError(
            "Non-finite gradient norm: non-finite\n"
            f"Parameters with non-finite gradients:\n{detail}"
        )
    if scale == 0.0:
        return torch.zeros((), dtype=torch.float64, device=first_device)

    # === Step 2. Accumulate squared gradients after scaling by max magnitude ===
    scaled_ssq = 0.0
    for _, param in named_parameters:
        grad = param.grad.detach()
        values = grad.coalesce().values() if grad.is_sparse else grad
        scaled = values.to(torch.float64) / scale
        scaled_ssq += float(torch.sum(scaled * scaled).item())

    total_norm = scale * math.sqrt(scaled_ssq)
    if not math.isfinite(total_norm):
        raise RuntimeError(
            f"Non-finite gradient norm: {total_norm}\n"
            "Parameters with non-finite gradients:\n"
            "  (all individual grads finite, overflow in stable norm reduction)"
        )

    clip_coef = float(max_norm) / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for _, param in named_parameters:
            param.grad.detach().mul_(clip_coef)

    return torch.tensor(total_norm, dtype=torch.float64, device=first_device)


@contextmanager
def scoped_env_defaults(defaults: dict[str, str]) -> Generator[None, None, None]:
    """Temporarily set missing environment variables and restore them afterward."""
    previous = {key: os.environ.get(key) for key in defaults}
    try:
        for key, value in defaults.items():
            os.environ.setdefault(key, value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
