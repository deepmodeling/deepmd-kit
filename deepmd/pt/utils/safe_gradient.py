# SPDX-License-Identifier: LGPL-3.0-or-later
"""Safe versions of some functions that have problematic gradients.

Check https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
for more information.
"""

import torch


def safe_for_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Safe version of sqrt that has a gradient of 0 at x = 0."""
    mask = x > 0.0
    x_safe = torch.where(mask, x, torch.ones_like(x))
    return torch.where(mask, torch.sqrt(x_safe), torch.zeros_like(x))


def safe_for_norm(
    x: torch.Tensor,
    dim: int | None = None,
    keepdim: bool = False,
    ord: float = 2.0,
) -> torch.Tensor:
    """Safe version of vector_norm that has a gradient of 0 at x = 0."""
    if dim is None:
        mask = torch.sum(torch.square(x)) > 0
        x_safe = torch.where(mask, x, torch.ones_like(x))
        norm = torch.linalg.norm(x_safe, ord=ord)
        return torch.where(mask, norm, torch.zeros_like(norm))

    dim_list = [dim]
    mask = torch.sum(torch.square(x), dim=dim_list, keepdim=True) > 0
    mask_out = mask if keepdim else (torch.sum(torch.square(x), dim=dim_list) > 0)

    x_safe = torch.where(mask, x, torch.ones_like(x))
    norm = torch.linalg.norm(x_safe, ord=ord, dim=dim_list, keepdim=keepdim)
    return torch.where(mask_out, norm, torch.zeros_like(norm))
