# SPDX-License-Identifier: LGPL-3.0-or-later
"""Lebedev quadrature data loader for SeZM S2 projections."""

from __future__ import (
    annotations,
)

import torch

from deepmd.dpmodel.utils.lebedev import (
    LEBEDEV_PRECISION_TO_NPOINTS,
)
from deepmd.dpmodel.utils.lebedev import load_lebedev_rule as load_lebedev_rule_np

__all__ = [
    "LEBEDEV_PRECISION_TO_NPOINTS",
    "load_lebedev_rule",
]


def load_lebedev_rule(
    precision: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load one Lebedev rule from the packaged compressed data file.

    Parameters
    ----------
    precision
        Algebraic precision of the requested Lebedev rule.
    dtype
        Output tensor dtype.
    device
        Output tensor device.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Cartesian unit points with shape ``(A, 3)`` and normalized weights with
        shape ``(A,)``. The weights sum to one, so the sphere integral is
        ``4*pi*sum(weights*f(points))``.
    """
    points_np, weights_np = load_lebedev_rule_np(precision)
    points = torch.as_tensor(points_np, dtype=dtype, device=device)
    weights = torch.as_tensor(weights_np, dtype=dtype, device=device)
    return points, weights
