# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import torch

log = logging.getLogger(__name__)


def compute_smooth_weight(distance, rmin: float, rmax: float):
    """Compute smooth weight for descriptor elements."""
    if rmin >= rmax:
        raise ValueError("rmin should be less than rmax.")
    distance = torch.clamp(distance, min=rmin, max=rmax)
    uu = (distance - rmin) / (rmax - rmin)
    uu2 = uu * uu
    vv = uu2 * uu * (-6 * uu2 + 15 * uu - 10) + 1
    return vv


def compute_exp_sw(distance, rmin: float, rmax: float):
    """Compute the exponential switch function for neighbor update."""
    if rmin >= rmax:
        raise ValueError("rmin should be less than rmax.")
    distance = torch.clamp(distance, min=0.0, max=rmax)
    C = 20
    a = C / rmin
    b = rmin
    exp_sw = torch.exp(-torch.exp(a * (distance - b)))
    return exp_sw
