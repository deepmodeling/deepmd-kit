# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

import numpy as np


def r2s(r: float, rmin: float, rmax: float) -> float:
    v = 0.0
    if (r > 0.01) and (r <= rmin):
        v = 1.0 / r
    elif (r > rmin) and (r <= rmax):
        uu = (r - rmin) / (rmax - rmin)
        v = (uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1) / r
    return v


def map_nvnmd(
    x: Any, map_y: Any, map_dy: Any, prec: float, nbit: int | None = None
) -> Any:
    r"""Mapping function implemented by numpy."""
    xk = int(np.floor(x / prec))
    dx = x - xk * prec
    y = map_y[xk] + map_dy[xk] * dx
    return y
