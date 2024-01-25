# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


def r2s(r, rmin, rmax):
    v = 0.0
    if (r > 0.01) and (r <= rmin):
        v = 1.0 / r
    elif (r > rmin) and (r <= rmax):
        uu = (r - rmin) / (rmax - rmin)
        v = (uu * uu * uu * (-6 * uu * uu + 15 * uu - 10) + 1) / r
    return v


def map_nvnmd(x, map_y, map_dy, prec, nbit=None):
    r"""Mapping function implemented by numpy."""
    xk = int(np.floor(x / prec))
    dx = x - xk * prec
    y = map_y[xk] + map_dy[xk] * dx
    return y
