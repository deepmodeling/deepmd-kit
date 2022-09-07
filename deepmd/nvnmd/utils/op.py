
import numpy as np


def map_nvnmd(x, map_y, map_dy, prec, nbit=None):
    r"""Mapping function implemented by numpy
    """
    xk = int(np.floor(x / prec))
    dx = x - xk * prec
    y = map_y[xk] + map_dy[xk] * dx
    return y
