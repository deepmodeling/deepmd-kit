
import numpy as np

def map_nvnmd(x, map_y, map_dy, prec, nbit=None):
    """: x, map_y, map_dy, prec, nbit
    """
    xk = int(np.floor(x / prec))
    dx = x - xk * prec 
    y = map_y[xk] + map_dy[xk] * dx 
    return y 
