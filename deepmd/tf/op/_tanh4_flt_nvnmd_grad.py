#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    tf,
)


@ops.RegisterGradient("Tanh4FltNvnmd")
def _Tanh4FltNvnmdGrad(op, grad):
    prechi = 2**23
    preclo = 2**19
    x = op.inputs[0]
    xa = tf.abs(x)
    xc = tf.clip_by_value(xa, 0, 2)
    xhi = xc + tf.stop_gradient(tf.floor(xc * prechi) / prechi - xc)
    xlo = xc + tf.stop_gradient(tf.floor(xc * preclo) / preclo - xc)
    xx = xhi * xlo
    xxhi = xx + tf.stop_gradient(tf.floor(xx * prechi) / prechi - xx)
    xxlo = xx + tf.stop_gradient(tf.floor(xx * preclo) / preclo - xx)
    #
    dydx = xxlo * (xhi / 4 - 3 / 4) + 1
    # dydx = xxhi * (xlo/4 - 3/4) + 1
    dydxhi = dydx + tf.stop_gradient(tf.floor(dydx * prechi) / prechi - dydx)
    dydxlo = dydx + tf.stop_gradient(tf.floor(dydx * preclo) / preclo - dydx)
    #
    gradhi = grad + tf.stop_gradient(tf.floor(grad * prechi) / prechi - grad)
    dx = dydxlo * gradhi
    dx = dx + tf.stop_gradient(tf.floor(dx * prechi) / prechi - dx)
    return dx
