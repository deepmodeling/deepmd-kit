#!/usr/bin/env python3

from tensorflow.python.framework import ops
from deepmd.env import op_module
from deepmd.env import tf 

@ops.RegisterGradient("Tanh2Nvnmd")
def _Tanh2NvnmdGrad(op, grad):
    isround = op.get_attr("isround")
    nbit1 = op.get_attr("nbit1")
    nbit2 = op.get_attr("nbit2")
    nbit3 = op.get_attr("nbit3")
    prec = 2 ** nbit2
    x = op.inputs[0]
    x_abs = tf.abs(x)
    x1 = tf.clip_by_value(x_abs, 0, 2)
    x2 = tf.clip_by_value(x_abs, 0, 4)
    dydx = (132-64*x1-x2) * 0.0078125
    if (nbit2 > -1):
        dydx = dydx + tf.stop_gradient( tf.floor(dydx * prec) / prec - dydx )
    dx = dydx * grad
    if (nbit2 > -1):
        dx = dx + tf.stop_gradient( tf.floor(dx * prec) / prec - dx )
    return dx
