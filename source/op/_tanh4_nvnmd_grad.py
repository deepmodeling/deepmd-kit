#!/usr/bin/env python3

from tensorflow.python.framework import ops
from deepmd.env import op_module
from deepmd.env import tf 

@ops.RegisterGradient("Tanh4Nvnmd")
def _Tanh4NvnmdGrad(op, grad):
    isround = op.get_attr("isround")
    nbit1 = op.get_attr("nbit1")
    nbit2 = op.get_attr("nbit2")
    nbit3 = op.get_attr("nbit3")
    prec = 2 ** nbit2
    x = op.inputs[0]
    xc = tf.clip_by_value(x, -2, 2)
    xa = tf.abs(xc)
    xx = xa * xa 
    if (nbit2 > -1):
        xx = xx + tf.stop_gradient(tf.floor(xx * prec) / prec - xx)
    #
    dydx = xx * (xa/4 - 3/4) + 1
    if (nbit2 > -1):
        dydx = dydx + tf.stop_gradient( tf.floor(dydx * prec) / prec - dydx)
    #
    dx = dydx * grad
    if (nbit2 > -1):
        dx = dx + tf.stop_gradient( tf.floor(dx * prec) / prec - dx )
    return dx
