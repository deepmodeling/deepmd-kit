#!/usr/bin/env python3

from tensorflow.python.framework import ops
from deepmd.env import op_module
from deepmd.env import tf 

@ops.RegisterGradient("MapNvnmd")
def _MapNvnmdGrad(op, grad):
    x = op.inputs[0]
    v = op.inputs[1]
    dv = op.inputs[2]
    grad_v = op.inputs[3]
    grad_dv = op.inputs[4]
    prec = op.get_attr("prec")
    nbit = op.get_attr("nbit")
    y = op.outputs[0]
    dydx = op_module.map_nvnmd(x, grad_v, grad_dv, tf.zeros_like(v), tf.zeros_like(dv), prec, nbit)
    dydx = op_module.quantize_nvnmd(dydx, 0, nbit, -1, -1)
    dx = tf.reshape(tf.reduce_sum(dydx * grad, axis=1), [-1, 1])

    d_v = None
    d_dv = None
    d_grad_v = None
    d_grad_dv = None
    return [dx, d_v, d_dv, d_grad_v, d_grad_dv]

