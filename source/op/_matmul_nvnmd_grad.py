#!/usr/bin/env python3

from tensorflow.python.framework import ops
from deepmd.env import op_module
from deepmd.env import tf 

@ops.RegisterGradient("MatmulNvnmd")
def _MatmulNvnmdGrad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    isround = op.get_attr("isround")
    nbit1 = op.get_attr("nbit1")
    nbit2 = op.get_attr("nbit2")
    nbit3 = op.get_attr("nbit3")
    dx = op_module.matmul_nvnmd(grad, tf.transpose(w), isround, nbit2, nbit3, nbit1)
    dw = op_module.matmul_nvnmd(tf.transpose(x), grad, isround, nbit2, nbit3, nbit1)
    return [dx, dw]
