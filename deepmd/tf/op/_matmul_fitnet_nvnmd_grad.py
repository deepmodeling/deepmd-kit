#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
    tf,
)


@ops.RegisterGradient("MatmulFitnetNvnmd")
def _MatmulFitnetNvnmdGrad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    nbitx = op.get_attr("nbitx")
    nbitw = op.get_attr("nbitw")
    normw = op.get_attr("normw")
    dx = op_module.matmul_fitnet_nvnmd(grad, tf.transpose(w), nbitx, nbitw, normw)
    dw = tf.matmul(tf.transpose(x), grad)
    return [dx, dw]
