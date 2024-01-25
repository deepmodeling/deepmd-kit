#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
    tf,
)


@ops.RegisterGradient("MatmulFlt2fixNvnmd")
def _MatmulFlt2fixNvnmdGrad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    # transpose for 2-dimension and 3-dimension multiplication
    if len(x.shape) == 3:
        x_T = tf.transpose(x, [0, 2, 1])
        w_T = tf.transpose(w, [0, 2, 1])
    else:
        x_T = tf.transpose(x)
        w_T = tf.transpose(w)
    # calcualte
    # dx = tf.matmul(grad, w_T)
    # dw = tf.matmul(x_T, grad)
    dx = op_module.matmul_flt_nvnmd(grad, w_T, 1, 1)
    dw = op_module.matmul_flt_nvnmd(x_T, grad, 1, 1)
    # add shape for output of matmul_nvnmd
    shx = x.shape.as_list()
    shw = w.shape.as_list()
    shx = [None if (d == -1) else d for d in shx]
    shw = [None if (d == -1) else d for d in shw]
    dx = tf.ensure_shape(dx, shx)
    dw = tf.ensure_shape(dw, shw)
    return [dx, dw]
