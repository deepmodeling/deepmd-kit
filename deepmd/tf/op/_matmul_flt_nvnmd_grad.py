#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
    tf,
)


@ops.RegisterGradient("MatmulFltNvnmd")
def _MatmulFltNvnmdGrad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    normx = op.get_attr("normx")
    normw = op.get_attr("normw")
    # transpose for 2-dimension and 3-dimension multiplication
    if len(x.shape) == 3:
        x_T = tf.transpose(x, [0, 2, 1])
        w_T = tf.transpose(w, [0, 2, 1])
    else:
        x_T = tf.transpose(x)
        w_T = tf.transpose(w)
    # calcualte
    modex = (normx >> 4) & 15
    modew = (normw >> 4) & 15
    if modex:
        dx = op_module.matmul_flt2fix_nvnmd(grad, w_T, 23)
    else:
        dx = op_module.matmul_flt_nvnmd(grad, w_T, normx, normw)
    if modew:
        dw = op_module.matmul_flt2fix_nvnmd(x_T, grad, 23)
    else:
        dw = op_module.matmul_flt_nvnmd(x_T, grad, 1, normx)
    # add shape for output of matmul_nvnmd
    shx = x.shape.as_list()
    shw = w.shape.as_list()
    shx = [None if (d == -1) else d for d in shx]
    shw = [None if (d == -1) else d for d in shw]
    dx = tf.ensure_shape(dx, shx)
    dw = tf.ensure_shape(dw, shw)
    return [dx, dw]
