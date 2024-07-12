#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
    tf,
)


@ops.RegisterGradient("DotmulFltNvnmd")
def _DotmulFltNvnmdGrad(op, grad):
    x = op.inputs[0]
    w = op.inputs[1]
    # calcualte
    dx = op_module.mul_flt_nvnmd(grad, w)
    dw = op_module.mul_flt_nvnmd(grad, x)
    # add shape for output of matmul_nvnmd
    shx = x.shape.as_list()
    shw = w.shape.as_list()
    shx = [None if (d == -1) else d for d in shx]
    shw = [None if (d == -1) else d for d in shw]
    dx = tf.ensure_shape(dx, shx)
    dw = tf.ensure_shape(dw, shw)
    return [dx, dw]
