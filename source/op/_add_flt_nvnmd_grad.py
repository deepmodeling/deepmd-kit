#!/usr/bin/env python3

from tensorflow.python.framework import (
    ops,
)

from deepmd.env import (
    op_module,
)


@ops.RegisterGradient("AddFltNvnmd")
def _AddFltNvnmdGrad(op, grad):
    dx = op_module.flt_nvnmd(grad)
    dw = dx
    return [dx, dw]
