#!/usr/bin/env python3

from tensorflow.python.framework import (
    ops,
)

from deepmd.env import (
    op_module,
)


@ops.RegisterGradient("FltNvnmd")
def _FltNvnmdGrad(op, grad):
    dx = op_module.flt_nvnmd(grad)
    return [dx]
