#!/usr/bin/env python3

from tensorflow.python.framework import (
    ops,
)

from deepmd.env import (
    op_module,
)


@ops.RegisterGradient("CopyFltNvnmd")
def _CpoyFltNvnmdGrad(op, grad1, grad2):
    dx = op_module.add_flt_nvnmd(grad1, grad2)
    return [dx]
