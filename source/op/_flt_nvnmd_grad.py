#!/usr/bin/env python3

from tensorflow.python.framework import ops
from deepmd.env import op_module
from deepmd.env import tf 

@ops.RegisterGradient("FltNvnmd")
def _FltNvnmdGrad(op, grad):
    dx = op_module.flt_nvnmd(grad)
    # print(op.outputs[0], dx)
    return [dx]

