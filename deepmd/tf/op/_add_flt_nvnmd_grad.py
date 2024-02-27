#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
)


@ops.RegisterGradient("AddFltNvnmd")
def _AddFltNvnmdGrad(op, grad):
    dx = op_module.flt_nvnmd(grad)
    dw = dx
    return [dx, dw]
