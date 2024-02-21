#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
)


@ops.RegisterGradient("CopyFltNvnmd")
def _CpoyFltNvnmdGrad(op, grad1, grad2):
    dx = op_module.add_flt_nvnmd(grad1, grad2)
    return [dx]
