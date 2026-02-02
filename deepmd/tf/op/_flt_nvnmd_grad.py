#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
    tf,
)


@ops.RegisterGradient("FltNvnmd")
def _FltNvnmdGrad(op: tf.Operation, grad: tf.Tensor) -> list[tf.Tensor]:
    dx = op_module.flt_nvnmd(grad)
    return [dx]
