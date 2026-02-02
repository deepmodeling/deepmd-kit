#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
    tf,
)


@ops.RegisterGradient("CopyFltNvnmd")
def _CpoyFltNvnmdGrad(
    op: tf.Operation, grad1: tf.Tensor, grad2: tf.Tensor
) -> list[tf.Tensor]:
    dx = op_module.add_flt_nvnmd(grad1, grad2)
    return [dx]
