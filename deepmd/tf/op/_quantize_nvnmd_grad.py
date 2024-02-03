#!/usr/bin/env python3

# SPDX-License-Identifier: LGPL-3.0-or-later
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
)


@ops.RegisterGradient("QuantizeNvnmd")
def _QuantizeNvnmdGrad(op, grad):
    isround = op.get_attr("isround")
    nbit1 = op.get_attr("nbit1")
    nbit2 = op.get_attr("nbit2")
    nbit3 = op.get_attr("nbit3")
    dx = op_module.quantize_nvnmd(grad, isround, nbit2, nbit3, nbit1)
    return dx
