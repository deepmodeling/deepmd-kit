#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""First-order derivatives and second-order derivatives for gelu function."""
import tensorflow
from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
)

try:
    gelu = tensorflow.nn.gelu
except AttributeError:

    @ops.RegisterGradient("Gelu")
    def _gelu_cc(op, dy):
        return op_module.gelu_grad_custom(dy, op.inputs[0])

    @ops.RegisterGradient("GeluGrad")
    def _gelu_grad_cc(op, dy):
        return [
            op_module.gelu_grad_custom(dy, op.inputs[1]),
            op_module.gelu_grad_grad_custom(dy, op.inputs[0], op.inputs[1]),
        ]


@ops.RegisterGradient("GeluCustom")
def _gelu_custom_cc(op, dy):
    return op_module.gelu_grad_custom(dy, op.inputs[0])


@ops.RegisterGradient("GeluGradCustom")
def _gelu_grad_custom_cc(op, dy):
    return [
        op_module.gelu_grad_custom(dy, op.inputs[1]),
        op_module.gelu_grad_grad_custom(dy, op.inputs[0], op.inputs[1]),
    ]
