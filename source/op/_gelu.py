#!/usr/bin/env python3
"""
First-order derivatives and second-order derivatives for gelu function.
"""

from tensorflow.python.framework import ops
from deepmd.env import op_module

@ops.RegisterGradient("Gelu")
def _gelu_cc (op, dy) :
    return op_module.gelu_grad(dy, op.inputs[0])     

@ops.RegisterGradient("GeluGrad")
def _gelu_grad_cc (op, dy) :
    return [op_module.gelu_grad(dy, op.inputs[1]), op_module.gelu_grad_grad(dy, op.inputs[0], op.inputs[1])]
