#!/usr/bin/env python3
"""
First-order derivatives and second-order derivatives for gelu function.
"""

from tensorflow.python.framework import ops
from deepmd.env import op_module

@ops.RegisterGradient("Gelu")
def gelu_cc (op, dy) :
    return op_module.gelu_grad(dy, op.inputs[0])     

@ops.RegisterGradient("GeluGrad")
def gelu_grad_cc (op, dy) :
    return [None, op_module.gelu_grad_grad(dy, op.inputs[0], op.inputs[1])]
