#!/usr/bin/env python3
"""
Gradients for tabulate.
"""

from tensorflow.python.framework import ops
from deepmd.env import op_module
from deepmd.env import tf
# from deepmd.DescrptSeATabulate import last_layer_size

@ops.RegisterGradient("TabulateFusion")
def _tabulate_fusion_grad_cc (op, dy):    
    dy_dx, dy_df = op_module.tabulate_fusion_grad(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], dy, op.outputs[0])
    return [None, None, dy_dx, dy_df]

@ops.RegisterGradient("TabulateFusionGrad")
def _tabulate_fusion_grad_grad_cc (op, dy, dy_):
    dz_dy = op_module.tabulate_fusion_grad_grad(op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], dy, dy_, op.inputs[5])
    return [None, None, None, None, dz_dy, None]