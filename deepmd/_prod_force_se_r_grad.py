#!/usr/bin/env python3
"""
Gradients for prod force.
"""

from tensorflow.python.framework import ops
from deepmd.env import op_grads_module
     
@ops.RegisterGradient("ProdForceSeR")
def _prod_force_se_a_grad_cc (op, grad):    
    net_grad =  op_grads_module.prod_force_se_r_grad (grad, 
                                                      op.inputs[0], 
                                                      op.inputs[1], 
                                                      op.inputs[2], 
                                                      op.inputs[3])
    return [net_grad, None, None, None]
