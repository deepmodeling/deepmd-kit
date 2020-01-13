#!/usr/bin/env python3
"""
Gradients for prod virial.
"""

from tensorflow.python.framework import ops
from deepmd.env import op_grads_module
     
@ops.RegisterGradient("ProdVirialSeR")
def _prod_virial_se_a_grad_cc (op, grad, grad_atom):    
    net_grad =  op_grads_module.prod_virial_se_r_grad (grad, 
                                                       op.inputs[0], 
                                                       op.inputs[1], 
                                                       op.inputs[2], 
                                                       op.inputs[3], 
                                                       op.inputs[4])
    return [net_grad, None, None, None, None]
