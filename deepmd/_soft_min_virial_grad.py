#!/usr/bin/env python3
"""
Gradients for soft min virial.
"""

from tensorflow.python.framework import ops
from deepmd.env import op_grads_module

     
@ops.RegisterGradient("SoftMinVirial")
def _soft_min_virial_grad_cc (op, grad, grad_atom):    
    net_grad =  op_grads_module.soft_min_virial_grad (grad, 
                                                      op.inputs[0], 
                                                      op.inputs[1], 
                                                      op.inputs[2], 
                                                      op.inputs[3], 
                                                      op.inputs[4], 
                                                      n_a_sel = op.get_attr("n_a_sel"),
                                                      n_r_sel = op.get_attr("n_r_sel"))
    return [net_grad, None, None, None, None]
