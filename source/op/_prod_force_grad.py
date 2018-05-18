#!/usr/bin/env python3
"""
Gradients for prod force.
"""

import os
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

force_module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (force_module_path  + "libprod_force_grad.so" )), "force module grad does not exist"
prod_force_grad_module = tf.load_op_library(force_module_path + 'libprod_force_grad.so')
     
@ops.RegisterGradient("ProdForce")
def _prod_force_grad_cc (op, grad):    
    net_grad =  prod_force_grad_module.prod_force_grad (grad, 
                                                        op.inputs[0], 
                                                        op.inputs[1], 
                                                        op.inputs[2], 
                                                        op.inputs[3], 
                                                        op.inputs[4], 
                                                        n_a_sel = op.get_attr("n_a_sel"),
                                                        n_r_sel = op.get_attr("n_r_sel"))
    return [net_grad, None, None, None, None]
