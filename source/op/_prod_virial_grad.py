#!/usr/bin/env python3
"""
Gradients for prod virial.
"""

import os
import platform
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

if platform.system() == "Windows":
    ext = "dll"
elif platform.system() == "Darwin":
    ext = "dylib"
else:
    ext = "so"
module_path = os.path.dirname(os.path.realpath(__file__))
module_file = os.path.join(module_path, 'libop_grads.{}'.format(ext))
assert (os.path.isfile(module_file)), 'module op_grads does not exist'
op_grads_module = tf.load_op_library(module_file)
     
@ops.RegisterGradient("ProdVirial")
def _prod_virial_grad_cc (op, grad, grad_atom):    
    net_grad =  op_grads_module.prod_virial_grad (grad, 
                                                  op.inputs[0], 
                                                  op.inputs[1], 
                                                  op.inputs[2], 
                                                  op.inputs[3], 
                                                  op.inputs[4], 
                                                  op.inputs[5], 
                                                  n_a_sel = op.get_attr("n_a_sel"),
                                                  n_r_sel = op.get_attr("n_r_sel"))
    return [net_grad, None, None, None, None, None]
