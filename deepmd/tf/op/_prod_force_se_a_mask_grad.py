#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gradients for prod force se_a_mask."""

from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_grads_module,
)


@ops.RegisterGradient("ProdForceSeAMask")
def _prod_force_se_a_mask_grad_cc(op, grad):
    net_grad = op_grads_module.prod_force_se_a_mask_grad(
        grad,
        op.inputs[0],
        op.inputs[1],
        op.inputs[2],
        op.inputs[3],
        total_atom_num=op.get_attr("total_atom_num"),
    )
    return [net_grad, None, None, None]
