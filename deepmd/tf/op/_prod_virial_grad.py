#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gradients for prod virial."""

from typing import (
    Any,
)

from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_grads_module,
)


@ops.RegisterGradient("ProdVirial")
def _prod_virial_grad_cc(op: Any, grad: Any, grad_atom: Any) -> list[Any]:
    net_grad = op_grads_module.prod_virial_grad(
        grad,
        op.inputs[0],
        op.inputs[1],
        op.inputs[2],
        op.inputs[3],
        op.inputs[4],
        op.inputs[5],
        n_a_sel=op.get_attr("n_a_sel"),
        n_r_sel=op.get_attr("n_r_sel"),
    )
    return [net_grad, None, None, None, None, None]
