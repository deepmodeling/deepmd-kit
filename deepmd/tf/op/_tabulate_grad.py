#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Gradients for tabulate."""

from tensorflow.python.framework import (
    ops,
)

from deepmd.tf.env import (
    op_module,
)

# from deepmd.tf.DescrptSeATabulate import last_layer_size


@ops.RegisterGradient("TabulateFusion")
@ops.RegisterGradient("TabulateFusionSeA")
def _tabulate_fusion_se_a_grad_cc(op, dy):
    dy_dx, dy_df = op_module.tabulate_fusion_se_a_grad(
        op.inputs[0],
        op.inputs[1],
        op.inputs[2],
        op.inputs[3],
        dy,
        op.outputs[0],
    )
    return [None, None, dy_dx, dy_df]


@ops.RegisterGradient("TabulateFusionGrad")
@ops.RegisterGradient("TabulateFusionSeAGrad")
def _tabulate_fusion_se_a_grad_grad_cc(op, dy, dy_):
    dz_dy = op_module.tabulate_fusion_se_a_grad_grad(
        op.inputs[0],
        op.inputs[1],
        op.inputs[2],
        op.inputs[3],
        dy,
        dy_,
        op.inputs[5],
        is_sorted=True,
    )
    return [None, None, None, None, dz_dy, None]


@ops.RegisterGradient("TabulateFusionSeAtten")
def _tabulate_fusion_se_atten_grad_cc(op, dy):
    dy_dx, dy_df, dy_dtwo = op_module.tabulate_fusion_se_atten_grad(
        op.inputs[0],
        op.inputs[1],
        op.inputs[2],
        op.inputs[3],
        op.inputs[4],
        dy,
        op.outputs[0],
        is_sorted=op.get_attr("is_sorted"),
    )
    return [None, None, dy_dx, dy_df, dy_dtwo]


@ops.RegisterGradient("TabulateFusionSeAttenGrad")
def _tabulate_fusion_se_atten_grad_grad_cc(op, dy, dy_, dy_dtwo):
    dz_dy = op_module.tabulate_fusion_se_atten_grad_grad(
        op.inputs[0],
        op.inputs[1],
        op.inputs[2],
        op.inputs[3],
        op.inputs[4],
        dy,
        dy_,
        dy_dtwo,
        op.inputs[6],
        is_sorted=op.get_attr("is_sorted"),
    )
    return [None, None, None, None, None, dz_dy, None]


@ops.RegisterGradient("TabulateFusionSeT")
def _tabulate_fusion_se_t_grad_cc(op, dy):
    dy_dx, dy_df = op_module.tabulate_fusion_se_t_grad(
        op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], dy, op.outputs[0]
    )
    return [None, None, dy_dx, dy_df]


@ops.RegisterGradient("TabulateFusionSeTGrad")
def _tabulate_fusion_se_t_grad_grad_cc(op, dy, dy_):
    dz_dy = op_module.tabulate_fusion_se_t_grad_grad(
        op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], dy, dy_, op.inputs[5]
    )
    return [None, None, None, None, dz_dy, None]


@ops.RegisterGradient("TabulateFusionSeR")
def _tabulate_fusion_se_r_grad_cc(op, dy):
    dy_df = op_module.tabulate_fusion_se_r_grad(
        op.inputs[0], op.inputs[1], op.inputs[2], dy, op.outputs[0]
    )
    return [None, None, dy_df]


@ops.RegisterGradient("TabulateFusionSeRGrad")
def _tabulate_fusion_se_r_grad_grad_cc(op, dy):
    dz_dy = op_module.tabulate_fusion_se_r_grad_grad(
        op.inputs[0], op.inputs[1], op.inputs[2], dy, op.inputs[4]
    )
    return [None, None, None, dz_dy, None]
