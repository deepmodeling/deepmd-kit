#!/usr/bin/env python3
"""Gradients for tabulate."""

from tensorflow.python.framework import (
    ops,
)

from deepmd.env import (
    op_module,
)
import tensorflow.compat.v1 as tf

# from deepmd.DescrptSeATabulate import last_layer_size


@ops.RegisterGradient("TabulateFusion")
@ops.RegisterGradient("TabulateFusionSeA")
def _tabulate_fusion_se_a_grad_cc(op, dy):
    dy_dx, dy_df = op_module.tabulate_fusion_se_a_grad(
        op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], dy, op.outputs[0]
    )
    return [None, None, dy_dx, dy_df]


@ops.RegisterGradient("TabulateFusionGrad")
@ops.RegisterGradient("TabulateFusionSeAGrad")
def _tabulate_fusion_se_a_grad_grad_cc(op, dy, dy_):
    dz_dy = op_module.tabulate_fusion_se_a_grad_grad(
        op.inputs[0], op.inputs[1], op.inputs[2], op.inputs[3], dy, dy_, op.inputs[5]
    )
    return [None, None, None, None, dz_dy, None]


@ops.RegisterGradient("TabulateFusionSeAtten")
def _tabulate_fusion_se_atten_grad_cc(op, dy):
    dy_dx, dy_df, dy_dtwo = op_module.tabulate_fusion_se_atten_grad(
        op.inputs[0],  # table
        op.inputs[1],  # table_info
        op.inputs[2],  # em_x
        op.inputs[3],  # em
        op.inputs[4],  # two_embed
        dy,            # dy
        op.outputs[0]  # descriptor
    )
    return [None,      # table
            None,      # table_info
            dy_dx,     # em_x
            dy_df,     # em
            dy_dtwo]   # two_embed

@ops.RegisterGradient("TabulateFusionSeAttenGrad")
def _tabulate_fusion_se_atten_grad_grad_cc(op, dy, dy_, dy_dtwo):
    dz_dy = op_module.tabulate_fusion_se_atten_grad_grad(
        op.inputs[0],  # table
        op.inputs[1],  # table_info
        op.inputs[2],  # em_x
        op.inputs[3],  # em
        dy_dtwo,       # two_embed
        dy,            # dz_dy_dem_x
        dy_,           # dz_dy_dem
        op.inputs[6]   # descriptor
    )
    return [None,      # table
            None,      # table_info
            None,      # em_x
            None,      # em
            None,      # two_embed
            dz_dy,     # dy
            None]      # descriptor


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
