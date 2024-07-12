# SPDX-License-Identifier: LGPL-3.0-or-later
import logging

import numpy as np

from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    op_module,
    tf,
)
from deepmd.tf.nvnmd.utils.config import (
    nvnmd_cfg,
)
from deepmd.tf.nvnmd.utils.weight import (
    get_constant_initializer,
)
from deepmd.tf.utils.network import (
    variable_summaries,
)

log = logging.getLogger(__name__)


def get_sess():
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    return sess


def matmul2_qq(a, b, nbit):
    r"""Quantized matmul operation for 2d tensor.
    a and b is input tensor, nbit represent quantification precision.
    """
    sh_a = a.get_shape().as_list()
    sh_b = b.get_shape().as_list()
    a = tf.reshape(a, [-1, 1, sh_a[1]])
    b = tf.reshape(tf.transpose(b), [1, sh_b[1], sh_b[0]])
    y = a * b
    y = qf(y, nbit)
    y = tf.reduce_sum(y, axis=2)
    return y


def matmul3_qq(a, b, nbit):
    r"""Quantized matmul operation for 3d tensor.
    a and b is input tensor, nbit represent quantification precision.
    """
    sh_a = a.get_shape().as_list()
    sh_b = b.get_shape().as_list()
    a = tf.reshape(a, [-1, sh_a[1], 1, sh_a[2]])
    b = tf.reshape(tf.transpose(b, [0, 2, 1]), [-1, 1, sh_b[2], sh_b[1]])
    y = a * b
    if nbit == -1:
        y = y
    else:
        y = qf(y, nbit)
    y = tf.reduce_sum(y, axis=3)
    return y


def qf(x, nbit):
    r"""Quantize and floor tensor `x` with quantification precision `nbit`."""
    prec = 2**nbit

    y = tf.floor(x * prec) / prec
    y = x + tf.stop_gradient(y - x)
    return y


def qr(x, nbit):
    r"""Quantize and round tensor `x` with quantification precision `nbit`."""
    prec = 2**nbit

    y = tf.round(x * prec) / prec
    y = x + tf.stop_gradient(y - x)
    return y


def tanh4(x):
    with tf.name_scope("tanh4"):
        sign = tf.sign(x)
        xclp = tf.clip_by_value(x, -2, 2)
        xabs = tf.abs(xclp)
        y1 = (1.0 / 16.0) * tf.pow(xabs, 4) + (-1.0 / 4.0) * tf.pow(xabs, 3) + xabs
        y2 = y1 * sign
        return y2


def one_layer_wb(
    shape,
    outputs_size,
    bavg,
    stddev,
    precision,
    trainable,
    initial_variables,
    seed,
    uniform_seed,
    name,
):
    if nvnmd_cfg.restore_fitting_net:
        # initializer
        w_initializer = get_constant_initializer(nvnmd_cfg.weight, "matrix")
        b_initializer = get_constant_initializer(nvnmd_cfg.weight, "bias")
    else:
        w_initializer = tf.random_normal_initializer(
            stddev=stddev / np.sqrt(shape[1] + outputs_size),
            seed=seed if (seed is None or uniform_seed) else seed + 0,
        )
        b_initializer = tf.random_normal_initializer(
            stddev=stddev,
            mean=bavg,
            seed=seed if (seed is None or uniform_seed) else seed + 1,
        )
        if initial_variables is not None:
            w_initializer = tf.constant_initializer(initial_variables[name + "/matrix"])
            b_initializer = tf.constant_initializer(initial_variables[name + "/bias"])
    # variable
    w = tf.get_variable(
        "matrix",
        [shape[1], outputs_size],
        precision,
        w_initializer,
        trainable=trainable,
    )
    variable_summaries(w, "matrix")
    b = tf.get_variable(
        "bias", [outputs_size], precision, b_initializer, trainable=trainable
    )
    variable_summaries(b, "bias")

    return w, b


def one_layer_t(
    shape,
    outputs_size,
    bavg,
    stddev,
    precision,
    trainable,
    initial_variables,
    seed,
    uniform_seed,
    name,
):
    NTAVC = nvnmd_cfg.fitn["NTAVC"]
    if nvnmd_cfg.restore_fitting_net:
        t_initializer = get_constant_initializer(nvnmd_cfg.weight, "tweight")
    else:
        t_initializer = tf.random_normal_initializer(
            stddev=stddev / np.sqrt(NTAVC + outputs_size),
            seed=seed if (seed is None or uniform_seed) else seed + 0,
        )
        if initial_variables is not None:
            t_initializer = tf.constant_initializer(
                initial_variables[name + "/tweight"]
            )
    t = tf.get_variable(
        "tweight",
        [NTAVC, outputs_size],
        precision,
        t_initializer,
        trainable=trainable,
    )
    variable_summaries(t, "matrix")
    return t


def one_layer(
    inputs,
    outputs_size,
    activation_fn=tf.nn.tanh,
    precision=GLOBAL_TF_FLOAT_PRECISION,
    stddev=1.0,
    bavg=0.0,
    name="linear",
    reuse=None,
    seed=None,
    use_timestep=False,
    trainable=True,
    useBN=False,
    uniform_seed=False,
    initial_variables=None,
    mixed_prec=None,
    final_layer=False,
):
    r"""Build one layer with continuous or quantized value.
    Its weight and bias can be initialed with random or constant value.
    """
    # USE FOR NEW FITTINGNET
    is_layer = (nvnmd_cfg.version == 1) and ("layer_0" in name)
    with tf.variable_scope(name, reuse=reuse):
        if is_layer:
            t = one_layer_t(
                None,
                outputs_size,
                bavg,
                stddev,
                precision,
                trainable,
                initial_variables,
                seed,
                uniform_seed,
                name,
            )
            #
            NTAVC = nvnmd_cfg.fitn["NTAVC"]
            nd = inputs.get_shape().as_list()[1] - NTAVC
            inputs2 = tf.slice(inputs, [0, nd], [-1, NTAVC])
            inputs = tf.slice(inputs, [0, 0], [-1, nd])
        # w & b
        shape = inputs.get_shape().as_list()
        w, b = one_layer_wb(
            shape,
            outputs_size,
            bavg,
            stddev,
            precision,
            trainable,
            initial_variables,
            seed,
            uniform_seed,
            name,
        )
        if nvnmd_cfg.quantize_fitting_net:
            NBIT_DATA_FL = nvnmd_cfg.nbit["NBIT_FIT_DATA_FL"]
            NBIT_SHORT_FL = nvnmd_cfg.nbit["NBIT_FIT_SHORT_FL"]
            # w
            with tf.variable_scope("w", reuse=reuse):
                w = op_module.quantize_nvnmd(w, 1, NBIT_DATA_FL, NBIT_DATA_FL, -1)
                w = tf.ensure_shape(w, [shape[1], outputs_size])
            # b
            with tf.variable_scope("b", reuse=reuse):
                b = op_module.quantize_nvnmd(b, 1, NBIT_DATA_FL, NBIT_DATA_FL, -1)
                b = tf.ensure_shape(b, [outputs_size])
            # x
            with tf.variable_scope("x", reuse=reuse):
                x = op_module.quantize_nvnmd(inputs, 1, NBIT_DATA_FL, NBIT_DATA_FL, -1)
                inputs = tf.ensure_shape(x, [None, shape[1]])
            # wx
            # normlize weight mode: 0 all | 1 column
            norm_mode = 0 if final_layer else 1
            wx = op_module.matmul_fitnet_nvnmd(
                inputs, w, NBIT_DATA_FL, NBIT_SHORT_FL, norm_mode
            )

            with tf.variable_scope("wx", reuse=reuse):
                wx = op_module.quantize_nvnmd(wx, 1, NBIT_DATA_FL, NBIT_DATA_FL - 2, -1)
                wx = tf.ensure_shape(wx, [None, outputs_size])

            if is_layer:
                wx2 = tf.matmul(inputs2, t)
                with tf.variable_scope("wx2", reuse=reuse):
                    wx2 = op_module.quantize_nvnmd(
                        wx2, 1, NBIT_DATA_FL, NBIT_DATA_FL, -1
                    )
                    wx2 = tf.ensure_shape(wx2, [None, outputs_size])
                wx = wx + wx2
            # wxb
            wxb = wx + b

            with tf.variable_scope("wxb", reuse=reuse):
                wxb = op_module.quantize_nvnmd(wxb, 1, NBIT_DATA_FL, NBIT_DATA_FL, -1)
                wxb = tf.ensure_shape(wxb, [None, outputs_size])
            # actfun
            if activation_fn is not None:
                # set activation function as tanh4
                y = op_module.tanh4_flt_nvnmd(wxb)
            else:
                y = wxb

            with tf.variable_scope("actfun", reuse=reuse):
                y = op_module.quantize_nvnmd(y, 1, NBIT_DATA_FL, NBIT_DATA_FL, -1)
                y = tf.ensure_shape(y, [None, outputs_size])
        else:
            if is_layer:
                hidden = tf.matmul(inputs, w) + tf.matmul(inputs2, t) + b
            else:
                hidden = tf.matmul(inputs, w) + b
            # set activation function as tanh4
            y = tanh4(hidden) if (activation_fn is not None) else hidden
    # 'reshape' is necessary
    # the next layer needs shape of input tensor to build weight
    y = tf.reshape(y, [-1, outputs_size])
    return y
