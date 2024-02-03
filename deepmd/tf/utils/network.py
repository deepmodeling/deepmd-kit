# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np

from deepmd.tf.common import (
    get_precision,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    tf,
)


def one_layer_rand_seed_shift():
    return 3


def one_layer(
    inputs,
    outputs_size,
    activation_fn=tf.nn.tanh,
    precision=GLOBAL_TF_FLOAT_PRECISION,
    stddev=1.0,
    bavg=0.0,
    name="linear",
    scope="",
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
    # For good accuracy, the last layer of the fitting network uses a higher precision neuron network.
    if mixed_prec is not None and final_layer:
        inputs = tf.cast(inputs, get_precision(mixed_prec["output_prec"]))
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
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
            w_initializer = tf.constant_initializer(
                initial_variables[scope + name + "/matrix"]
            )
            b_initializer = tf.constant_initializer(
                initial_variables[scope + name + "/bias"]
            )
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

        if mixed_prec is not None and not final_layer:
            inputs = tf.cast(inputs, get_precision(mixed_prec["compute_prec"]))
            w = tf.cast(w, get_precision(mixed_prec["compute_prec"]))
            b = tf.cast(b, get_precision(mixed_prec["compute_prec"]))

        hidden = tf.nn.bias_add(tf.matmul(inputs, w), b)
        if activation_fn is not None and use_timestep:
            idt_initializer = tf.random_normal_initializer(
                stddev=0.001,
                mean=0.1,
                seed=seed if (seed is None or uniform_seed) else seed + 2,
            )
            if initial_variables is not None:
                idt_initializer = tf.constant_initializer(
                    initial_variables[scope + name + "/idt"]
                )
            idt = tf.get_variable(
                "idt", [outputs_size], precision, idt_initializer, trainable=trainable
            )
            variable_summaries(idt, "idt")
        if activation_fn is not None:
            if useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
                # return activation_fn(hidden_bn)
            else:
                if use_timestep:
                    if mixed_prec is not None and not final_layer:
                        idt = tf.cast(idt, get_precision(mixed_prec["compute_prec"]))
                    hidden = tf.reshape(activation_fn(hidden), [-1, outputs_size]) * idt
                else:
                    hidden = tf.reshape(activation_fn(hidden), [-1, outputs_size])

        if mixed_prec is not None:
            hidden = tf.cast(hidden, get_precision(mixed_prec["output_prec"]))
        return hidden


def embedding_net_rand_seed_shift(network_size):
    shift = 3 * (len(network_size) + 1)
    return shift


def embedding_net(
    xx,
    network_size,
    precision,
    activation_fn=tf.nn.tanh,
    resnet_dt=False,
    name_suffix="",
    stddev=1.0,
    bavg=0.0,
    seed=None,
    trainable=True,
    uniform_seed=False,
    initial_variables=None,
    mixed_prec=None,
):
    r"""The embedding network.

    The embedding network function :math:`\mathcal{N}` is constructed by is the
    composition of multiple layers :math:`\mathcal{L}^{(i)}`:

    .. math::
        \mathcal{N} = \mathcal{L}^{(n)} \circ \mathcal{L}^{(n-1)}
        \circ \cdots \circ \mathcal{L}^{(1)}

    A layer :math:`\mathcal{L}` is given by one of the following forms,
    depending on the number of nodes: [1]_

    .. math::
        \mathbf{y}=\mathcal{L}(\mathbf{x};\mathbf{w},\mathbf{b})=
        \begin{cases}
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}) + \mathbf{x}, & N_2=N_1 \\
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}) + (\mathbf{x}, \mathbf{x}), & N_2 = 2N_1\\
            \boldsymbol{\phi}(\mathbf{x}^T\mathbf{w}+\mathbf{b}), & \text{otherwise} \\
        \end{cases}

    where :math:`\mathbf{x} \in \mathbb{R}^{N_1}` is the input vector and :math:`\mathbf{y} \in \mathbb{R}^{N_2}`
    is the output vector. :math:`\mathbf{w} \in \mathbb{R}^{N_1 \times N_2}` and
    :math:`\mathbf{b} \in \mathbb{R}^{N_2}` are weights and biases, respectively,
    both of which are trainable if `trainable` is `True`. :math:`\boldsymbol{\phi}`
    is the activation function.

    Parameters
    ----------
    xx : Tensor
        Input tensor :math:`\mathbf{x}` of shape [-1,1]
    network_size : list of int
        Size of the embedding network. For example [16,32,64]
    precision:
        Precision of network weights. For example, tf.float64
    activation_fn:
        Activation function :math:`\boldsymbol{\phi}`
    resnet_dt : boolean
        Using time-step in the ResNet construction
    name_suffix : str
        The name suffix append to each variable.
    stddev : float
        Standard deviation of initializing network parameters
    bavg : float
        Mean of network intial bias
    seed : int
        Random seed for initializing network parameters
    trainable : boolean
        If the network is trainable
    uniform_seed : boolean
        Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    initial_variables : dict
        The input dict which stores the embedding net variables
    mixed_prec
        The input dict which stores the mixed precision setting for the embedding net

    References
    ----------
    .. [1] Kaiming  He,  Xiangyu  Zhang,  Shaoqing  Ren,  and  Jian  Sun. Identitymappings
       in deep residual networks. InComputer Vision - ECCV 2016,pages 630-645. Springer
       International Publishing, 2016.
    """
    input_shape = xx.get_shape().as_list()
    outputs_size = [input_shape[1], *network_size]

    for ii in range(1, len(outputs_size)):
        w_initializer = tf.random_normal_initializer(
            stddev=stddev / np.sqrt(outputs_size[ii] + outputs_size[ii - 1]),
            seed=seed if (seed is None or uniform_seed) else seed + ii * 3 + 0,
        )
        b_initializer = tf.random_normal_initializer(
            stddev=stddev,
            mean=bavg,
            seed=seed if (seed is None or uniform_seed) else seed + 3 * ii + 1,
        )
        if initial_variables is not None:
            scope = tf.get_variable_scope().name
            w_initializer = tf.constant_initializer(
                initial_variables[scope + "/matrix_" + str(ii) + name_suffix]
            )
            b_initializer = tf.constant_initializer(
                initial_variables[scope + "/bias_" + str(ii) + name_suffix]
            )
        w = tf.get_variable(
            "matrix_" + str(ii) + name_suffix,
            [outputs_size[ii - 1], outputs_size[ii]],
            precision,
            w_initializer,
            trainable=trainable,
        )
        variable_summaries(w, "matrix_" + str(ii) + name_suffix)

        b = tf.get_variable(
            "bias_" + str(ii) + name_suffix,
            [outputs_size[ii]],
            precision,
            b_initializer,
            trainable=trainable,
        )
        variable_summaries(b, "bias_" + str(ii) + name_suffix)

        if mixed_prec is not None:
            xx = tf.cast(xx, get_precision(mixed_prec["compute_prec"]))
            w = tf.cast(w, get_precision(mixed_prec["compute_prec"]))
            b = tf.cast(b, get_precision(mixed_prec["compute_prec"]))
        if activation_fn is not None:
            hidden = tf.reshape(
                activation_fn(tf.nn.bias_add(tf.matmul(xx, w), b)),
                [-1, outputs_size[ii]],
            )
        else:
            hidden = tf.reshape(
                tf.nn.bias_add(tf.matmul(xx, w), b), [-1, outputs_size[ii]]
            )
        if resnet_dt:
            idt_initializer = tf.random_normal_initializer(
                stddev=0.001,
                mean=1.0,
                seed=seed if (seed is None or uniform_seed) else seed + 3 * ii + 2,
            )
            if initial_variables is not None:
                scope = tf.get_variable_scope().name
                idt_initializer = tf.constant_initializer(
                    initial_variables[scope + "/idt_" + str(ii) + name_suffix]
                )
            idt = tf.get_variable(
                "idt_" + str(ii) + name_suffix,
                [1, outputs_size[ii]],
                precision,
                idt_initializer,
                trainable=trainable,
            )
            variable_summaries(idt, "idt_" + str(ii) + name_suffix)
            if mixed_prec is not None:
                idt = tf.cast(idt, get_precision(mixed_prec["compute_prec"]))

        if outputs_size[ii] == outputs_size[ii - 1]:
            if resnet_dt:
                xx += hidden * idt
            else:
                xx += hidden
        elif outputs_size[ii] == outputs_size[ii - 1] * 2:
            if resnet_dt:
                xx = tf.concat([xx, xx], 1) + hidden * idt
            else:
                xx = tf.concat([xx, xx], 1) + hidden
        else:
            xx = hidden
    if mixed_prec is not None:
        xx = tf.cast(xx, get_precision(mixed_prec["output_prec"]))
    return xx


def variable_summaries(var: tf.Variable, name: str):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    Parameters
    ----------
    var : tf.Variable
        [description]
    name : str
        variable name
    """
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)

        with tf.name_scope("stddev"):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)
