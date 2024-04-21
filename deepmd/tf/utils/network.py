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


def layer_norm_tf(x, shape, weight=None, bias=None, eps=1e-5):
    """
    Layer normalization implementation in TensorFlow.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor.
    shape : tuple
        The shape of the weight and bias tensors.
    weight : tf.Tensor
        The weight tensor.
    bias : tf.Tensor
        The bias tensor.
    eps : float
        A small value added to prevent division by zero.

    Returns
    -------
    tf.Tensor
        The normalized output tensor.
    """
    # Calculate the mean and variance
    mean = tf.reduce_mean(x, axis=list(range(-len(shape), 0)), keepdims=True)
    variance = tf.reduce_mean(
        tf.square(x - mean), axis=list(range(-len(shape), 0)), keepdims=True
    )

    # Normalize the input
    x_ln = (x - mean) / tf.sqrt(variance + eps)

    # Scale and shift the normalized input
    if weight is not None and bias is not None:
        x_ln = x_ln * weight + bias

    return x_ln


def layernorm(
    inputs,
    outputs_size,
    precision=GLOBAL_TF_FLOAT_PRECISION,
    name="linear",
    scope="",
    reuse=None,
    seed=None,
    uniform_seed=False,
    uni_init=True,
    eps=1e-5,
    trainable=True,
    initial_variables=None,
):
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        if uni_init:
            gamma_initializer = tf.ones_initializer()
            beta_initializer = tf.zeros_initializer()
        else:
            gamma_initializer = tf.random_normal_initializer(
                seed=seed if (seed is None or uniform_seed) else seed + 0
            )
            beta_initializer = tf.random_normal_initializer(
                seed=seed if (seed is None or uniform_seed) else seed + 1
            )
        if initial_variables is not None:
            gamma_initializer = tf.constant_initializer(
                initial_variables[scope + name + "/gamma"]
            )
            beta_initializer = tf.constant_initializer(
                initial_variables[scope + name + "/beta"]
            )
        gamma = tf.get_variable(
            "gamma",
            [outputs_size],
            precision,
            gamma_initializer,
            trainable=trainable,
        )
        variable_summaries(gamma, "gamma")
        beta = tf.get_variable(
            "beta", [outputs_size], precision, beta_initializer, trainable=trainable
        )
        variable_summaries(beta, "beta")

        output = layer_norm_tf(
            inputs,
            (outputs_size,),
            weight=gamma,
            bias=beta,
            eps=eps,
        )
        return output


# class LayerNormCompat:
#     """Implementation of Layer Normalization layer for testing with other backend references.
#
#     Parameters
#     ----------
#     num_in : int
#         The input dimension of the layer.
#     eps : float, optional
#         A small value added to prevent division by zero in calculations.
#     uni_init : bool, optional
#         If initialize the weights to be zeros and ones.
#     precision : str, optional
#         The precision of the layer parameters. Supported options are |PRECISION|
#     """
#
#     def __init__(
#         self,
#         num_in: int,
#         eps: float = 1e-5,
#         uni_init: bool = True,
#         precision: str = "default",
#     ) -> None:
#         self.eps = eps
#         self.uni_init = uni_init
#         self.num_in = num_in
#         self.filter_precision = get_precision(precision)
#         self.layer_norm_variables = None
#
#     def build(
#         self,
#         inputs,
#         input_shape: List[int],
#         reuse=None,
#         suffix="",
#     ):
#         """Build the computational graph for the layer normalization.
#
#         Parameters
#         ----------
#         input_shape
#             The shape of the input tensor.
#         reuse
#             The weights in the networks should be reused when get the variable.
#         suffix
#             Name suffix to identify this layer
#
#         Returns
#         -------
#         normalized_output
#             The computational graph for the normalized output
#         """
#         assert input_shape[-1] == self.num_in
#         name = "layer_norm" + suffix
#         with tf.variable_scope(name, reuse=reuse):
#             gamma = tf.get_variable(
#                 "gamma",
#                 shape=[self.num_in],
#                 initializer=tf.ones_initializer(),
#                 dtype=self.filter_precision,
#                 trainable=True,
#             )
#             beta = tf.get_variable(
#                 "beta",
#                 shape=[self.num_in],
#                 initializer=tf.zeros_initializer(),
#                 dtype=self.filter_precision,
#                 trainable=True,
#             )
#             normalized_output = tf.contrib.layers.layer_norm(
#                 inputs=input,
#                 begin_norm_axis=-1,
#                 begin_params_axis=-1,
#                 epsilon=self.eps,
#                 activation_fn=None,
#                 param_initializers={
#                     "gamma": tf.ones_initializer(),
#                     "beta": tf.zeros_initializer(),
#                 },
#                 trainable=True,
#                 reuse=reuse,
#                 variables_collections=None,
#                 outputs_collections=None,
#                 data_format="NHWC",
#                 name=name,
#             )
#         return normalized_output
#
#     def init_variables(
#         self,
#         graph: tf.Graph,
#         graph_def: tf.GraphDef,
#         suffix="",
#         model_type="original_model",
#     ) -> None:
#         """Init the layer norm variables with the given dict.
#
#         Parameters
#         ----------
#         graph : tf.Graph
#             The input frozen model graph
#         graph_def : tf.GraphDef
#             The input frozen model graph_def
#         suffix
#             Name suffix to identify this layer
#         model_type
#             Indicator of whether this model is a compressed model
#         """
#         self.layer_norm_variables = get_layer_norm_variables_from_graph_def(
#             graph_def, suffix=suffix
#         )
#
#     @classmethod
#     def deserialize(cls, data: dict, suffix: str = ""):
#         """Deserialize the layer from a dict.
#
#         Parameters
#         ----------
#         data : dict
#             The dict to deserialize from.
#         suffix : str, optional
#             The suffix of the scope
#
#         Returns
#         -------
#         LayerNorm
#             The deserialized layer
#         """
#         data = data.copy()
#         check_version_compatibility(data.pop("@version", 1), 1, 1)
#         data_cls = data.pop("@class")
#         assert data_cls == "LayerNorm", f"Invalid class {data_cls}"
#         variables = data.pop("@variables")
#         obj = cls(
#             num_in=variables["w"].shape[0],
#             eps=data.pop("eps"),
#             precision=data.pop("precision"),
#         )
#         obj.layer_norm_variables = {
#             f"layer_norm{suffix}/gamma": variables["w"],
#             f"layer_norm{suffix}/beta": variables["b"],
#         }
#         return obj
#
#     def serialize(self, suffix: str = "") -> dict:
#         """Serialize the layer to a dict.
#
#         Parameters
#         ----------
#         suffix : str, optional
#             The suffix of the scope
#
#         Returns
#         -------
#         dict
#             The serialized layer.
#         """
#         assert self.layer_norm_variables is not None
#         gamma = self.layer_norm_variables[f"layer_norm{suffix}/gamma"]
#         beta = self.layer_norm_variables[f"layer_norm{suffix}/beta"]
#         return {
#             "@class": "LayerNorm",
#             "@version": 1,
#             "eps": self.eps,
#             "precision": self.filter_precision.name,
#             "@variables": {
#                 "w": gamma,
#                 "b": beta,
#             },
#         }


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
