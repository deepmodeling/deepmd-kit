import numpy as np
from paddle import nn

from deepmd.common import get_precision
from deepmd.env import GLOBAL_PD_FLOAT_PRECISION
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import paddle
from deepmd.env import tf


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
       in deep residual networks. InComputer Vision – ECCV 2016,pages 630–645. Springer
       International Publishing, 2016.
    """
    input_shape = xx.get_shape().as_list()
    outputs_size = [input_shape[1]] + network_size

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


class OneLayer(paddle.nn.Layer):
    def __init__(
        self,
        in_features,
        out_features,
        activation_fn=paddle.nn.functional.tanh,
        precision=GLOBAL_PD_FLOAT_PRECISION,
        stddev=1.0,
        bavg=0.0,
        name="linear",
        seed=None,
        use_timestep=False,
        trainable=True,
        useBN=False,
    ):
        super(OneLayer, self).__init__(name)
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.use_timestep = use_timestep
        self.useBN = useBN
        self.seed = seed
        paddle.seed(seed)

        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype=precision,
            is_bias=False,
            attr=paddle.ParamAttr(trainable=trainable),
            default_initializer=paddle.nn.initializer.Normal(
                std=stddev / np.sqrt(in_features + out_features)
            ),
        )
        # print(bavg, stddev)
        self.bias = self.create_parameter(
            shape=[out_features],
            dtype=precision,
            is_bias=True,
            attr=paddle.ParamAttr(trainable=trainable),
            default_initializer=paddle.nn.initializer.Normal(
                mean=bavg if isinstance(bavg, float) else bavg[0], std=stddev
            ),
        )
        if self.activation_fn is not None and self.use_timestep:
            self.idt = self.create_parameter(
                shape=[out_features],
                dtype=precision,
                attr=paddle.ParamAttr(trainable=trainable),
                default_initializer=paddle.nn.initializer.Normal(mean=0.1, std=0.001),
            )

    def forward(self, input):
        hidden = paddle.matmul(input, self.weight) + self.bias
        if self.activation_fn is not None:
            if self.useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
                # return activation_fn(hidden_bn)
            else:
                if self.use_timestep:
                    hidden = (
                        paddle.reshape(
                            self.activation_fn(hidden), [-1, self.out_features]
                        )
                        * self.idt
                    )
                else:
                    hidden = paddle.reshape(
                        self.activation_fn(hidden), [-1, self.out_features]
                    )
        return hidden


class EmbeddingNet(paddle.nn.Layer):
    """Parameters
    ----------
    xx : Tensor
        Input tensor of shape [-1,1]
    network_size: list of int
        Size of the embedding network. For example [16,32,64]
    precision:
        Precision of network weights. For example, tf.float64
    activation_fn:
        Activation function
    resnet_dt: boolean
        Using time-step in the ResNet construction
    name_suffix: str
        The name suffix append to each variable.
    stddev: float
        Standard deviation of initializing network parameters
    bavg: float
        Mean of network intial bias
    seed: int
        Random seed for initializing network parameters
    trainable: boolean
        If the netowk is trainable
    """

    def __init__(
        self,
        network_size,
        precision,
        activation_fn=paddle.nn.functional.tanh,
        resnet_dt=False,
        stddev=1.0,
        bavg=0.0,
        seed=42,
        trainable=True,
        name="",
    ):
        super().__init__(name)
        self.name = name
        self.outputs_size = [1] + network_size
        self.activation_fn = activation_fn
        self.resnet_dt = resnet_dt
        self.seed = seed
        paddle.seed(seed)

        outputs_size = self.outputs_size
        weight = []
        bias = []
        idt = []
        for ii in range(1, len(outputs_size)):
            weight.append(
                self.create_parameter(
                    shape=[outputs_size[ii - 1], outputs_size[ii]],
                    dtype=precision,
                    is_bias=False,
                    attr=paddle.ParamAttr(trainable=trainable),
                    default_initializer=paddle.nn.initializer.Normal(
                        std=stddev / np.sqrt(outputs_size[ii] + outputs_size[ii - 1])
                    ),
                )
            )
            # print(outputs_size[ii-1], precision, False, trainable, outputs_size[ii]+outputs_size[ii-1])
            # exit()
            bias.append(
                self.create_parameter(
                    shape=[1, outputs_size[ii]],
                    dtype=precision,
                    is_bias=True,
                    attr=paddle.ParamAttr(trainable=trainable),
                    default_initializer=paddle.nn.initializer.Normal(
                        mean=bavg, std=stddev
                    ),
                )
            )
            if resnet_dt:
                idt.append(
                    self.create_parameter(
                        shape=[1, outputs_size[ii]],
                        dtype=precision,
                        attr=paddle.ParamAttr(trainable=trainable),
                        default_initializer=paddle.nn.initializer.Normal(
                            mean=0.1, std=0.001
                        ),
                    )
                )

        self.weight = paddle.nn.ParameterList(weight)
        self.bias = paddle.nn.ParameterList(bias)
        self.idt = paddle.nn.ParameterList(idt)

    def forward(self, xx):
        # outputs_size = self.outputs_size
        # print(self.outputs_size)
        # for ii in range(1, len(outputs_size)):
        #     # if self.activation_fn is not None:
        #     hidden = paddle.reshape(
        #         self.activation_fn(paddle.matmul(xx, self.weight[ii-1]) + self.bias[ii-1]),
        #         [-1, outputs_size[ii]]
        #     )
        #     # print(__file__, 1)
        #     # else:
        #     #     hidden = paddle.reshape(
        #     #         paddle.matmul(xx, self.weight[ii-1]) + self.bias[ii-1],
        #     #         [-1, outputs_size[ii]]
        #     #     )
        #         # print(__file__, 2)

        #     if outputs_size[ii] == outputs_size[ii - 1]:
        #         if self.resnet_dt:
        #             xx += hidden * self.idt[ii]
        #             # print(__file__, 3)
        #         else:
        #             xx += hidden
        #             # print(__file__, 4)
        #     elif outputs_size[ii] == outputs_size[ii-1] * 2:
        #         if self.resnet_dt:
        #             xx = paddle.concat([xx,xx], axis=1) + hidden * self.idt[ii]
        #             # print(__file__, 5)
        #         else:
        #             xx = paddle.concat([xx,xx], axis=1) + hidden
        #             # print(__file__, 6)
        #     else:
        #         # print(__file__, 7)
        #         xx = hidden
        # # exit()

        # return xx
        # if not hasattr(self, "xx1"):
        #     self.xx1 = xx
        # paddle.save(self.xx1.numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_xx1.npy")
        # paddle.save(self.weight[0].numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_weight_0.npy")
        # paddle.save(self.bias[0].numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_bias_0.npy")

        hidden = nn.functional.tanh(
            nn.functional.linear(xx, self.weight[0], self.bias[0])
        ).reshape(
            [-1, 25]
        )  # 1
        xx = hidden  # 7

        # if not hasattr(self, "hidden1"):
        #     self.hidden1 = hidden
        # paddle.save(self.hidden1.numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_hidden1.npy")

        # if not hasattr(self, "xx2"):
        #     self.xx2 = xx
        # paddle.save(self.xx2.numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_xx2.npy")

        hidden = nn.functional.tanh(
            nn.functional.linear(xx, self.weight[1], self.bias[1])
        ).reshape(
            [-1, 50]
        )  # 1
        xx = paddle.concat([xx, xx], axis=1) + hidden  # 6

        # if not hasattr(self, "hidden2"):
        #     self.hidden2 = hidden
        # paddle.save(self.hidden2.numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_hidden2.npy")

        # if not hasattr(self, "xx3"):
        #     self.xx3 = xx
        # paddle.save(self.xx3.numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_xx3.npy")

        hidden = nn.functional.tanh(
            nn.functional.linear(xx, self.weight[2], self.bias[2])
        ).reshape(
            [-1, 100]
        )  # 1
        xx = paddle.concat([xx, xx], axis=1) + hidden  # 6

        # if not hasattr(self, "hidden3"):
        #     self.hidden3 = hidden
        # paddle.save(self.hidden3.numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_hidden3.npy")

        # if not hasattr(self, "xx4"):
        #     self.xx4 = xx
        # paddle.save(self.xx4.numpy(), f"/workspace/hesensen/deepmd_backend/debug_emb/{self.name}_xx4.npy")

        return xx
