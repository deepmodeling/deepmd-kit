import numpy as np

from deepmd.env import tf, paddle
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION, GLOBAL_PD_FLOAT_PRECISION

w1 = 0.001
b1 = -0.05

w2 = -0.002
b2 = 0.03

def one_layer(inputs, 
              outputs_size, 
              activation_fn=tf.nn.tanh, 
              precision = GLOBAL_TF_FLOAT_PRECISION, 
              stddev=1.0,
              bavg=0.0,
              name='linear', 
              reuse=None,
              seed=None, 
              use_timestep = False, 
              trainable = True,
              useBN = False):
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        w = tf.get_variable('matrix', 
                            [shape[1], outputs_size], 
                            precision,
                            tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed), 
                            trainable = trainable)
        variable_summaries(w, 'matrix')
        b = tf.get_variable('bias', 
                            [outputs_size], 
                            precision,
                            tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed), 
                            trainable = trainable)
        variable_summaries(b, 'bias')
        hidden = tf.matmul(inputs, w) + b
        if activation_fn != None and use_timestep :
            idt = tf.get_variable('idt',
                                  [outputs_size],
                                  precision,
                                  tf.random_normal_initializer(stddev=0.001, mean = 0.1, seed = seed), 
                                  trainable = trainable)
            variable_summaries(idt, 'idt')
        if activation_fn != None:
            if useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                # return activation_fn(hidden_bn)
            else:
                if use_timestep :
                    return tf.reshape(activation_fn(hidden), [-1, outputs_size]) * idt
                else :
                    return tf.reshape(activation_fn(hidden), [-1, outputs_size])                    
        else:
            if useBN:
                None
                # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden



def embedding_net(xx,
                  network_size,
                  precision,
                  activation_fn = tf.nn.tanh,
                  resnet_dt = False,
                  name_suffix = '',
                  stddev = 1.0,
                  bavg = 0.0,
                  seed = None,
                  trainable = True):
    """
    Parameters
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
    outputs_size = [1] + network_size

    for ii in range(1, len(outputs_size)):
        print("for ii in range(1, len(outputs_size)):")
        w = tf.get_variable('matrix_'+str(ii)+name_suffix, 
                            [outputs_size[ii - 1], outputs_size[ii]], 
                            precision,
                            tf.random_normal_initializer(stddev=stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]), seed = seed), 
                            trainable = trainable)
        variable_summaries(w, 'matrix_'+str(ii)+name_suffix)

        b = tf.get_variable('bias_'+str(ii)+name_suffix, 
                            [1, outputs_size[ii]], 
                            precision,
                            tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed), 
                            trainable = trainable)
        variable_summaries(b, 'bias_'+str(ii)+name_suffix)

        hidden = tf.reshape(activation_fn(tf.matmul(xx, w) + b), [-1, outputs_size[ii]])
        if resnet_dt :
            print("resnet_dt")
            idt = tf.get_variable('idt_'+str(ii)+name_suffix, 
                                  [1, outputs_size[ii]], 
                                  precision,
                                  tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed), 
                                  trainable = trainable)
            variable_summaries(idt, 'idt_'+str(ii)+name_suffix)

        if outputs_size[ii] == outputs_size[ii-1]:
            print("outputs_size[ii] == outputs_size[ii-1]")
            if resnet_dt :
                xx += hidden * idt
            else :
                xx += hidden
        elif outputs_size[ii] == outputs_size[ii-1] * 2: 
            if resnet_dt :
                xx = tf.concat([xx,xx], 1) + hidden * idt
            else :
                xx = tf.concat([xx,xx], 1) + hidden
        else:
            xx = hidden

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
        tf.summary.scalar('mean', mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class OneLayer(paddle.nn.Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 activation_fn=paddle.nn.functional.relu, 
                 precision = GLOBAL_PD_FLOAT_PRECISION, 
                 stddev=1.0,
                 bavg=0.0,
                 name='linear', 
                 seed=None, 
                 use_timestep = False, 
                 trainable = True,
                 useBN = False):
        super(OneLayer, self).__init__(name)
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.use_timestep = use_timestep
        self.useBN = useBN
        self.seed = seed
        paddle.seed(seed)

        self.weight = self.create_parameter(
            shape=[in_features, out_features],
            dtype = precision,
            is_bias= False,
            default_initializer = paddle.fluid.initializer.Constant(w1))
            #default_initializer = paddle.nn.initializer.Normal(std = stddev/np.sqrt(in_features+out_features)))
        self.bias = self.create_parameter(
            shape=[out_features],
            dtype = precision,
            is_bias=True,
            default_initializer = paddle.fluid.initializer.Constant(b1))
            #default_initializer = paddle.nn.initializer.Normal(mean = bavg, std = stddev))
        if self.activation_fn != None and self.use_timestep :
            self.idt = self.create_parameter(
                                  shape=[out_features],
                                  dtype=precision,
                                  default_initializer = paddle.fluid.initializer.Constant(b1))
                                  #default_initializer = paddle.nn.initializer.Normal(mean = 0.1, std = 0.001))

    def forward(self, input):
        hidden = paddle.fluid.layers.matmul(input, self.weight) + self.bias
        if self.activation_fn != None:
            if self.useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                # return activation_fn(hidden_bn)
            else:
                if self.use_timestep :
                    out = paddle.reshape(self.activation_fn(hidden), [-1, self.out_features]) * self.idt
                else :
                    out = paddle.reshape(self.activation_fn(hidden), [-1, self.out_features])
        else:
            if self.useBN:
                None
                # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                out = hidden
        return out



class EmbeddingNet(paddle.nn.Layer):
    """
    Parameters
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
    def __init__(self,
                 network_size, 
                 precision, 
                 activation_fn = paddle.nn.functional.relu, 
                 resnet_dt = False, 
                 seed = None, 
                 trainable = True, 
                 stddev = 1.0, 
                 bavg = 0.0, 
                 name=''):
        super(EmbeddingNet, self).__init__(name)
        self.outputs_size = [1] + network_size
        self.activation_fn = activation_fn
        self.seed = seed
        paddle.seed(seed)

        outputs_size = self.outputs_size
        weight = []
        bias = []
        for ii in range(1, len(outputs_size)):
            weight.append(self.create_parameter(
                                shape = [outputs_size[ii-1], outputs_size[ii]], 
                                dtype = precision,
                                is_bias= False,
                                default_initializer = paddle.fluid.initializer.Constant(w2)))
                                #default_initializer = paddle.nn.initializer.Normal(std = stddev/np.sqrt(outputs_size[ii]+outputs_size[ii-1]))))

            bias.append(self.create_parameter(
                                shape = [1, outputs_size[ii]], 
                                dtype = precision,
                                is_bias= True,
                                default_initializer = paddle.fluid.initializer.Constant(b2)))
                                #default_initializer = paddle.nn.initializer.Normal(mean = bavg, std = stddev)))

        self.weight = paddle.nn.ParameterList(weight)
        self.bias = paddle.nn.ParameterList(bias)

    
    def forward(self, xx):
        outputs_size = self.outputs_size
        for ii in range(1, len(outputs_size)):
            hidden = paddle.reshape(self.activation_fn(paddle.fluid.layers.matmul(xx, self.weight[ii-1]) + self.bias[ii-1]), [-1, outputs_size[ii]])
            if outputs_size[ii] == outputs_size[ii-1] * 2: 
                xx = paddle.concat([xx,xx], axis=1) + hidden
            else:
                xx = hidden

        return xx
