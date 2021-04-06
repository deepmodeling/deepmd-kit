import numpy as np

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION

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
            idt = tf.get_variable('idt_'+str(ii)+name_suffix, 
                                  [1, outputs_size[ii]], 
                                  precision,
                                  tf.random_normal_initializer(stddev=0.001, mean = 1.0, seed = seed), 
                                  trainable = trainable)
            variable_summaries(idt, 'idt_'+str(ii)+name_suffix)

        if outputs_size[ii] == outputs_size[ii-1]:
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