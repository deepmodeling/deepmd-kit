import os,warnings
import numpy as np

from deepmd.env import tf
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision

def one_layer(inputs, 
              outputs_size, 
              activation_fn=tf.nn.tanh, 
              stddev=1.0,
              bavg=0.0,
              name='linear', 
              reuse=None,
              seed=None, 
              use_timestep = False, 
              useBN = False):
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        w = tf.get_variable('matrix', 
                            [shape[1], outputs_size], 
                            global_tf_float_precision,
                            tf.random_normal_initializer(stddev=stddev/np.sqrt(shape[1]+outputs_size), seed = seed))
        b = tf.get_variable('bias', 
                            [outputs_size], 
                            global_tf_float_precision,
                            tf.random_normal_initializer(stddev=stddev, mean = bavg, seed = seed))
        hidden = tf.matmul(inputs, w) + b
        if activation_fn != None and use_timestep :
            idt = tf.get_variable('idt',
                                  [outputs_size],
                                  global_tf_float_precision,
                                  tf.random_normal_initializer(stddev=0.001, mean = 0.1, seed = seed))
        if activation_fn != None:
            if useBN:
                None
                # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                # return activation_fn(hidden_bn)
            else:
                if use_timestep :
                    return activation_fn(hidden) * idt
                else :
                    return activation_fn(hidden)                    
        else:
            if useBN:
                None
                # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
            else:
                return hidden
