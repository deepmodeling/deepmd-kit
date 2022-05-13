
import numpy as np

from deepmd.env import tf
from deepmd.env import GLOBAL_TF_FLOAT_PRECISION
from deepmd.env import op_module

from deepmd.nvnmd.utils.config import nvnmd_cfg
from deepmd.nvnmd.utils.weight import get_constant_initializer
from deepmd.utils.network import variable_summaries

def get_sess():
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)
    return sess

def matmul2_qq(a, b, nbit):
    sh_a = a.get_shape().as_list()
    sh_b = b.get_shape().as_list()
    a = tf.reshape(a, [-1, 1, sh_a[1]])
    b = tf.reshape(tf.transpose(b), [1, sh_b[1], sh_b[0]])
    y = a * b
    y = qf(y, nbit)
    y = tf.reduce_sum(y, axis=2)
    return y

def matmul3_qq(a, b, nbit):
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
    prec = 2**nbit
    
    y = tf.floor(x * prec) / prec
    y = x + tf.stop_gradient(y - x)
    return y

def qr(x, nbit):
    prec = 2**nbit
    
    y = tf.round(x * prec) / prec
    y = x + tf.stop_gradient(y - x)
    return y

# fitting_net
def tanh2(x,nbit=-1,nbit2=-1):
    y = op_module.tanh2_nvnmd(x, 0, nbit, nbit2, -1)
    # y = tf.tanh(x)
    # x1 = tf.clip_by_value(x, -2, 2)
    # xa = tf.abs(x1)
    # x2 = x1 * x1
    # x3 = x2 * x1 
    # a = 1/16
    # b = -1/4
    # y = a*x3*xa + b*x3 + x1
    return y

def tanh4(x, nbit=-1, nbit2=-1):
    y = op_module.tanh4_nvnmd(x, 0, nbit, nbit2, -1)
    # y = tf.tanh(x)
    return y 


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
    name):
    
    if nvnmd_cfg.restore_fitting_net:
        # initializer
        w_initializer = get_constant_initializer(nvnmd_cfg.weight, 'matrix')
        b_initializer = get_constant_initializer(nvnmd_cfg.weight, 'bias')
    else:
        w_initializer  = tf.random_normal_initializer(
                            stddev=stddev / np.sqrt(shape[1] + outputs_size),
                            seed=seed if (seed is None or uniform_seed) else seed + 0)
        b_initializer  = tf.random_normal_initializer(
                            stddev=stddev,
                            mean=bavg,
                            seed=seed if (seed is None or uniform_seed) else seed + 1)
        if initial_variables is not None:
            w_initializer = tf.constant_initializer(initial_variables[name + '/matrix'])
            b_initializer = tf.constant_initializer(initial_variables[name + '/bias'])
    # variable
    w = tf.get_variable('matrix', 
                        [shape[1], outputs_size], 
                        precision,
                        w_initializer, 
                        trainable = trainable)
    variable_summaries(w, 'matrix')
    b = tf.get_variable('bias', 
                        [outputs_size], 
                        precision,
                        b_initializer, 
                        trainable = trainable)
    variable_summaries(b, 'bias')

    return w, b

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
              useBN = False, 
              uniform_seed = False,
              initial_variables = None):
    if activation_fn != None: activation_fn = tanh4 
    with tf.variable_scope(name, reuse=reuse):
        shape = inputs.get_shape().as_list()
        w, b = one_layer_wb(shape, outputs_size, bavg, stddev, precision, trainable, initial_variables, seed, uniform_seed, name)
        if nvnmd_cfg.quantize_fitting_net:
            NBIT_DATA_FL = nvnmd_cfg.nbit['NBIT_DATA_FL']
            NBIT_WEIGHT_FL = nvnmd_cfg.nbit['NBIT_WEIGHT_FL']
            #
            inputs = qf(inputs, NBIT_DATA_FL)
            w = qr(w, NBIT_WEIGHT_FL)
            with tf.variable_scope('wx', reuse=reuse):
                wx = op_module.matmul_nvnmd(inputs, w, 0, NBIT_DATA_FL, NBIT_DATA_FL, -1)
            #
            b = qr(b, NBIT_DATA_FL)
            with tf.variable_scope('wxb', reuse=reuse):
                hidden = wx + b
            #
            with tf.variable_scope('actfun', reuse=reuse):
                if activation_fn != None:
                    y = activation_fn(hidden, NBIT_DATA_FL, NBIT_DATA_FL)
                else:
                    y = hidden + 0
        else:
            hidden = tf.matmul(inputs, w) + b
            y = activation_fn(hidden, -1, -1) if activation_fn != None else hidden 
    return y


