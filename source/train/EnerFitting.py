import os,warnings
import numpy as np
import tensorflow as tf

from deepmd.common import j_must_have, j_must_have_d, j_have

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

class EnerFitting ():
    def __init__ (self, jdata, descrpt):
        # model param
        self.ntypes = descrpt.get_ntypes()
        self.dim_descrpt = descrpt.get_dim_out()
        # fparam
        self.numb_fparam = 0
        if j_have(jdata, 'numb_fparam') :
            self.numb_fparam = jdata['numb_fparam']
        # network size
        self.n_neuron = j_must_have_d (jdata, 'fitting_neuron', ['n_neuron'])
        self.resnet_dt = True
        if j_have(jdata, 'resnet_dt') :
            warnings.warn("the key \"%s\" is deprecated, please use \"%s\" instead" % ('resnet_dt','fitting_resnet_dt'))
            self.resnet_dt = jdata['resnet_dt']
        if j_have(jdata, 'fitting_resnet_dt') :
            self.resnet_dt = jdata['fitting_resnet_dt']
        
        self.seed = None
        if j_have (jdata, 'seed') :
            self.seed = jdata['seed']
        self.useBN = False

    def get_numb_fparam(self) :
        return self.numb_fparam

    def build (self, 
               inputs,
               fparam,
               natoms,
               bias_atom_e = None,
               reuse = None,
               suffix = '') :
        with tf.variable_scope('model_attr' + suffix, reuse = reuse) :
            t_dfparam = tf.constant(self.numb_fparam, 
                                    name = 'dfparam', 
                                    dtype = tf.int32)
        start_index = 0
        inputs = tf.reshape(inputs, [-1, self.dim_descrpt * natoms[0]])
        shape = inputs.get_shape().as_list()

        if bias_atom_e is not None :
            assert(len(bias_atom_e) == self.ntypes)

        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.dim_descrpt],
                                 [-1, natoms[2+type_i]* self.dim_descrpt] )
            inputs_i = tf.reshape(inputs_i, [-1, self.dim_descrpt])
            start_index += natoms[2+type_i]
            if bias_atom_e is None :
                type_bias_ae = 0.0
            else :
                type_bias_ae = bias_atom_e[type_i]

            layer = inputs_i
            if self.numb_fparam > 0 :
                ext_fparam = tf.reshape(fparam, [-1, self.numb_fparam])
                ext_fparam = tf.tile(ext_fparam, [1, natoms[0]])
                ext_fparam = tf.reshape(ext_fparam, [-1, self.numb_fparam])
                layer = tf.concat([layer, ext_fparam], axis = 1)
            for ii in range(0,len(self.n_neuron)) :
                if ii >= 1 and self.n_neuron[ii] == self.n_neuron[ii-1] :
                    layer+= self._one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt)
                else :
                    layer = self._one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            final_layer = self._one_layer(layer, 1, activation_fn = None, bavg = type_bias_ae, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            final_layer = tf.reshape(final_layer, [-1, natoms[2+type_i]])

            # concat the results
            if type_i == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)

        return tf.reshape(outs, [-1])
        

    def _one_layer(self, 
                   inputs, 
                   outputs_size, 
                   activation_fn=tf.nn.tanh, 
                   stddev=1.0,
                   bavg=0.0,
                   name='linear', 
                   reuse=None,
                   seed=None, 
                   use_timestep = False):
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
                if self.useBN:
                    None
                    # hidden_bn = self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)   
                    # return activation_fn(hidden_bn)
                else:
                    if use_timestep :
                        return activation_fn(hidden) * idt
                    else :
                        return activation_fn(hidden)                    
            else:
                if self.useBN:
                    None
                    # return self._batch_norm(hidden, name=name+'_normalization', reuse=reuse)
                else:
                    return hidden
