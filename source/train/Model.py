import numpy as np
import tensorflow as tf

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

class Model() :
    def data_stat(self, data):
        all_stat_coord = []
        all_stat_box = []
        all_stat_type = []
        all_natoms_vec = []
        all_default_mesh = []
        for ii in range(data.get_nsystems()) :
            stat_prop_c, \
                stat_energy, stat_force, stat_virial, start_atom_ener, \
                stat_coord, stat_box, stat_type, stat_fparam, natoms_vec, default_mesh \
                = data.get_batch (sys_idx = ii)
            natoms_vec = natoms_vec.astype(np.int32)            
            all_stat_coord.append(stat_coord)
            all_stat_box.append(stat_box)
            all_stat_type.append(stat_type)
            all_natoms_vec.append(natoms_vec)
            all_default_mesh.append(default_mesh)

        if not self.coord_norm :
            davg, dstd = self.no_norm_dstats ()
        else :
            davg, dstd = self.compute_dstats (all_stat_coord, all_stat_box, all_stat_type, all_natoms_vec, all_default_mesh)
        # if self.run_opt.is_chief:
        #     np.savetxt ("stat.avg.out", davg.T)
        #     np.savetxt ("stat.std.out", dstd.T)

        bias_atom_e = data.compute_energy_shift()
        # self._message("computed energy bias")

        return davg, dstd, bias_atom_e
    
    def one_layer(self, 
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
    
