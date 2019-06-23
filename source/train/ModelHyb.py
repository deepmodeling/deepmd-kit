import os,warnings
import numpy as np
import tensorflow as tf
from deepmd.common import j_must_have, j_must_have_d, j_have
from deepmd.Model import Model
from deepmd.ModelSeA import ModelSeA
from deepmd.ModelSeR import ModelSeR

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.so" )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.so")

class ModelHyb() :
    def __init__ (self, jdata_a, jdata_r):
        self.model_a = ModelSeA(jdata_a)
        self.model_r = ModelSeR(jdata_r)
            
    def get_rcut(self): 
        return self.model_r.get_rcut()

    def get_ntypes(self): 
        return self.model_r.get_ntypes()

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

        davg, dstd = self.compute_dstats (all_stat_coord, all_stat_box, all_stat_type, all_natoms_vec, all_default_mesh)

        bias_atom_e = data.compute_energy_shift()

        return davg, dstd, bias_atom_e

    def compute_dstats (self,
                        data_coord, 
                        data_box, 
                        data_atype, 
                        natoms_vec,
                        mesh,
                        reuse = None) :    
        davg_a, dstd_a = self.model_a.compute_dstats(data_coord, data_box, data_atype, natoms_vec, mesh, reuse)
        davg_r, dstd_r = self.model_r.compute_dstats(data_coord, data_box, data_atype, natoms_vec, mesh, reuse)
        return [davg_a, davg_r], [dstd_a, dstd_r]

    def build_interaction(self, 
                          coord, 
                          atype, 
                          natoms, 
                          box, 
                          mesh, 
                          fparam, 
                          davg,
                          dstd,
                          bias_atom_e, 
                          suffix = '',
                          reuse_attr = None,
                          reuse_weights = None) :
        e_a, f_a, v_a, ae_a, av_a \
            = self.model_a.build_interaction(coord, atype, natoms, box, mesh, fparam, davg[0], dstd[0], bias_atom_e, suffix = '_a'+suffix, reuse_attr = reuse_attr, reuse_weights = reuse_weights)
        e_r, f_r, v_r, ae_r, av_r \
            = self.model_r.build_interaction(coord, atype, natoms, box, mesh, fparam, davg[1], dstd[1], bias_atom_e, suffix = '_r'+suffix, reuse_attr = reuse_attr, reuse_weights = reuse_weights)
        with tf.variable_scope('model_attr' + suffix, reuse = reuse_attr) :
            t_rcut = tf.constant(self.model_r.get_rcut(), 
                                 name = 'rcut', 
                                 dtype = global_tf_float_precision)
            t_ntypes = tf.constant(self.model_r.get_ntypes(), 
                                   name = 'ntypes', 
                                   dtype = tf.int32)
            t_dfparam = tf.constant(self.model_r.get_numb_fparam(), 
                                    name = 'dfparam', 
                                    dtype = tf.int32)
            t_tmap = tf.constant(' '.join(self.model_r.get_type_map()), 
                                 name = 'tmap', 
                                 dtype = tf.string)
            
        energy = tf.add(e_a, e_r, name = 'o_energy'+suffix)
        force  = tf.add(f_a, f_r, name = 'o_force' +suffix)
        virial = tf.add(v_a, v_r, name = 'o_virial'+suffix)
        ae = tf.add(ae_a, ae_r, name = 'o_atom_energy'+suffix)
        av = tf.add(av_a, av_r, name = 'o_atom_virial'+suffix)
        return energy, force, virial, ae, av
