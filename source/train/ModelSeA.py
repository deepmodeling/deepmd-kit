import os,warnings
import numpy as np
import tensorflow as tf
from deepmd.common import j_must_have, j_must_have_d, j_have
from deepmd.Model import Model
from deepmd.DescrptSeA import DescrptSeA

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.so" )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.so")

class ModelSeA (Model):
    def __init__ (self, jdata):
        self.descrpt = DescrptSeA(jdata)
        self.ntypes = self.descrpt.get_ntypes()
        # fparam
        self.numb_fparam = 0
        if j_have(jdata, 'numb_fparam') :
            self.numb_fparam = jdata['numb_fparam']
	# type_map
        self.type_map = []
        if j_have(jdata, 'type_map') :
            self.numb_fparam = jdata['type_map']
        # network size
        self.n_neuron = j_must_have_d (jdata, 'fitting_neuron', ['n_neuron'])
        self.resnet_dt = True
        if j_have(jdata, 'resnet_dt') :
            warnings.warn("the key \"%s\" is deprecated, please use \"%s\" instead" % ('resnet_dt','fitting_resnet_dt'))
            self.resnet_dt = jdata['resnet_dt']
        if j_have(jdata, 'fitting_resnet_dt') :
            self.resnet_dt = jdata['fitting_resnet_dt']
        if j_have(jdata, 'type_fitting_net') :
            self.type_fitting_net = jdata['type_fitting_net']
        else :
            self.type_fitting_net = False            

        # short-range tab
        if 'use_srtab' in jdata :
            self.srtab = TabInter(jdata['use_srtab'])
            self.smin_alpha = j_must_have(jdata, 'smin_alpha')
            self.sw_rmin = j_must_have(jdata, 'sw_rmin')
            self.sw_rmax = j_must_have(jdata, 'sw_rmax')
        else :
            self.srtab = None

        self.seed = None
        if j_have (jdata, 'seed') :
            self.seed = jdata['seed']
        self.useBN = False


    def get_rcut (self) :
        return self.descrpt.get_rcut()

    def get_ntypes (self) :
        return self.ntypes

    def get_numb_fparam (self) :
        return self.numb_fparam

    def get_type_map (self) :
        return self.type_map

    def compute_dstats (self,
                        data_coord, 
                        data_box, 
                        data_atype, 
                        natoms_vec,
                        mesh,
                        reuse = None) :        
        return self.descrpt.compute_dstats(data_coord, data_box, data_atype, natoms_vec, mesh, reuse)
    

    def build_interaction (self, 
                           coord_, 
                           atype_,
                           natoms,
                           box, 
                           mesh,
                           fparam,
                           davg = None, 
                           dstd = None,
                           bias_atom_e = None,
                           suffix = '', 
                           reuse_attr = None,
                           reuse_weights = None):

        with tf.variable_scope('model_attr' + suffix, reuse = reuse_attr) :
            t_dfparam = tf.constant(self.numb_fparam, 
                                    name = 'dfparam', 
                                    dtype = tf.int32)
            t_tmap = tf.constant(' '.join(self.type_map), 
                                 name = 'tmap', 
                                 dtype = tf.string)

            if self.srtab is not None :
                tab_info, tab_data = self.srtab.get()
                self.tab_info = tf.get_variable('t_tab_info',
                                                tab_info.shape,
                                                dtype = tf.float64,
                                                trainable = False,
                                                initializer = tf.constant_initializer(tab_info, dtype = tf.float64))
                self.tab_data = tf.get_variable('t_tab_data',
                                                tab_data.shape,
                                                dtype = tf.float64,
                                                trainable = False,
                                                initializer = tf.constant_initializer(tab_data, dtype = tf.float64))

        coord = tf.reshape (coord_, [-1, natoms[1] * 3])
        atype = tf.reshape (atype_, [-1, natoms[1]])

        descrpt = self.descrpt.build(coord_, atype_, natoms, box, mesh, davg = davg, dstd = dstd, suffix = suffix, reuse_attr = reuse_attr, reuse_weights = reuse_weights)

        atom_ener = self.build_atom_net (descrpt, 
                                         fparam, 
                                         natoms, 
                                         bias_atom_e = bias_atom_e, 
                                         reuse = reuse_weights, 
                                         suffix = suffix)

        if self.srtab is not None :
            sw_lambda, sw_deriv \
                = op_module.soft_min_switch(atype, 
                                            rij, 
                                            nlist,
                                            natoms,
                                            sel_a = self.sel_a,
                                            sel_r = self.sel_r,
                                            alpha = self.smin_alpha,
                                            rmin = self.sw_rmin,
                                            rmax = self.sw_rmax)            
            inv_sw_lambda = 1.0 - sw_lambda
            # NOTICE:
            # atom energy is not scaled, 
            # force and virial are scaled
            tab_atom_ener, tab_force, tab_atom_virial \
                = op_module.tab_inter(self.tab_info,
                                      self.tab_data,
                                      atype,
                                      rij,
                                      nlist,
                                      natoms,
                                      sw_lambda,
                                      sel_a = self.sel_a,
                                      sel_r = self.sel_r)
            energy_diff = tab_atom_ener - tf.reshape(atom_ener, [-1, natoms[0]])
            tab_atom_ener = tf.reshape(sw_lambda, [-1]) * tf.reshape(tab_atom_ener, [-1])
            atom_ener = tf.reshape(inv_sw_lambda, [-1]) * atom_ener
            energy_raw = tab_atom_ener + atom_ener
        else :
            energy_raw = atom_ener

        energy_raw = tf.reshape(energy_raw, [-1, natoms[0]], name = 'o_atom_energy'+suffix)
        energy = tf.reduce_sum(global_cvt_2_ener_float(energy_raw), axis=1, name='o_energy'+suffix)

        force, virial, atom_virial \
            = self.descrpt.prod_force_virial (atom_ener, natoms)

        if self.srtab is not None :
            sw_force \
                = op_module.soft_min_force(energy_diff, 
                                           sw_deriv,
                                           nlist, 
                                           natoms,
                                           n_a_sel = self.nnei_a,
                                           n_r_sel = self.nnei_r)
            force = force + sw_force + tab_force

        force = tf.reshape (force, [-1, 3 * natoms[1]], name = "o_force"+suffix)

        if self.srtab is not None :
            sw_virial, sw_atom_virial \
                = op_module.soft_min_virial (energy_diff,
                                             sw_deriv,
                                             rij,
                                             nlist,
                                             natoms,
                                             n_a_sel = self.nnei_a,
                                             n_r_sel = self.nnei_r)
            atom_virial = atom_virial + sw_atom_virial + tab_atom_virial
            virial = virial + sw_virial \
                     + tf.reduce_sum(tf.reshape(tab_atom_virial, [-1, natoms[1], 9]), axis = 1)

        virial = tf.reshape (virial, [-1, 9], name = "o_virial"+suffix)
        atom_virial = tf.reshape (atom_virial, [-1, 9 * natoms[1]], name = "o_atom_virial"+suffix)

        return energy, force, virial, energy_raw, atom_virial


    def build_atom_net (self, 
                        inputs,
                        fparam,
                        natoms,
                        bias_atom_e = None,
                        reuse = None,
                        suffix = '') :
        start_index = 0
        inputs = tf.reshape(inputs, [-1, self.descrpt.get_dim_out() * natoms[0]])
        shape = inputs.get_shape().as_list()
        if bias_atom_e is not None :
            assert(len(bias_atom_e) == self.ntypes)

        for type_i in range(self.ntypes):
            # cut-out inputs
            inputs_i = tf.slice (inputs,
                                 [ 0, start_index*      self.descrpt.get_dim_out()],
                                 [-1, natoms[2+type_i]* self.descrpt.get_dim_out()] )
            inputs_i = tf.reshape(inputs_i, [-1, self.descrpt.get_dim_out()])
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
                    layer+= self.one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed, use_timestep = self.resnet_dt)
                else :
                    layer = self.one_layer(layer, self.n_neuron[ii], name='layer_'+str(ii)+'_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            final_layer = self.one_layer(layer, 1, activation_fn = None, bavg = type_bias_ae, name='final_layer_type_'+str(type_i)+suffix, reuse=reuse, seed = self.seed)
            final_layer = tf.reshape(final_layer, [-1, natoms[2+type_i]])
            # final_layer = tf.cond (tf.equal(natoms[2+type_i], 0), lambda: tf.zeros((0, 0), dtype=global_tf_float_precision), lambda : tf.reshape(final_layer, [-1, natoms[2+type_i]]))

            # concat the results
            if type_i == 0:
                outs = final_layer
            else:
                outs = tf.concat([outs, final_layer], axis = 1)

        return tf.reshape(outs, [-1])






