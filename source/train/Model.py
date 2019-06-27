import os,warnings
import numpy as np
import tensorflow as tf
from deepmd.common import j_must_have, j_must_have_d, j_have

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

class Model() :
    def __init__ (self, jdata, descrpt, fitting):
        self.descrpt = descrpt
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
	# type_map
        self.type_map = []
        if j_have(jdata, 'type_map') :
            self.type_map = jdata['type_map']
        # fitting
        self.fitting = fitting
        self.numb_fparam = self.fitting.get_numb_fparam()
        # short-range tab
        if 'use_srtab' in jdata :
            self.srtab = TabInter(jdata['use_srtab'])
            self.smin_alpha = j_must_have(jdata, 'smin_alpha')
            self.sw_rmin = j_must_have(jdata, 'sw_rmin')
            self.sw_rmax = j_must_have(jdata, 'sw_rmax')
        else :
            self.srtab = None


    def get_rcut (self) :
        return self.rcut

    def get_ntypes (self) :
        return self.ntypes

    def get_numb_fparam (self) :
        return self.numb_fparam

    def get_type_map (self) :
        return self.type_map

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
        # if self.run_opt.is_chief:
        #     np.savetxt ("stat.avg.out", davg.T)
        #     np.savetxt ("stat.std.out", dstd.T)

        bias_atom_e = data.compute_energy_shift()
        # self._message("computed energy bias")

        return davg, dstd, bias_atom_e

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
                           reuse = None):

        with tf.variable_scope('model_attr' + suffix, reuse = reuse) :
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

        descrpt = self.descrpt.build(coord_, 
                                     atype_, 
                                     natoms, 
                                     box, 
                                     mesh, 
                                     davg = davg, 
                                     dstd = dstd, 
                                     suffix = suffix, 
                                     reuse = reuse)

        atom_ener = self.fitting.build (descrpt, 
                                        fparam, 
                                        natoms, 
                                        bias_atom_e = bias_atom_e, 
                                        reuse = reuse, 
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
    
