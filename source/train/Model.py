import os,sys,warnings
import platform
import numpy as np
from deepmd.env import tf
from collections import defaultdict
from deepmd.TabInter import TabInter
from deepmd.common import ClassArg

from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision
from deepmd.RunOptions import global_ener_float_precision
from deepmd.RunOptions import global_cvt_2_tf_float
from deepmd.RunOptions import global_cvt_2_ener_float

if platform.system() == "Windows":
    ext = "dll"
elif platform.system() == "Darwin":
    ext = "dylib"
else:
    ext = "so"
module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
assert (os.path.isfile (module_path  + "libop_abi.{}".format(ext) )), "op module does not exist"
op_module = tf.load_op_library(module_path + "libop_abi.{}".format(ext))

class Model() :
    model_type = 'ener'

    def __init__ (self, jdata, descrpt, fitting):
        self.descrpt = descrpt
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
        # fitting
        self.fitting = fitting
        self.numb_fparam = self.fitting.get_numb_fparam()

        args = ClassArg()\
               .add('type_map',         list,   default = []) \
               .add('rcond',            float,  default = 1e-3) \
               .add('data_stat_nbatch', int,    default = 10) \
               .add('data_stat_protect',float,  default = 1e-2) \
               .add('use_srtab',        str)
        class_data = args.parse(jdata)
        self.type_map = class_data['type_map']
        self.srtab_name = class_data['use_srtab']
        self.rcond = class_data['rcond']
        self.data_stat_nbatch = class_data['data_stat_nbatch']
        self.data_stat_protect = class_data['data_stat_protect']
        if self.srtab_name is not None :
            self.srtab = TabInter(self.srtab_name)
            args.add('smin_alpha',      float,  must = True)\
                .add('sw_rmin',         float,  must = True)\
                .add('sw_rmax',         float,  must = True)
            class_data = args.parse(jdata)
            self.smin_alpha = class_data['smin_alpha']
            self.sw_rmin = class_data['sw_rmin']
            self.sw_rmax = class_data['sw_rmax']
        else :
            self.srtab = None


    def get_rcut (self) :
        return self.rcut

    def get_ntypes (self) :
        return self.ntypes

    def get_type_map (self) :
        return self.type_map

    def data_stat(self, data):
        all_stat = defaultdict(list)
        for ii in range(data.get_nsystems()) :
            for jj in range(self.data_stat_nbatch) :
                stat_data = data.get_batch (sys_idx = ii)
                for dd in stat_data:
                    if dd == "natoms_vec":
                        stat_data[dd] = stat_data[dd].astype(np.int32) 
                    all_stat[dd].append(stat_data[dd])        
        self._compute_dstats (all_stat, protection = self.data_stat_protect)
        self.bias_atom_e = data.compute_energy_shift(self.rcond)


    def _compute_dstats (self, all_stat, protection = 1e-2) :
        self.davg, self.dstd \
            = self.descrpt.compute_dstats(all_stat['coord'],
                                          all_stat['box'],
                                          all_stat['type'],
                                          all_stat['natoms_vec'],
                                          all_stat['default_mesh'])        
        self.fitting.compute_dstats(all_stat, protection = protection)
    
    def build (self, 
               coord_, 
               atype_,
               natoms,
               box, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):

        with tf.variable_scope('model_attr' + suffix, reuse = reuse) :
            t_tmap = tf.constant(' '.join(self.type_map), 
                                 name = 'tmap', 
                                 dtype = tf.string)
            t_mt = tf.constant(self.model_type, 
                               name = 'model_type', 
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

        dout \
            = self.descrpt.build(coord_,
                                 atype_,
                                 natoms,
                                 box,
                                 mesh,
                                 davg = self.davg,
                                 dstd = self.dstd,
                                 suffix = suffix,
                                 reuse = reuse)
        dout = tf.identity(dout, name='o_descriptor')

        if self.srtab is not None :
            nlist, rij, sel_a, sel_r = self.descrpt.get_nlist()
            nnei_a = np.cumsum(sel_a)[-1]
            nnei_r = np.cumsum(sel_r)[-1]

        atom_ener = self.fitting.build (dout, 
                                        input_dict, 
                                        natoms, 
                                        bias_atom_e = self.bias_atom_e, 
                                        reuse = reuse, 
                                        suffix = suffix)

        if self.srtab is not None :
            sw_lambda, sw_deriv \
                = op_module.soft_min_switch(atype, 
                                            rij, 
                                            nlist,
                                            natoms,
                                            sel_a = sel_a,
                                            sel_r = sel_r,
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
                                      sel_a = sel_a,
                                      sel_r = sel_r)
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
                                           n_a_sel = nnei_a,
                                           n_r_sel = nnei_r)
            force = force + sw_force + tab_force

        force = tf.reshape (force, [-1, 3 * natoms[1]], name = "o_force"+suffix)

        if self.srtab is not None :
            sw_virial, sw_atom_virial \
                = op_module.soft_min_virial (energy_diff,
                                             sw_deriv,
                                             rij,
                                             nlist,
                                             natoms,
                                             n_a_sel = nnei_a,
                                             n_r_sel = nnei_r)
            atom_virial = atom_virial + sw_atom_virial + tab_atom_virial
            virial = virial + sw_virial \
                     + tf.reduce_sum(tf.reshape(tab_atom_virial, [-1, natoms[1], 9]), axis = 1)

        virial = tf.reshape (virial, [-1, 9], name = "o_virial"+suffix)
        atom_virial = tf.reshape (atom_virial, [-1, 9 * natoms[1]], name = "o_atom_virial"+suffix)

        model_dict = {}
        model_dict['energy'] = energy
        model_dict['force'] = force
        model_dict['virial'] = virial
        model_dict['atom_ener'] = energy_raw
        model_dict['atom_virial'] = atom_virial
        
        return model_dict


class TensorModel() :
    def __init__ (self, jdata, descrpt, fitting, var_name):
        self.model_type = var_name        
        self.descrpt = descrpt
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
        # fitting
        self.fitting = fitting

        args = ClassArg()\
               .add('type_map',         list,   default = []) \
               .add('data_stat_nbatch', int,    default = 10)
        class_data = args.parse(jdata)
        self.type_map = class_data['type_map']
        self.data_stat_nbatch = class_data['data_stat_nbatch']
    
    def get_rcut (self) :
        return self.rcut

    def get_ntypes (self) :
        return self.ntypes

    def get_type_map (self) :
        return self.type_map

    def get_sel_type(self):
        return self.fitting.get_sel_type()

    def get_out_size (self) :
        return self.fitting.get_out_size()

    def data_stat(self, data):
        all_stat = defaultdict(list)
        for ii in range(data.get_nsystems()) :
            for jj in range(self.data_stat_nbatch) :
                stat_data = data.get_batch (sys_idx = ii)
                for dd in stat_data:
                    if dd == "natoms_vec":
                        stat_data[dd] = stat_data[dd].astype(np.int32) 
                    all_stat[dd].append(stat_data[dd])        
        self._compute_dstats (all_stat)

    def _compute_dstats (self, all_stat) :        
        self.davg, self.dstd \
            = self.descrpt.compute_dstats(all_stat['coord'],
                                          all_stat['box'],
                                          all_stat['type'],
                                          all_stat['natoms_vec'],
                                          all_stat['default_mesh'])

    def build (self, 
               coord_, 
               atype_,
               natoms,
               box, 
               mesh,
               input_dict,
               suffix = '', 
               reuse = None):
        with tf.variable_scope('model_attr' + suffix, reuse = reuse) :
            t_tmap = tf.constant(' '.join(self.type_map), 
                                 name = 'tmap', 
                                 dtype = tf.string)
            t_st = tf.constant(self.get_sel_type(), 
                               name = 'sel_type',
                               dtype = tf.int32)
            t_mt = tf.constant(self.model_type, 
                               name = 'model_type', 
                               dtype = tf.string)

        coord = tf.reshape (coord_, [-1, natoms[1] * 3])
        atype = tf.reshape (atype_, [-1, natoms[1]])

        dout \
            = self.descrpt.build(coord_,
                                 atype_,
                                 natoms,
                                 box,
                                 mesh,
                                 davg = self.davg,
                                 dstd = self.dstd,
                                 suffix = suffix,
                                 reuse = reuse)
        dout = tf.identity(dout, name='o_descriptor')
        rot_mat = self.descrpt.get_rot_mat()
        rot_mat = tf.identity(rot_mat, name = 'o_rot_mat')

        output = self.fitting.build (dout, 
                                     rot_mat,
                                     natoms, 
                                     reuse = reuse, 
                                     suffix = suffix)
        output = tf.identity(output, name = 'o_' + self.model_type)

        return {self.model_type: output}


class WFCModel(TensorModel):
    def __init__(self, jdata, descrpt, fitting) :
        TensorModel.__init__(self, jdata, descrpt, fitting, 'wfc')


class DipoleModel(TensorModel):
    def __init__(self, jdata, descrpt, fitting) :
        TensorModel.__init__(self, jdata, descrpt, fitting, 'dipole')


class PolarModel(TensorModel):
    def __init__(self, jdata, descrpt, fitting) :
        TensorModel.__init__(self, jdata, descrpt, fitting, 'polar')


class GlobalPolarModel(TensorModel):
    def __init__(self, jdata, descrpt, fitting) :
        TensorModel.__init__(self, jdata, descrpt, fitting, 'global_polar')


