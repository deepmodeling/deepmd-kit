#!/usr/bin/env python3

import os, sys
import numpy as np

module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append (module_path)
from Data import DataSets

class DataSystem (object) :
    def __init__ (self,
                  systems,
                  set_prefix,
                  batch_size,
                  test_size,
                  rcut, 
                  run_opt = None) : 
        self.system_dirs = systems
        self.nsystems = len(self.system_dirs)
        self.batch_size = batch_size
        if isinstance(self.batch_size, int) :
            self.batch_size = self.batch_size * np.ones(self.nsystems, dtype=int)
        assert(isinstance(self.batch_size, (list,np.ndarray)))
        assert(len(self.batch_size) == self.nsystems)
        self.data_systems = []
        self.ntypes = []
        self.natoms = []
        self.natoms_vec = []
        self.nbatches = []
        for ii in self.system_dirs :
            self.data_systems.append(DataSets(ii, set_prefix))
            sys_all_types = np.loadtxt(os.path.join(ii, "type.raw")).astype(int)
            self.ntypes.append(np.max(sys_all_types) + 1)
        self.sys_ntypes = max(self.ntypes)
        for ii in range(self.nsystems) :
            self.natoms.append(self.data_systems[ii].get_natoms())
            self.natoms_vec.append(self.data_systems[ii].get_natoms_vec(self.sys_ntypes).astype(int))
            self.nbatches.append(self.data_systems[ii].get_sys_numb_batch(self.batch_size[ii]))

        # check the size of data if they satisfy the requirement of batch and test
        for ii in range(self.nsystems) :
            chk_ret = self.data_systems[ii].check_batch_size(self.batch_size[ii])
            if chk_ret is not None :
                raise RuntimeError(" required batch size %d is larger than the size %d of the dataset %s" % (self.batch_size[ii], chk_ret[1], chk_ret[0]))
            chk_ret = self.data_systems[ii].check_test_size(test_size)
            if chk_ret is not None :
                raise RuntimeError(" required test size %d is larger than the size %d of the dataset %s" % (test_size, chk_ret[1], chk_ret[0]))

        if run_opt is not None:
            self.print_summary(run_opt)

        self.prob_nbatches = [ float(i) for i in self.nbatches] / np.sum(self.nbatches)

        self.test_prop_c = []
        self.test_energy = []
        self.test_force = []
        self.test_virial = []
        self.test_atom_ener = []
        self.test_coord = []
        self.test_box = []
        self.test_type = []
        self.default_mesh = []
        for ii in range(self.nsystems) :
            test_prop_c, test_energy, test_force, test_virial, test_atom_ener, test_coord, test_box, test_type \
                = self.data_systems[ii].get_test ()
            self.test_prop_c.append(test_prop_c)
            self.test_energy.append(test_energy)
            self.test_force.append(test_force)
            self.test_virial.append(test_virial)
            self.test_atom_ener.append(test_atom_ener)
            self.test_coord.append(test_coord)
            self.test_box.append(test_box)
            self.test_type.append(test_type)
            ncell = np.ones (3, dtype=np.int32)
            cell_size = np.max (rcut)
            avg_box = np.average (test_box, axis = 0)
            avg_box = np.reshape (avg_box, [3,3])
            for ii in range (3) :
                ncell[ii] = int ( np.linalg.norm(avg_box[ii]) / cell_size )
                if (ncell[ii] < 2) : ncell[ii] = 2
            default_mesh = np.zeros (6, dtype = np.int32)
            default_mesh[3] = ncell[0]
            default_mesh[4] = ncell[1]
            default_mesh[5] = ncell[2]
            self.default_mesh.append(default_mesh)
        self.pick_idx = 0

    def format_name_length(self, name, width) :
        if len(name) <= width:
            return '{: >{}}'.format(name, width)
        else :
            name = name[-(width-3):]
            name = '-- ' + name
            return name 

    def print_summary(self, run_opt) :
        tmp_msg = ""
        # width 65
        sys_width = 42
        tmp_msg += "---Summary of DataSystem-----------------------------------------\n"
        tmp_msg += "find %d system(s):\n" % self.nsystems
        tmp_msg += "%s  " % self.format_name_length('system', sys_width)
        tmp_msg += "%s  %s  %s\n" % ('natoms', 'bch_sz', 'n_bch')
        for ii in range(self.nsystems) :
            tmp_msg += ("%s  %6d  %6d  %5d\n" % 
                        (self.format_name_length(self.system_dirs[ii], sys_width),
                         self.natoms[ii], 
                         self.batch_size[ii], 
                         self.nbatches[ii]) )
        tmp_msg += "-----------------------------------------------------------------\n"
        run_opt.message(tmp_msg)

    def compute_energy_shift(self) :
        sys_ener = np.array([])
        for ss in self.data_systems :
            sys_ener = np.append(sys_ener, ss.get_ener())
        sys_tynatom = np.array(self.natoms_vec, dtype = float)
        sys_tynatom = np.reshape(sys_tynatom, [self.nsystems,-1])
        sys_tynatom = sys_tynatom[:,2:]
        energy_shift,resd,rank,s_value \
            = np.linalg.lstsq(sys_tynatom, sys_ener, rcond = -1)
        return energy_shift

    def process_sys_weights(self, sys_weights) :
        sys_weights = np.array(sys_weights)
        type_filter = sys_weights >= 0
        assigned_sum_prob = np.sum(type_filter * sys_weights)
        assert assigned_sum_prob <= 1, "the sum of assigned probability should be less than 1"
        rest_sum_prob = 1. - assigned_sum_prob
        rest_nbatch = (1 - type_filter) * self.nbatches
        rest_prob = rest_sum_prob * rest_nbatch / np.sum(rest_nbatch)
        ret_prob = rest_prob + type_filter * sys_weights
        assert np.sum(ret_prob) == 1, "sum of probs should be 1"
        return ret_prob

    def get_batch (self, 
                   sys_idx = None,
                   sys_weights = None,
                   style = "prob_sys_size") :
        if sys_idx is not None :
            self.pick_idx = sys_idx
        else :
            if sys_weights is None :
                if style == "prob_sys_size" :
                    prob = self.prob_nbatches
                elif style == "prob_uniform" :
                    prob = None
                else :
                    raise RuntimeError("unkown get_batch style")
            else :
                prob = self.process_sys_weights(sys_weights)
            self.pick_idx = np.random.choice(np.arange(self.nsystems), p = prob)
        b_prop_c, b_energy, b_force, b_virial, b_atom_ener, b_coord, b_box, b_type \
            = self.data_systems[self.pick_idx].get_batch(self.batch_size[self.pick_idx])
        return \
            b_prop_c, \
            b_energy, b_force, b_virial, b_atom_ener, \
            b_coord, b_box, b_type, \
            self.natoms_vec[self.pick_idx], \
            self.default_mesh[self.pick_idx]

    def get_test (self, 
                  sys_idx = None) :
        if sys_idx is not None :
            idx = sys_idx
        else :
            idx = self.pick_idx
        
        return \
            self.test_prop_c[idx], \
            self.test_energy[idx], \
            self.test_force[idx], \
            self.test_virial[idx], \
            self.test_atom_ener[idx], \
            self.test_coord[idx], \
            self.test_box[idx], \
            self.test_type[idx], \
            self.natoms_vec[idx], \
            self.default_mesh[idx]
            
    def get_nbatches (self) : 
        return self.nbatches
    
    def get_ntypes (self) :
        return self.sys_ntypes

    def get_nsystems (self) :
        return self.nsystems

    def get_sys (self, idx) :
        return self.data_systems[idx]

    def get_batch_size(self) :
        return self.batch_size

def _main () :
    sys =  ['/home/wanghan/study/deep.md/results.01/data/mos2/only_raws/20', 
            '/home/wanghan/study/deep.md/results.01/data/mos2/only_raws/30', 
            '/home/wanghan/study/deep.md/results.01/data/mos2/only_raws/38', 
            '/home/wanghan/study/deep.md/results.01/data/mos2/only_raws/MoS2', 
            '/home/wanghan/study/deep.md/results.01/data/mos2/only_raws/Pt_cluster']
    set_prefix = 'set'
    ds = DataSystem (sys, set_prefix, 4, 6)
    r = ds.get_batch()
    print(r[1][0])

if __name__ == '__main__':
    _main()
            
