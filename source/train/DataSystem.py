#!/usr/bin/env python3

import os, sys
import collections
import numpy as np

module_path = os.path.dirname(os.path.realpath(__file__)) + "/"
sys.path.append (module_path)
from Data import DataSets
from Data import DeepmdData


class DeepmdDataSystem() :
    def __init__ (self,
                  systems, 
                  batch_size,
                  test_size,
                  rcut,
                  set_prefix = 'set',
                  shuffle_test = True,
                  run_opt = None, 
                  type_map = None) :
        # init data
        self.rcut = rcut
        self.system_dirs = systems
        self.nsystems = len(self.system_dirs)
        self.data_systems = []
        for ii in self.system_dirs :
            self.data_systems.append(DeepmdData(ii, 
                                                set_prefix=set_prefix, 
                                                shuffle_test=shuffle_test, 
                                                type_map = type_map))

        # batch size
        self.batch_size = batch_size
        if isinstance(self.batch_size, int) :
            self.batch_size = self.batch_size * np.ones(self.nsystems, dtype=int)
        elif isinstance(self.batch_size, str):
            words = self.batch_size.split(':')
            if 'auto' == words[0] :
                rule = 32
                if len(words) == 2 :
                    rule = int(words[1])
            else:
                raise RuntimeError('unknown batch_size rule ' + words[0])
            self.batch_size = self._make_auto_bs(rule)
        elif isinstance(self.batch_size, list):
            pass
        else :
            raise RuntimeError('invalid batch_size')            
        assert(isinstance(self.batch_size, (list,np.ndarray)))
        assert(len(self.batch_size) == self.nsystems)

        # natoms, nbatches
        ntypes = []
        for ii in self.data_systems :
            ntypes.append(np.max(ii.get_atom_type()) + 1)
        self.sys_ntypes = max(ntypes)
        self.natoms = []
        self.natoms_vec = []
        self.nbatches = []
        type_map_list = []
        for ii in range(self.nsystems) :
            self.natoms.append(self.data_systems[ii].get_natoms())
            self.natoms_vec.append(self.data_systems[ii].get_natoms_vec(self.sys_ntypes).astype(int))
            self.nbatches.append(self.data_systems[ii].get_sys_numb_batch(self.batch_size[ii]))
            type_map_list.append(self.data_systems[ii].get_type_map())
        self.type_map = self._check_type_map_consistency(type_map_list)

        # prob of batch, init pick idx
        self.prob_nbatches = [ float(i) for i in self.nbatches] / np.sum(self.nbatches)        
        self.pick_idx = 0

        # check batch and test size
        for ii in range(self.nsystems) :
            chk_ret = self.data_systems[ii].check_batch_size(self.batch_size[ii])
            if chk_ret is not None :
                raise RuntimeError ("system %s required batch size %d is larger than the size %d of the dataset %s" % \
                                    (self.system_dirs[ii], self.batch_size[ii], chk_ret[1], chk_ret[0]))
            chk_ret = self.data_systems[ii].check_test_size(test_size)
            if chk_ret is not None :
                print("WARNNING: system %s required test size %d is larger than the size %d of the dataset %s" % \
                      (self.system_dirs[ii], test_size, chk_ret[1], chk_ret[0]))

        # print summary
        if run_opt is not None:
            self.print_summary(run_opt)


    def _load_test(self):
        self.test_data = collections.defaultdict(list)
        self.default_mesh = []
        for ii in range(self.nsystems) :
            test_system_data = self.data_systems[ii].get_test ()
            for nn in test_system_data:
                self.test_data[nn].append(test_system_data[nn])
            cell_size = np.max (self.rcut)
            avg_box = np.average (test_system_data["box"], axis = 0)
            avg_box = np.reshape (avg_box, [3,3])
            ncell = (np.linalg.norm(avg_box, axis=1)/ cell_size).astype(np.int32)
            ncell[ncell < 2] = 2
            default_mesh = np.zeros (6, dtype = np.int32)
            default_mesh[3:6] = ncell
            self.default_mesh.append(default_mesh)


    def compute_energy_shift(self, rcond = 1e-3, key = 'energy') :
        sys_ener = np.array([])
        for ss in self.data_systems :
            sys_ener = np.append(sys_ener, ss.avg(key))
        sys_tynatom = np.array(self.natoms_vec, dtype = float)
        sys_tynatom = np.reshape(sys_tynatom, [self.nsystems,-1])
        sys_tynatom = sys_tynatom[:,2:]
        energy_shift,resd,rank,s_value \
            = np.linalg.lstsq(sys_tynatom, sys_ener, rcond = rcond)
        return energy_shift


    def add_dict(self, adict) :
        for kk in adict :
            self.add(kk, 
                     adict[kk]['ndof'], 
                     atomic=adict[kk]['atomic'], 
                     must=adict[kk]['must'], 
                     high_prec=adict[kk]['high_prec'], 
                     type_sel=adict[kk]['type_sel'], 
                     repeat=adict[kk]['repeat'])

    def add(self, 
            key, 
            ndof, 
            atomic = False, 
            must = False, 
            high_prec = False,
            type_sel = None,
            repeat = 1) :
        for ii in self.data_systems:
            ii.add(key, ndof, atomic=atomic, must=must, high_prec=high_prec, repeat=repeat, type_sel=type_sel)

    def reduce(self, 
               key_out,
               key_in) :
        for ii in self.data_systems:
            ii.reduce(key_out, k_in)

    def get_data_dict(self) :
        return self.data_systems[0].get_data_dict()

    def get_batch (self, 
                   sys_idx = None,
                   sys_weights = None,
                   style = "prob_sys_size") :
        if not hasattr(self, 'default_mesh') :
            self._load_test()
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
        b_data = self.data_systems[self.pick_idx].get_batch(self.batch_size[self.pick_idx])
        b_data["natoms_vec"] = self.natoms_vec[self.pick_idx]
        b_data["default_mesh"] = self.default_mesh[self.pick_idx]
        return b_data

    def get_test (self, 
                  sys_idx = None) :
        if not hasattr(self, 'default_mesh') :
            self._load_test()
        if sys_idx is not None :
            idx = sys_idx
        else :
            idx = self.pick_idx
        test_system_data = {}
        for nn in self.test_data:
            test_system_data[nn] = self.test_data[nn][idx]
        test_system_data["natoms_vec"] = self.natoms_vec[idx]
        test_system_data["default_mesh"] = self.default_mesh[idx]
        return test_system_data

            
    def get_type_map(self):
        return self.type_map

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

    def _format_name_length(self, name, width) :
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
        tmp_msg += "%s  " % self._format_name_length('system', sys_width)
        tmp_msg += "%s  %s  %s\n" % ('natoms', 'bch_sz', 'n_bch')
        for ii in range(self.nsystems) :
            tmp_msg += ("%s  %6d  %6d  %5d\n" % 
                        (self._format_name_length(self.system_dirs[ii], sys_width),
                         self.natoms[ii], 
                         self.batch_size[ii], 
                         self.nbatches[ii]) )
        tmp_msg += "-----------------------------------------------------------------\n"
        run_opt.message(tmp_msg)

        
    def _make_auto_bs(self, rule) :
        bs = []
        for ii in self.data_systems:
            ni = ii.get_natoms()
            bsi = rule // ni
            if bsi * ni < rule:
                bsi += 1
            bs.append(bsi)
        return bs

    def _check_type_map_consistency(self, type_map_list):
        ret = []
        for ii in type_map_list:
            if ii is not None:
                min_len = min([len(ii), len(ret)])
                for idx in range(min_len) :
                    if ii[idx] != ret[idx] :
                        raise RuntimeError('inconsistent type map: %s %s' % (str(ret), str(ii)))
                if len(ii) > len(ret) :
                    ret = ii
        return ret

    def _process_sys_weights(self, sys_weights) :
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
        type_map = []
        for ii in range(self.nsystems) :
            self.natoms.append(self.data_systems[ii].get_natoms())
            self.natoms_vec.append(self.data_systems[ii].get_natoms_vec(self.sys_ntypes).astype(int))
            self.nbatches.append(self.data_systems[ii].get_sys_numb_batch(self.batch_size[ii]))
            type_map.append(self.data_systems[ii].get_type_map())
        self.type_map = self.check_type_map_consistency(type_map)

        # check frame parameters
        has_fparam = [ii.numb_fparam() for ii in self.data_systems]
        for ii in has_fparam :
            if ii != has_fparam[0] :
                raise RuntimeError("if any system has frame parameter, then all systems should have the same number of frame parameter")
        self.has_fparam = has_fparam[0]

        # check the size of data if they satisfy the requirement of batch and test
        for ii in range(self.nsystems) :
            chk_ret = self.data_systems[ii].check_batch_size(self.batch_size[ii])
            if chk_ret is not None :
                raise RuntimeError ("system %s required batch size %d is larger than the size %d of the dataset %s" % \
                                    (self.system_dirs[ii], self.batch_size[ii], chk_ret[1], chk_ret[0]))
            chk_ret = self.data_systems[ii].check_test_size(test_size)
            if chk_ret is not None :
                print("WARNNING: system %s required test size %d is larger than the size %d of the dataset %s" % \
                      (self.system_dirs[ii], test_size, chk_ret[1], chk_ret[0]))

        if run_opt is not None:
            self.print_summary(run_opt)

        self.prob_nbatches = [ float(i) for i in self.nbatches] / np.sum(self.nbatches)

        self.test_data = collections.defaultdict(list)
        self.default_mesh = []
        for ii in range(self.nsystems) :
            test_system_data = self.data_systems[ii].get_test ()
            for nn in test_system_data:
                self.test_data[nn].append(test_system_data[nn])
            cell_size = np.max (rcut)
            avg_box = np.average (test_system_data["box"], axis = 0)
            avg_box = np.reshape (avg_box, [3,3])
            ncell = (np.linalg.norm(avg_box, axis=1)/ cell_size).astype(np.int32)
            ncell[ncell < 2] = 2
            default_mesh = np.zeros (6, dtype = np.int32)
            default_mesh[3:6] = ncell
            self.default_mesh.append(default_mesh)
        self.pick_idx = 0


    def check_type_map_consistency(self, type_map_list):
        ret = []
        for ii in type_map_list:
            if ii is not None:
                min_len = min([len(ii), len(ret)])
                for idx in range(min_len) :
                    if ii[idx] != ret[idx] :
                        raise RuntimeError('inconsistent type map: %s %s' % (str(ret), str(ii)))
                if len(ii) > len(ret) :
                    ret = ii
        return ret


    def get_type_map(self):
        return self.type_map


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
            = np.linalg.lstsq(sys_tynatom, sys_ener, rcond = 1e-3)
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
        b_data = self.data_systems[self.pick_idx].get_batch(self.batch_size[self.pick_idx])
        b_data["natoms_vec"] = self.natoms_vec[self.pick_idx]
        b_data["default_mesh"] = self.default_mesh[self.pick_idx]
        return b_data

    def get_test (self, 
                  sys_idx = None) :
        if sys_idx is not None :
            idx = sys_idx
        else :
            idx = self.pick_idx
        test_system_data = {}
        for nn in self.test_data:
            test_system_data[nn] = self.test_data[nn][idx]
        test_system_data["natoms_vec"] = self.natoms_vec[idx]
        test_system_data["default_mesh"] = self.default_mesh[idx]
        return test_system_data
            
    def get_nbatches (self) : 
        return self.nbatches
    
    def get_ntypes (self) :
        return self.sys_ntypes

    def get_nsystems (self) :
        return self.nsystems

    def get_sys (self, sys_idx) :
        return self.data_systems[sys_idx]

    def get_batch_size(self) :
        return self.batch_size

    def numb_fparam(self) :
        return self.has_fparam

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
            
