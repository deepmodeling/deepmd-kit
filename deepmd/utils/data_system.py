#!/usr/bin/env python3

import logging
import os
import collections
import warnings
import numpy as np
from typing import Tuple, List

from deepmd.utils.data import DataSets
from deepmd.utils.data import DeepmdData

log = logging.getLogger(__name__)


class DeepmdDataSystem() :
    """
    Class for manipulating many data systems. 
    It is implemented with the help of DeepmdData
    """
    def __init__ (self,
                  systems : List[str], 
                  batch_size : int,
                  test_size : int,
                  rcut : float,
                  set_prefix : str = 'set',
                  shuffle_test : bool = True,
                  type_map : List[str] = None, 
                  modifier = None, 
                  trn_all_set = False) :
        """
        Constructor
        
        Parameters
        ----------
        systems
                Specifying the paths to systems
        batch_size
                The batch size
        test_size
                The size of test data
        rcut
                The cut-off radius
        set_prefix
                Prefix for the directories of different sets
        shuffle_test
                If the test data are shuffled
        type_map
                Gives the name of different atom types
        modifier
                Data modifier that has the method `modify_data`        
        trn_all_set
                Use all sets as training dataset. Otherwise, if the number of sets is more than 1, the last set is left for test.
        """
        # init data
        self.rcut = rcut
        self.system_dirs = systems
        self.nsystems = len(self.system_dirs)
        self.data_systems = []
        for ii in self.system_dirs :
            self.data_systems.append(
                DeepmdData(
                    ii, 
                    set_prefix=set_prefix, 
                    shuffle_test=shuffle_test, 
                    type_map = type_map, 
                    modifier = modifier, 
                    trn_all_set = trn_all_set
                ))
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
            ntypes.append(ii.get_ntypes())
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

        # ! altered by Marián Rynik
        # test size
        # now test size can be set as a percentage of systems data or test size
        # can be set for each system individualy in the same manner as batch
        # size. This enables one to use systems with diverse number of
        # structures and different number of atoms.
        self.test_size = test_size
        if isinstance(self.test_size, int):
            self.test_size = self.test_size * np.ones(self.nsystems, dtype=int)
        elif isinstance(self.test_size, str):
            words = self.test_size.split('%')
            try:
                percent = int(words[0])
            except ValueError:
                raise RuntimeError('unknown test_size rule ' + words[0])
            self.test_size = self._make_auto_ts(percent)
        elif isinstance(self.test_size, list):
            pass
        else :
            raise RuntimeError('invalid test_size')            
        assert(isinstance(self.test_size, (list,np.ndarray)))
        assert(len(self.test_size) == self.nsystems)

        # prob of batch, init pick idx
        self.prob_nbatches = [ float(i) for i in self.nbatches] / np.sum(self.nbatches)        
        self.pick_idx = 0

        # check batch and test size
        for ii in range(self.nsystems) :
            chk_ret = self.data_systems[ii].check_batch_size(self.batch_size[ii])
            if chk_ret is not None :
                warnings.warn("system %s required batch size is larger than the size of the dataset %s (%d > %d)" % \
                              (self.system_dirs[ii], chk_ret[0], self.batch_size[ii], chk_ret[1]))
            chk_ret = self.data_systems[ii].check_test_size(self.test_size[ii])
            if chk_ret is not None :
                warnings.warn("system %s required test size is larger than the size of the dataset %s (%d > %d)" % \
                              (self.system_dirs[ii], chk_ret[0], self.test_size[ii], chk_ret[1]))


    def _load_test(self, ntests = -1):
        self.test_data = collections.defaultdict(list)
        for ii in range(self.nsystems) :
            test_system_data = self.data_systems[ii].get_test(ntests = ntests)
            for nn in test_system_data:
                self.test_data[nn].append(test_system_data[nn])


    def _make_default_mesh(self):
        self.default_mesh = []
        cell_size = np.max (self.rcut)
        for ii in range(self.nsystems) :
            if self.data_systems[ii].pbc :
                test_system_data = self.data_systems[ii].get_batch(self.batch_size[ii])
                self.data_systems[ii].reset_get_batch()
                # test_system_data = self.data_systems[ii].get_test()
                avg_box = np.average (test_system_data["box"], axis = 0)
                avg_box = np.reshape (avg_box, [3,3])
                ncell = (np.linalg.norm(avg_box, axis=1)/ cell_size).astype(np.int32)
                ncell[ncell < 2] = 2
                default_mesh = np.zeros (6, dtype = np.int32)
                default_mesh[3:6] = ncell
                self.default_mesh.append(default_mesh)
            else:
                self.default_mesh.append(np.array([], dtype = np.int32))


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


    def add_dict(self, 
                 adict : dict
    ) -> None:
        """
        Add items to the data system by a `dict`.
        `adict` should have items like
        adict[key] = {
                   'ndof': ndof, 
                   'atomic': atomic,
                   'must': must, 
                   'high_prec': high_prec,
                   'type_sel': type_sel,
                   'repeat': repeat,
        }        
        For the explaination of the keys see `add`
        """
        for kk in adict :
            self.add(kk, 
                     adict[kk]['ndof'], 
                     atomic=adict[kk]['atomic'], 
                     must=adict[kk]['must'], 
                     high_prec=adict[kk]['high_prec'], 
                     type_sel=adict[kk]['type_sel'], 
                     repeat=adict[kk]['repeat'])

    def add(self, 
            key : str, 
            ndof : int, 
            atomic : bool = False, 
            must : bool = False, 
            high_prec : bool = False,
            type_sel : List[int] = None,
            repeat : int = 1
    ) :
        """
        Add a data item that to be loaded

        Parameters
        ----------
        key 
                The key of the item. The corresponding data is stored in `sys_path/set.*/key.npy`
        ndof
                The number of dof
        atomic
                The item is an atomic property.
                If False, the size of the data should be nframes x ndof
                If True, the size of data should be nframes x natoms x ndof
        must
                The data file `sys_path/set.*/key.npy` must exist.
                If must is False and the data file does not exist, the `data_dict[find_key]` is set to 0.0
        high_prec
                Load the data and store in float64, otherwise in float32
        type_sel
                Select certain type of atoms
        repeat
                The data will be repeated `repeat` times.
        """
        for ii in self.data_systems:
            ii.add(key, ndof, atomic=atomic, must=must, high_prec=high_prec, repeat=repeat, type_sel=type_sel)

    def reduce(self, 
               key_out,
               key_in) :
        """
        Generate a new item from the reduction of another atom

        Parameters
        ----------
        key_out
                The name of the reduced item
        key_in
                The name of the data item to be reduced
        """
        for ii in self.data_systems:
            ii.reduce(key_out, k_in)

    def get_data_dict(self, 
                      ii : int = 0) -> dict:
        return self.data_systems[ii].get_data_dict()


    def _get_sys_probs(self,
                       sys_probs,
                       auto_prob_style) :        
        if sys_probs is None :
            if auto_prob_style == "prob_uniform" :
                prob_v = 1./float(self.nsystems)
                prob = [prob_v for ii in range(self.nsystems)]
            elif auto_prob_style == "prob_sys_size" :
                prob = self.prob_nbatches
            elif auto_prob_style[:14] == "prob_sys_size;" :
                prob = self._prob_sys_size_ext(auto_prob_style)
            else :
                raise RuntimeError("unkown style " + auto_prob_style )
        else :
            prob = self._process_sys_probs(sys_probs)
        return prob


    def get_batch (self, 
                   sys_idx : int = None,
                   sys_probs : List[float] = None,
                   auto_prob_style : str = "prob_sys_size") :
        """
        Get a batch of data from the data systems

        Parameters
        ----------
        sys_idx: int
            The index of system from which the batch is get. 
            If sys_idx is not None, `sys_probs` and `auto_prob_style` are ignored
            If sys_idx is None, automatically determine the system according to `sys_probs` or `auto_prob_style`, see the following.
        sys_probs: list of float
            The probabilitis of systems to get the batch.
            Summation of positive elements of this list should be no greater than 1.
            Element of this list can be negative, the probability of the corresponding system is determined automatically by the number of batches in the system.
        auto_prob_style: str
            Determine the probability of systems automatically. The method is assigned by this key and can be
            - "prob_uniform"  : the probability all the systems are equal, namely 1.0/self.get_nsystems()
            - "prob_sys_size" : the probability of a system is proportional to the number of batches in the system
            - "prob_sys_size;stt_idx:end_idx:weight;stt_idx:end_idx:weight;..." : 
                                the list of systems is devided into blocks. A block is specified by `stt_idx:end_idx:weight`, 
                                where `stt_idx` is the starting index of the system, `end_idx` is then ending (not including) index of the system,
                                the probabilities of the systems in this block sums up to `weight`, and the relatively probabilities within this block is proportional 
                                to the number of batches in the system.
        """
        if not hasattr(self, 'default_mesh') :
            self._make_default_mesh()
        if sys_idx is not None :
            self.pick_idx = sys_idx
        else :
            prob = self._get_sys_probs(sys_probs, auto_prob_style)
            self.pick_idx = np.random.choice(np.arange(self.nsystems), p = prob)
        b_data = self.data_systems[self.pick_idx].get_batch(self.batch_size[self.pick_idx])
        b_data["natoms_vec"] = self.natoms_vec[self.pick_idx]
        b_data["default_mesh"] = self.default_mesh[self.pick_idx]
        return b_data

    # ! altered by Marián Rynik
    def get_test (self, 
                  sys_idx : int = None,
                  n_test : int = -1
    ) :
        """
        Get test data from the the data systems.

        Parameters
        ----------
        sys_idx
                The test dat of system with index `sys_idx` will be returned. 
                If is None, the currently selected system will be returned.
        n_test
                Number of test data. If set to -1 all test data will be get.
        """
        if not hasattr(self, 'default_mesh') :
            self._make_default_mesh()
        if not hasattr(self, 'test_data') :
            self._load_test(ntests = n_test)
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

    def get_sys_ntest(self, sys_idx=None):
        """
        Get number of tests for the currently selected system,
            or one defined by sys_idx.
        """
        if sys_idx is not None :
            return self.test_size[sys_idx]
        else :
            return self.test_size[self.pick_idx]
            
    def get_type_map(self) -> List[str]:
        """
        Get the type map
        """
        return self.type_map

    def get_nbatches (self) -> int: 
        """
        Get the total number of batches
        """
        return self.nbatches
    
    def get_ntypes (self) -> int:
        """
        Get the number of types
        """
        return self.sys_ntypes

    def get_nsystems (self) -> int:
        """
        Get the number of data systems
        """
        return self.nsystems

    def get_sys (self, idx : int) -> DeepmdData:
        """
        Get a certain data system
        """
        return self.data_systems[idx]

    def get_batch_size(self) -> int:
        """
        Get the batch size
        """
        return self.batch_size

    def _format_name_length(self, name, width) :
        if len(name) <= width:
            return '{: >{}}'.format(name, width)
        else :
            name = name[-(width-3):]
            name = '-- ' + name
            return name 

    def print_summary(self, 
                      run_opt,
                      sys_probs = None,
                      auto_prob_style = "prob_sys_size") :
        prob = self._get_sys_probs(sys_probs, auto_prob_style)
        # width 65
        sys_width = 42
        log.info("---Summary of DataSystem--------------------------------------------------------------")
        log.info("found %d system(s):" % self.nsystems)
        log.info(("%s  " % self._format_name_length('system', sys_width)) + 
                 ("%6s  %6s  %6s  %6s  %5s  %3s" % ('natoms', 'bch_sz', 'n_bch', "n_test", 'prob', 'pbc')))
        for ii in range(self.nsystems) :
            log.info("%s  %6d  %6d  %6d  %6d  %5.3f  %3s" % 
                     (self._format_name_length(self.system_dirs[ii], sys_width),
                      self.natoms[ii], 
                      # TODO batch size * nbatches = number of structures
                      self.batch_size[ii],
                      self.nbatches[ii],
                      self.test_size[ii],
                      prob[ii],
                      "T" if self.data_systems[ii].pbc else "F"
                     ) )
        log.info("--------------------------------------------------------------------------------------")

    def _make_auto_bs(self, rule) :
        bs = []
        for ii in self.data_systems:
            ni = ii.get_natoms()
            bsi = rule // ni
            if bsi * ni < rule:
                bsi += 1
            bs.append(bsi)
        return bs

    # ! added by Marián Rynik
    def _make_auto_ts(self, percent):
        ts = []
        for ii in range(self.nsystems):
            ni = self.batch_size[ii] * self.nbatches[ii]
            tsi = int(ni * percent / 100)
            ts.append(tsi)

        return ts

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

    def _process_sys_probs(self, sys_probs) :
        sys_probs = np.array(sys_probs)
        type_filter = sys_probs >= 0
        assigned_sum_prob = np.sum(type_filter * sys_probs)
        assert assigned_sum_prob <= 1, "the sum of assigned probability should be less than 1"
        rest_sum_prob = 1. - assigned_sum_prob
        if rest_sum_prob != 0 :
            rest_nbatch = (1 - type_filter) * self.nbatches
            rest_prob = rest_sum_prob * rest_nbatch / np.sum(rest_nbatch)
            ret_prob = rest_prob + type_filter * sys_probs
        else :
            ret_prob = sys_probs
        assert np.sum(ret_prob) == 1, "sum of probs should be 1"
        return ret_prob
    
    def _prob_sys_size_ext(self, keywords):
        block_str = keywords.split(';')[1:]
        block_stt = []
        block_end = []
        block_weights = []
        for ii in block_str:
            stt = int(ii.split(':')[0])
            end = int(ii.split(':')[1])
            weight = float(ii.split(':')[2])
            assert(weight >= 0), "the weight of a block should be no less than 0"
            block_stt.append(stt)
            block_end.append(end)
            block_weights.append(weight)
        nblocks = len(block_str)
        block_probs = np.array(block_weights) / np.sum(block_weights)
        sys_probs = np.zeros([self.get_nsystems()])
        for ii in range(nblocks):
            nbatch_block = self.nbatches[block_stt[ii]:block_end[ii]]
            tmp_prob = [float(i) for i in nbatch_block] / np.sum(nbatch_block)
            sys_probs[block_stt[ii]:block_end[ii]] = tmp_prob * block_probs[ii]
        return sys_probs



class DataSystem (object) :
    """
    Outdated class for the data systems. Not maintained anymore.    
    """
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
        log.info(tmp_msg)

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
            
