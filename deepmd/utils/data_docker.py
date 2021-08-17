#!/usr/bin/env python3

import logging
import os
import collections
import warnings
import numpy as np
from typing import Tuple, List

from deepmd.utils.data import DataSets
from deepmd.utils.data import DeepmdData
from deepmd.utils.data_system import DeepmdDataSystem, DataSystem
from deepmd.common import data_requirement

log = logging.getLogger(__name__)

class DeepmdDataDocker() :
    """
    Class for manipulating many data dockers. 
    It is implemented with the help of DeepmdData
    """
    def __init__ (self,
                  data_systems,
                  batch_size : int,
                  rcut : int,
                  type_map : List[str] = None,
                  sys_probs = None,
                  auto_prob_style = "prob_sys_size",
                  auto_prob_style_method = "prob_uniform",
                  modifier = None,
                  ) :
        """
        Constructor
        
        Parameters
        ----------
        datasystems
                Combination of several DeepmdDataSystems
        batch_size
                The batch size
        type_map
                Gives the name of different atom types
        sys_probs
                sys_probs of systems in a DeepmdDataSystem
        auto_prob_style
                auto_prob_style of the systems in a DeepmdDataSystem
        auto_prob_style_method
                auto_prob_style of the methods in the DeepmdDataDocker
        modifier
                Data modifier that has the method `modify_data` 
        """
        # init data
        total_data = []
        for sub_sys in data_systems:
            sys_name = sub_sys['name']
            data = DeepmdDataSystem(
                systems=sub_sys['data'],
                batch_size=batch_size,
                test_size=1,        # to satisfy the old api
                shuffle_test=True,  # to satisfy the old api
                rcut=rcut,
                #type_map=sub_fitting['type_map'],  # this is the local type map
                type_map=type_map,  # this is the local type map
                modifier=modifier,
                trn_all_set=True,    # sample from all sets
                sys_probs=sys_probs,
                auto_prob_style=auto_prob_style,
                name = sys_name
            )
            data.add_dict(data_requirement)
            total_data.append(data)
        self.data_systems = total_data
        self.batch_size = batch_size
        self.sys_probs = sys_probs
        # natoms, nbatches
        self.nmethod = len(self.data_systems)
        self.pick_idx = 0
        nbatch_list = []
        batch_size_list=[]
        name_list = []
        method_nbatch = []
        for ii in range(self.nmethod) :
            nbatch_list.extend(self.data_systems[ii].get_nbatches())
            method_nbatch.append(np.sum(self.data_systems[ii].get_nbatches()))
            batch_size_list.extend(self.data_systems[ii].get_batch_size())
            name_list.append(str(self.data_systems[ii].get_name()))
        self.type_map = type_map
        self.nbatches = list(nbatch_list)
        self.method_nbatch = list(method_nbatch)
        self.batch_size = list(batch_size_list)
        self.name_list = list(name_list)
        self.prob_nmethod = [ float(i) for i in self.method_nbatch] / np.sum(self.method_nbatch) 
        self.set_sys_probs(sys_probs, auto_prob_style_method)
        



    def get_nmethod(self):
        return self.nmethod
    
    def set_sys_probs(self, sys_probs=None,
                      auto_prob_style: str = "prob_sys_size"):
        if sys_probs is None :
            if auto_prob_style == "prob_uniform":
                prob_v = 1./float(self.nmethod)
                probs = [prob_v for ii in range(self.nmethod)]
            elif auto_prob_style == "prob_sys_size":
                probs = self.prob_nmethod
            elif auto_prob_style[:14] == "prob_sys_size;":
                probs = self._prob_sys_size_ext(auto_prob_style)
            else:
                raise RuntimeError("Unknown auto prob style: " + auto_prob_style)
        else:
            probs = self._process_sys_probs(sys_probs)
        self.sys_probs = probs

    def _get_sys_probs(self,
                       sys_probs,
                       auto_prob_style) :  # depreciated
        if sys_probs is None :
            if auto_prob_style == "prob_uniform" :
                prob_v = 1./float(self.nmethod)
                prob = [prob_v for ii in range(self.nmethod)]
            elif auto_prob_style == "prob_sys_size" :
                prob = self.prob_nmethod
            elif auto_prob_style[:14] == "prob_sys_size;" :
                prob = self._prob_sys_size_ext(auto_prob_style)
            else :
                raise RuntimeError("unkown style " + auto_prob_style )
        else :
            prob = self._process_sys_probs(sys_probs)
        return prob

    def get_batch(self, method_idx : int = None, sys_idx : int = None):
        # batch generation style be the same as DeepmdDataSystem
        """
        Get a batch of data from the data systems

        Parameters
        ----------
        method_idx: int
            The index of method from which the batch is get.
        sys_idx: int
            The index of system from which the batch is get. 
            If sys_idx is not None, `sys_probs` and `auto_prob_style` are ignored
            If sys_idx is None, automatically determine the system according to `sys_probs` or `auto_prob_style`, see the following.
        """
        if method_idx is not None :
            self.pick_idx = method_idx
        else :
            # prob = self._get_sys_probs(sys_probs, auto_prob_style)
            self.pick_idx = np.random.choice(np.arange(self.nmethod), p=self.sys_probs)
        
        s_data = self.data_systems[self.pick_idx]
        b_data = {}
        b_data['data'] = s_data.get_batch(sys_idx)
        b_data['pick_method'] = self.pick_idx
        
        return b_data


    def get_data_system(self,name):
        for iname, idata_system in zip(self.name_list, self.data_systems):
            if iname == name:
                return idata_system

    def get_data_system_idx(self,idx):
        return self.data_systems[idx]

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
        return len(self.type_map)
    
    def get_batch_size(self) -> int:
        """
        Get the batch size
        """
        return self.batch_size

    def get_data_dict(self, ii: int = 0) -> dict:
        return self.data_systems[ii].get_data_dict()
    
    def get_name(self):
        return self.name_list

    def print_summary(self, name) :
        # width 65
        sys_width = 42
        log.info(f"---Summary of DataSystem: {name:13s}-----------------------------------------------")
        log.info("found %d methods(s):" % self.nmethod)
        for jj in range(self.nmethod):
            tmp_sys = self.data_systems[jj]
            tmp_sys_prob = self.sys_probs[jj]

            log.info(("%s  " % tmp_sys._format_name_length('system', sys_width)) + 
                 ("%6s  %6s  %6s  %5s  %3s" % ('natoms', 'bch_sz', 'n_bch', 'prob', 'pbc')))
            for ii in range(tmp_sys.nsystems) :
                log.info("%s  %6d  %6d  %6d  %5.3f  %3s" % 
                     (tmp_sys._format_name_length(tmp_sys.system_dirs[ii], sys_width),
                      tmp_sys.natoms[ii], 
                      # TODO batch size * nbatches = number of structures
                      tmp_sys.batch_size[ii],
                      tmp_sys.nbatches[ii],
                      tmp_sys_prob*tmp_sys.sys_probs[ii],
                      "T" if tmp_sys.data_systems[ii].pbc else "F"
                     ) )
            log.info("--------------------------------------------------------------------------------------")

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
            
