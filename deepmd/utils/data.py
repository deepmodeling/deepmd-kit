#!/usr/bin/env python3

import time
import glob
import numpy as np
import os.path
from typing import Tuple, List

from deepmd.env import GLOBAL_NP_FLOAT_PRECISION
from deepmd.env import GLOBAL_ENER_FLOAT_PRECISION

class DeepmdData() :
    """
    Class for a data system. 
    It loads data from hard disk, and mantains the data as a `data_dict`
    """
    def __init__ (self, 
                  sys_path : str, 
                  set_prefix : str = 'set',
                  shuffle_test : bool = True, 
                  type_map : List[str] = None, 
                  modifier = None,
                  trn_all_set : bool = False) :
        """
        Constructor
        
        Parameters
        ----------
        sys_path
                Path to the data system
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
        self.dirs = glob.glob (os.path.join(sys_path, set_prefix + ".*"))
        self.dirs.sort()
        # load atom type
        self.atom_type = self._load_type(sys_path)
        self.natoms = len(self.atom_type)
        # load atom type map
        self.type_map = self._load_type_map(sys_path)
        if self.type_map is not None:
            assert(len(self.type_map) >= max(self.atom_type)+1)
        # check pbc
        self.pbc = self._check_pbc(sys_path)
        # enforce type_map if necessary
        if type_map is not None and self.type_map is not None:
            atom_type_ = [type_map.index(self.type_map[ii]) for ii in self.atom_type]
            self.atom_type = np.array(atom_type_, dtype = np.int32)
            ntypes = len(self.type_map)
            self.type_map = type_map[:ntypes]
        # make idx map
        self.idx_map = self._make_idx_map(self.atom_type)
        # train dirs
        self.test_dir = self.dirs[-1]
        if trn_all_set:
            self.train_dirs = self.dirs
        else:
            if len(self.dirs) == 1 :
                self.train_dirs = self.dirs
            else :
                self.train_dirs = self.dirs[:-1]
        self.data_dict = {}        
        # add box and coord
        self.add('box', 9, must = True)
        self.add('coord', 3, atomic = True, must = True)
        # set counters
        self.set_count = 0
        self.iterator = 0
        self.shuffle_test = shuffle_test
        # set modifier
        self.modifier = modifier


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
        self.data_dict[key] = {'ndof': ndof, 
                               'atomic': atomic,
                               'must': must, 
                               'high_prec': high_prec,
                               'type_sel': type_sel,
                               'repeat': repeat,
                               'reduce': None,
        }
        return self

    
    def reduce(self, 
               key_out : str,
               key_in : str
    ) :
        """
        Generate a new item from the reduction of another atom

        Parameters
        ----------
        key_out
                The name of the reduced item
        key_in
                The name of the data item to be reduced
        """
        assert (key_in in self.data_dict), 'cannot find input key'
        assert (self.data_dict[key_in]['atomic']), 'reduced property should be atomic'
        assert (not(key_out in self.data_dict)), 'output key should not have been added'
        assert (self.data_dict[key_in]['repeat'] == 1), 'reduced proerties should not have been repeated'

        self.data_dict[key_out] = {'ndof': self.data_dict[key_in]['ndof'],
                                   'atomic': False,
                                   'must': True,
                                   'high_prec': True,
                                   'type_sel': None,
                                   'repeat': 1,
                                   'reduce': key_in,
        }
        return self

    def get_data_dict(self) -> dict:
        """
        Get the `data_dict`
        """
        return self.data_dict

    def check_batch_size (self, batch_size) :        
        """
        Check if the system can get a batch of data with `batch_size` frames.
        """
        for ii in self.train_dirs :
            if self.data_dict['coord']['high_prec'] :
                tmpe = np.load(os.path.join(ii, "coord.npy")).astype(GLOBAL_ENER_FLOAT_PRECISION)
            else:
                tmpe = np.load(os.path.join(ii, "coord.npy")).astype(GLOBAL_NP_FLOAT_PRECISION)
            if tmpe.ndim == 1:
                tmpe = tmpe.reshape([1,-1])            
            if tmpe.shape[0] < batch_size :
                return ii, tmpe.shape[0]
        return None

    def check_test_size (self, test_size) :
        """
        Check if the system can get a test dataset with `test_size` frames.
        """
        if self.data_dict['coord']['high_prec'] :
            tmpe = np.load(os.path.join(self.test_dir, "coord.npy")).astype(GLOBAL_ENER_FLOAT_PRECISION)
        else:
            tmpe = np.load(os.path.join(self.test_dir, "coord.npy")).astype(GLOBAL_NP_FLOAT_PRECISION)            
        if tmpe.ndim == 1:
            tmpe = tmpe.reshape([1,-1])            
        if tmpe.shape[0] < test_size :
            return self.test_dir, tmpe.shape[0]
        else :
            return None

    def get_batch(self, 
                  batch_size : int
    ) -> dict :
        """
        Get a batch of data with `batch_size` frames. The frames are randomly picked from the data system.

        Parameters
        ----------
        batch_size
                size of the batch
        """
        if hasattr(self, 'batch_set') :
            set_size = self.batch_set["coord"].shape[0]
        else :
            set_size = 0
        if self.iterator + batch_size > set_size :
            self._load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
            self.set_count += 1
            set_size = self.batch_set["coord"].shape[0]
            if self.modifier is not None:
                self.modifier.modify_data(self.batch_set)
        iterator_1 = self.iterator + batch_size
        if iterator_1 >= set_size :
            iterator_1 = set_size
        idx = np.arange (self.iterator, iterator_1)
        self.iterator += batch_size
        ret = self._get_subdata(self.batch_set, idx)
        return ret

    def get_test (self, 
                  ntests : int = -1
    ) -> dict:
        """
        Get the test data with `ntests` frames. 

        Parameters
        ----------
        ntests
                Size of the test data set. If `ntests` is -1, all test data will be get.
        """
        if not hasattr(self, 'test_set') :            
            self._load_test_set(self.test_dir, self.shuffle_test)
        if ntests == -1:
            idx = None
        else :
            ntests_ = ntests if ntests < self.test_set['type'].shape[0] else self.test_set['type'].shape[0]
            # print('ntest', self.test_set['type'].shape[0], ntests, ntests_)
            idx = np.arange(ntests_)
        ret = self._get_subdata(self.test_set, idx = idx)
        if self.modifier is not None:
            self.modifier.modify_data(ret)
        return ret

    def get_ntypes(self) -> int:
        """
        Number of atom types in the system
        """
        if self.type_map is not None:
            return len(self.type_map)
        else:
            return max(self.get_atom_type()) + 1

    def get_type_map(self) -> List[str]:
        """
        Get the type map
        """
        return self.type_map

    def get_atom_type(self) -> List[int]:
        """
        Get atom types
        """
        return self.atom_type

    def get_numb_set (self) -> int:
        """
        Get number of training sets
        """
        return len (self.train_dirs)

    def get_numb_batch (self, 
                        batch_size : int, 
                        set_idx : int
    ) -> int:
        """
        Get the number of batches in a set.
        """
        data = self._load_set(self.train_dirs[set_idx])
        ret = data["coord"].shape[0] // batch_size
        if ret == 0:
            ret = 1
        return ret

    def get_sys_numb_batch (self, 
                            batch_size : int
    ) -> int:
        """
        Get the number of batches in the data system.
        """
        ret = 0
        for ii in range(len(self.train_dirs)) :
            ret += self.get_numb_batch(batch_size, ii)
        return ret

    def get_natoms (self) :
        """
        Get number of atoms
        """
        return len(self.atom_type)

    def get_natoms_vec (self,
                        ntypes : int) :
        """
        Get number of atoms and number of atoms in different types

        Parameters
        ----------
        ntypes
                Number of types (may be larger than the actual number of types in the system).

        Returns
        -------
        natoms
                natoms[0]: number of local atoms
                natoms[1]: total number of atoms held by this processor
                natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        """
        natoms, natoms_vec = self._get_natoms_2 (ntypes)
        tmp = [natoms, natoms]
        tmp = np.append (tmp, natoms_vec)
        return tmp.astype(np.int32)
    
    def avg(self, key) :
        """
        Return the average value of an item.
        """
        if key not in self.data_dict.keys() :
            raise RuntimeError('key %s has not been added' % key)
        info = self.data_dict[key]  
        ndof = info['ndof']
        eners = np.array([])
        for ii in self.train_dirs:
            data = self._load_set(ii)
            ei = data[key].reshape([-1, ndof])
            if eners.size  == 0 :
                eners = ei
            else :
                eners = np.concatenate((eners, ei), axis = 0)
        if eners.size == 0 :
            return 0
        else :
            return np.average(eners, axis = 0)

    def _idx_map_sel(self, atom_type, type_sel) :
        new_types = []
        for ii in atom_type :
            if ii in type_sel:
                new_types.append(ii)
        new_types = np.array(new_types, dtype = int)
        natoms = new_types.shape[0]
        idx = np.arange(natoms)
        idx_map = np.lexsort((idx, new_types))
        return idx_map

    def _get_natoms_2 (self, ntypes) :
        sample_type = self.atom_type
        natoms = len(sample_type)
        natoms_vec = np.zeros (ntypes).astype(int)
        for ii in range (ntypes) :
            natoms_vec[ii] = np.count_nonzero(sample_type == ii)
        return natoms, natoms_vec

    def _get_subdata(self, data, idx = None) :
        new_data = {}
        for ii in data:
            dd = data[ii]
            if 'find_' in ii:
                new_data[ii] = dd                
            else:
                if idx is not None:
                    new_data[ii] = dd[idx]
                else :
                    new_data[ii] = dd
        return new_data

    def _load_batch_set (self,
                         set_name) :
        self.batch_set = self._load_set(set_name)
        self.batch_set, _ = self._shuffle_data(self.batch_set)
        self.reset_get_batch()

    def reset_get_batch(self):
        self.iterator = 0

    def _load_test_set (self,
                       set_name, 
                       shuffle_test) :
        self.test_set = self._load_set(set_name)        
        if shuffle_test :
            self.test_set, _ = self._shuffle_data(self.test_set)

    def _shuffle_data (self,
                       data) :
        ret = {}
        nframes = data['coord'].shape[0]
        idx = np.arange (nframes)
        np.random.shuffle (idx)
        for kk in data :
            if type(data[kk]) == np.ndarray and \
               len(data[kk].shape) == 2 and \
               data[kk].shape[0] == nframes and \
               not('find_' in kk) and \
               'type' != kk:
                ret[kk] = data[kk][idx]
            else :
                ret[kk] = data[kk]
        return ret, idx

    def _load_set(self, set_name) :
        # get nframes
        path = os.path.join(set_name, "coord.npy")
        if self.data_dict['coord']['high_prec'] :
            coord = np.load(path).astype(GLOBAL_ENER_FLOAT_PRECISION)
        else:
            coord = np.load(path).astype(GLOBAL_NP_FLOAT_PRECISION)            
        if coord.ndim == 1:
            coord = coord.reshape([1,-1])
        nframes = coord.shape[0]
        assert(coord.shape[1] == self.data_dict['coord']['ndof'] * self.natoms)
        # load keys
        data = {}
        data['type'] = np.tile (self.atom_type[self.idx_map], (nframes, 1))
        for kk in self.data_dict.keys():
            if self.data_dict[kk]['reduce'] is None :
                data['find_'+kk], data[kk] \
                    = self._load_data(set_name, 
                                      kk, 
                                      nframes, 
                                      self.data_dict[kk]['ndof'],
                                      atomic = self.data_dict[kk]['atomic'],
                                      high_prec = self.data_dict[kk]['high_prec'],
                                      must = self.data_dict[kk]['must'], 
                                      type_sel = self.data_dict[kk]['type_sel'],
                                      repeat = self.data_dict[kk]['repeat'])
        for kk in self.data_dict.keys():
            if self.data_dict[kk]['reduce'] is not None :
                k_in = self.data_dict[kk]['reduce']
                ndof = self.data_dict[kk]['ndof']
                data['find_'+kk] = data['find_'+k_in]
                tmp_in = data[k_in].astype(GLOBAL_ENER_FLOAT_PRECISION)
                data[kk] = np.sum(np.reshape(tmp_in, [nframes, self.natoms, ndof]), axis = 1)

        return data


    def _load_data(self, set_name, key, nframes, ndof_, atomic = False, must = True, repeat = 1, high_prec = False, type_sel = None):
        if atomic:
            natoms = self.natoms
            idx_map = self.idx_map
            # if type_sel, then revise natoms and idx_map
            if type_sel is not None:
                natoms = 0
                for jj in type_sel :
                    natoms += np.sum(self.atom_type == jj)                
                idx_map = self._idx_map_sel(self.atom_type, type_sel)
            ndof = ndof_ * natoms
        else:
            ndof = ndof_
        path = os.path.join(set_name, key+".npy")
        if os.path.isfile (path) :
            if high_prec :
                data = np.load(path).astype(GLOBAL_ENER_FLOAT_PRECISION)
            else:
                data = np.load(path).astype(GLOBAL_NP_FLOAT_PRECISION)
            if atomic :
                data = data.reshape([nframes, natoms, -1])
                data = data[:,idx_map,:]
                data = data.reshape([nframes, -1])
            data = np.reshape(data, [nframes, ndof])
            if repeat != 1:
                data = np.repeat(data, repeat).reshape([nframes, -1])
            return np.float32(1.0), data
        elif must:
            raise RuntimeError("%s not found!" % path)
        else:
            if high_prec :
                data = np.zeros([nframes,ndof]).astype(GLOBAL_ENER_FLOAT_PRECISION)                
            else :
                data = np.zeros([nframes,ndof]).astype(GLOBAL_NP_FLOAT_PRECISION)
            if repeat != 1:
                data = np.repeat(data, repeat).reshape([nframes, -1])
            return np.float32(0.0), data

        
    def _load_type (self, sys_path) :
        atom_type = np.loadtxt (os.path.join(sys_path, "type.raw"), dtype=np.int32, ndmin=1)
        return atom_type

    def _make_idx_map(self, atom_type):
        natoms = atom_type.shape[0]
        idx = np.arange (natoms)
        idx_map = np.lexsort ((idx, atom_type))
        return idx_map

    def _load_type_map(self, sys_path) :
        fname = os.path.join(sys_path, 'type_map.raw')
        if os.path.isfile(fname) :            
            with open(os.path.join(sys_path, 'type_map.raw')) as fp:
                return fp.read().split()                
        else :
            return None

    def _check_pbc(self, sys_path):
        pbc = True
        if os.path.isfile(os.path.join(sys_path, 'nopbc')) :
            pbc = False
        return pbc


class DataSets (object):
    """
    Outdated class for one data system. Not maintained anymore.
    """
    def __init__ (self, 
                  sys_path,
                  set_prefix,
                  seed = None, 
                  shuffle_test = True) :
        self.dirs = glob.glob (os.path.join(sys_path, set_prefix + ".*"))
        self.dirs.sort()
        # load atom type
        self.atom_type, self.idx_map, self.idx3_map = self.load_type (sys_path)
        # load atom type map
        self.type_map = self.load_type_map(sys_path)
        if self.type_map is not None:
            assert(len(self.type_map) >= max(self.atom_type)+1)
        # train dirs
        self.test_dir   = self.dirs[-1]
        if len(self.dirs) == 1 :
            self.train_dirs = self.dirs
        else :
            self.train_dirs = self.dirs[:-1]
        # check fparam
        has_fparam = [ os.path.isfile(os.path.join(ii, 'fparam.npy')) for ii in self.dirs ]
        if any(has_fparam) and (not all(has_fparam)) :
            raise RuntimeError("system %s: if any set has frame parameter, then all sets should have frame parameter" % sys_path)
        if all(has_fparam) :
            self.has_fparam = 0
        else :
            self.has_fparam = -1
        # check aparam
        has_aparam = [ os.path.isfile(os.path.join(ii, 'aparam.npy')) for ii in self.dirs ]
        if any(has_aparam) and (not all(has_aparam)) :
            raise RuntimeError("system %s: if any set has frame parameter, then all sets should have frame parameter" % sys_path)
        if all(has_aparam) :
            self.has_aparam = 0
        else :
            self.has_aparam = -1
        # energy norm
        self.eavg = self.stats_energy()
        # load sets
        self.set_count = 0
        self.load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
        self.load_test_set (self.test_dir, shuffle_test)

    def check_batch_size (self, batch_size) :
        for ii in self.train_dirs :
            tmpe = np.load(os.path.join(ii, "coord.npy"))
            if tmpe.shape[0] < batch_size :
                return ii, tmpe.shape[0]
        return None

    def check_test_size (self, test_size) :
        tmpe = np.load(os.path.join(self.test_dir, "coord.npy"))
        if tmpe.shape[0] < test_size :
            return self.test_dir, tmpe.shape[0]
        else :
            return None

    def load_type (self, sys_path) :
        atom_type = np.loadtxt (os.path.join(sys_path, "type.raw"), dtype=np.int32, ndmin=1)
        natoms = atom_type.shape[0]
        idx = np.arange (natoms)
        idx_map = np.lexsort ((idx, atom_type))
        atom_type3 = np.repeat(atom_type, 3)
        idx3 = np.arange (natoms * 3)
        idx3_map = np.lexsort ((idx3, atom_type3))
        return atom_type, idx_map, idx3_map

    def load_type_map(self, sys_path) :
        fname = os.path.join(sys_path, 'type_map.raw')
        if os.path.isfile(fname) :            
            with open(os.path.join(sys_path, 'type_map.raw')) as fp:
                return fp.read().split()                
        else :
            return None

    def get_type_map(self) :
        return self.type_map

    def get_numb_set (self) :
        return len (self.train_dirs)
    
    def stats_energy (self) :
        eners = np.array([])
        for ii in self.train_dirs:
            ener_file = os.path.join(ii, "energy.npy")
            if os.path.isfile(ener_file) :
                ei = np.load(ener_file)
                eners = np.append(eners, ei)
        if eners.size == 0 :
            return 0
        else :
            return np.average(eners)

    def load_energy(self, 
                    set_name,
                    nframes,
                    nvalues,
                    energy_file, 
                    atom_energy_file) :
        """
        return : coeff_ener, ener, coeff_atom_ener, atom_ener
        """
        # load atom_energy
        coeff_atom_ener, atom_ener = self.load_data(set_name, atom_energy_file, [nframes, nvalues], False)
        # ignore energy_file
        if coeff_atom_ener == 1:
            ener = np.sum(atom_ener, axis = 1)
            coeff_ener = 1
        # load energy_file
        else:
            coeff_ener, ener = self.load_data(set_name, energy_file, [nframes], False)
        return coeff_ener, ener, coeff_atom_ener, atom_ener

    def load_data(self, set_name, data_name, shape, is_necessary = True):
        path = os.path.join(set_name, data_name+".npy")
        if os.path.isfile (path) :
            data = np.load(path)
            data = np.reshape(data, shape)
            if is_necessary:
                return data
            return 1, data
        elif is_necessary:
            raise OSError("%s not found!" % path)
        else:
            data = np.zeros(shape)
        return 0, data

    def load_set(self, set_name, shuffle = True):
        data = {}
        data["box"] = self.load_data(set_name, "box", [-1, 9])
        nframe = data["box"].shape[0]
        data["coord"] = self.load_data(set_name, "coord", [nframe, -1])
        ncoord = data["coord"].shape[1]
        if self.has_fparam >= 0:
            data["fparam"] = self.load_data(set_name, "fparam", [nframe, -1])
            if self.has_fparam == 0 :
                self.has_fparam = data["fparam"].shape[1]
            else :
                assert self.has_fparam == data["fparam"].shape[1]
        if self.has_aparam >= 0:
            data["aparam"] = self.load_data(set_name, "aparam", [nframe, -1])
            if self.has_aparam == 0 :
                self.has_aparam = data["aparam"].shape[1] // (ncoord//3)
            else :
                assert self.has_aparam == data["aparam"].shape[1] // (ncoord//3)
        data["prop_c"] = np.zeros(5)
        data["prop_c"][0], data["energy"], data["prop_c"][3], data["atom_ener"] \
            = self.load_energy (set_name, nframe, ncoord // 3, "energy", "atom_ener")
        data["prop_c"][1], data["force"] = self.load_data(set_name, "force", [nframe, ncoord], False)
        data["prop_c"][2], data["virial"] = self.load_data(set_name, "virial", [nframe, 9], False)
        data["prop_c"][4], data["atom_pref"] = self.load_data(set_name, "atom_pref", [nframe, ncoord//3], False)
        data["atom_pref"] = np.repeat(data["atom_pref"], 3, axis=1)
        # shuffle data
        if shuffle:
            idx = np.arange (nframe)
            np.random.shuffle (idx)
            for ii in data:
                if ii != "prop_c":
                    data[ii] = data[ii][idx]
        data["type"] = np.tile (self.atom_type, (nframe, 1))
        # sort according to type
        for ii in ["type", "atom_ener"]:
            data[ii] = data[ii][:, self.idx_map]
        for ii in ["coord", "force", "atom_pref"]:
            data[ii] = data[ii][:, self.idx3_map]
        return data

    def load_batch_set (self,
                        set_name) :
        self.batch_set = self.load_set(set_name, True)
        self.reset_iter ()

    def load_test_set (self,
                       set_name, 
                       shuffle_test) :
        self.test_set = self.load_set(set_name, shuffle_test)
        
    def reset_iter (self) :
        self.iterator = 0              
        self.set_count += 1
    
    def get_set(self, data, idx = None) :
        new_data = {}
        for ii in data:
            dd = data[ii]
            if ii == "prop_c":
                new_data[ii] = dd.astype(np.float32)
            else:
                if idx is not None:
                    dd = dd[idx]
                if ii == "type":
                    new_data[ii] = dd
                else:
                    new_data[ii] = dd.astype(GLOBAL_NP_FLOAT_PRECISION)
        return new_data

    def get_test (self) :
        """
        returned property prefector [4] in order: 
        energy, force, virial, atom_ener
        """
        return self.get_set(self.test_set)
    
    def get_batch (self,
                   batch_size) :
        """
        returned property prefector [4] in order: 
        energy, force, virial, atom_ener
        """
        set_size = self.batch_set["energy"].shape[0]
        # assert (batch_size <= set_size), "batch size should be no more than set size"
        if self.iterator + batch_size > set_size :
            self.load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
            set_size = self.batch_set["energy"].shape[0]
        # print ("%d %d %d" % (self.iterator, self.iterator + batch_size, set_size))
        iterator_1 = self.iterator + batch_size
        if iterator_1 >= set_size :
            iterator_1 = set_size
        idx = np.arange (self.iterator, iterator_1)
        self.iterator += batch_size
        return self.get_set(self.batch_set, idx)
    
    def get_natoms (self) :
        sample_type = self.batch_set["type"][0]
        natoms = len(sample_type)
        return natoms

    def get_natoms_2 (self, ntypes) :
        sample_type = self.batch_set["type"][0]
        natoms = len(sample_type)
        natoms_vec = np.zeros (ntypes).astype(int)
        for ii in range (ntypes) :
            natoms_vec[ii] = np.count_nonzero(sample_type == ii)
        return natoms, natoms_vec

    def get_natoms_vec (self, ntypes) :
        natoms, natoms_vec = self.get_natoms_2 (ntypes)
        tmp = [natoms, natoms]
        tmp = np.append (tmp, natoms_vec)
        return tmp.astype(np.int32)

    def set_numb_batch (self, 
                        batch_size) :
        return self.batch_set["energy"].shape[0] // batch_size

    def get_sys_numb_batch (self, batch_size) :
        return self.set_numb_batch(batch_size) * self.get_numb_set()

    def get_ener (self) :
        return self.eavg

    def numb_fparam(self) :
        return self.has_fparam

    def numb_aparam(self) :
        return self.has_aparam

