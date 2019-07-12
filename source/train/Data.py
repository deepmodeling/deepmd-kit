#!/usr/bin/env python3

import time
import glob
import random
import numpy as np
import os.path
from deepmd.RunOptions import global_tf_float_precision
from deepmd.RunOptions import global_np_float_precision

class DataSets (object):
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
        atom_ener, coeff_atom_ener = self.load_data(set_name, atom_energy_file, [nframes, nvalues], False)
        # ignore energy_file
        if coeff_atom_ener == 1:
            ener = np.sum(atom_ener, axis = 1)
            coeff_atom_ener = 1
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
            return data, 1
        elif is_necessary:
            raise OSError("%s not found!" % path)
        else:
            data = np.zeros(shape)
        return data, 0

    def load_set(self, set_name, shuffle = True):
        start_time = time.time()
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
        data["prop_c"] = np.zeros(5)
        data["prop_c"][0], data["energy"], data["prop_c"][3], data["atom_ener"] \
            = self.load_energy (set_name, nframe, ncoord // 3, "energy", "atom_ener")
        data["prop_c"][1], data["force"] = self.load_data(set_name, "force", [nframe, ncoord], False)
        data["prop_c"][2], data["virial"] = self.load_data(set_name, "virial", [nframe, 9], False)
        data["prop_c"][4], data["atom_pref"] = self.load_data(set_name, "atom_pref", [nframe, ncoord//3], False)
        # shuffle data
        if shuffle:
            idx = np.arange (nframe)
            np.random.shuffle (idx)
            for ii in data:
                if ii != "prop_c":
                    data[ii] = data[ii][idx]
        data["type"] = np.tile (self.atom_type, (nframe, 1))
        # sort according to type
        for ii in ["type", "atom_ener", "atom_pref"]:
            data[ii] = data[ii][:, self.idx_map]
        for ii in ["coord", "force"]:
            data[ii] = data[ii][:, self.idx3_map]
        end_time = time.time()
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
                    new_data[ii] = dd.astype(global_np_float_precision)

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

