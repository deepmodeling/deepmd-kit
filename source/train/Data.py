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
                  seed = None) :
        self.dirs = glob.glob (os.path.join(sys_path, set_prefix + ".*"))
        self.dirs.sort()
        # load atom type
        self.atom_type, self.idx_map, self.idx3_map = self.load_type (sys_path)
        # train dirs
        self.test_dir   = self.dirs[-1]
        if len(self.dirs) == 1 :
            self.train_dirs = self.dirs
        else :
            self.train_dirs = self.dirs[:-1]
        # energy norm
        self.eavg = self.stats_energy()
        # load sets
        self.set_count = 0
        self.load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
        self.load_test_set (self.test_dir)

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
        atom_type = np.loadtxt (sys_path + "/type.raw", dtype=np.int32)
        if atom_type.shape == () :
            atom_type = np.array([atom_type])
        natoms = atom_type.shape[0]
        idx = np.arange (natoms)
        idx_map = np.lexsort ((idx, atom_type))
        atom_type3 = np.array([atom_type[ii//3] for ii in range (natoms * 3)])
        idx3 = np.arange (natoms * 3)
        idx3_map = np.lexsort ((idx3, atom_type3))
        return atom_type, idx_map, idx3_map

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
    
    def cond_load_vec (self, 
                       nframes,
                       file_name) :
        coeff = 0
        if os.path.isfile (file_name) :
            data = np.load (file_name)
            data = np.reshape (data, [nframes])
            coeff = 1
        else :
            data = np.zeros ([nframes])
            coeff = 0
        return coeff, data 

    def cond_load_mat (self, 
                       nframes,
                       nvalues,
                       file_name) :
        coeff = 0
        if os.path.isfile (file_name) :
            data = np.load (file_name)
            data = np.reshape (data, [nframes, nvalues])
            coeff = 1
        else :
            data = np.zeros ([nframes, nvalues])
            coeff = 0
        return coeff, data

    def load_energy(self, 
                    nframes, 
                    nvalues,
                    energy_file, 
                    atom_energy_file) :
        """
        return : coeff_ener, ener, coeff_atom_ener, atom_ener
        """
        coeff_ener = 0
        coeff_atom_ener = 0
        ener = None        
        atom_ener = None
        # load atom_energy and ignore energy_file
        if os.path.isfile(atom_energy_file) :
            atom_ener = np.load(atom_energy_file)
            atom_ener = np.reshape(atom_ener, [nframes, nvalues])
            ener = np.sum(atom_ener, axis = 1)
            coeff_atom_ener = 1.
            coeff_ener = 1.
        # load energy_file
        elif os.path.isfile(energy_file) :
            coeff_ener, ener = self.cond_load_vec(nframes, energy_file)
            atom_ener = np.zeros([nframes, nvalues])
            coeff_atom_ener = 0.
        else :
            atom_ener = np.zeros([nframes, nvalues])
            ener = np.zeros([nframes])
            coeff_atom_ener = 0.
            coeff_ener = 0.
        return coeff_ener, ener, coeff_atom_ener, atom_ener

    def load_batch_set (self,
                        set_name) :
        start_time = time.time()
        self.coord_batch = np.load(os.path.join(set_name, "coord.npy"))
        self.box_batch = np.load(os.path.join(set_name, "box.npy"))
        self.box_batch = np.reshape(self.box_batch, [-1, 9])
        nframe = self.box_batch.shape[0]
        self.coord_batch = np.reshape(self.coord_batch, [nframe, -1])
        ncoord = self.coord_batch.shape[1]        
        self.prop_c_batch = np.zeros (4)
        self.prop_c_batch[0], self.energy_batch, self.prop_c_batch[3], self.atom_ener_batch \
            = self.load_energy (nframe, ncoord // 3,
                                os.path.join(set_name, "energy.npy"), 
                                os.path.join(set_name, "atom_ener.npy")
            )
        self.prop_c_batch[1], self.force_batch \
            = self.cond_load_mat (nframe, ncoord, 
                                  os.path.join(set_name, "force.npy"))
        self.prop_c_batch[2], self.virial_batch \
            = self.cond_load_mat (nframe, 9, 
                                  os.path.join(set_name, "virial.npy"))
        # shuffle data
        idx = np.arange (nframe)
        np.random.shuffle (idx)
        self.energy_batch = self.energy_batch[idx]
        self.force_batch = self.force_batch[idx]
        self.virial_batch = self.virial_batch[idx]
        self.atom_ener_batch = self.atom_ener_batch[idx]
        self.coord_batch = self.coord_batch[idx]
        self.box_batch = self.box_batch[idx]
        self.type_batch = np.tile (self.atom_type, (nframe, 1))
        self.reset_iter ()
        # sort according to type
        self.type_batch = self.type_batch[:, self.idx_map]
        self.atom_ener_batch = self.atom_ener_batch[:, self.idx_map]
        self.coord_batch = self.coord_batch[:, self.idx3_map]
        self.force_batch = self.force_batch[:, self.idx3_map]
        end_time = time.time()

    def load_test_set (self,
                       set_name) :
        start_time = time.time()
        self.coord_test = np.load(os.path.join(set_name, "coord.npy"))
        self.box_test = np.load(os.path.join(set_name, "box.npy"))
        self.box_test = np.reshape(self.box_test, [-1, 9])
        nframe = self.box_test.shape[0]
        self.coord_test = np.reshape(self.coord_test, [nframe, -1])
        ncoord = self.coord_test.shape[1]
        self.prop_c_test = np.zeros (4)
        self.prop_c_test[0], self.energy_test, self.prop_c_test[3], self.atom_ener_test \
            = self.load_energy (nframe, ncoord // 3,
                                os.path.join(set_name, "energy.npy"), 
                                os.path.join(set_name, "atom_ener.npy")
            )
        self.prop_c_test[1], self.force_test \
            = self.cond_load_mat (nframe, ncoord, 
                                  os.path.join(set_name, "force.npy"))
        self.prop_c_test[2], self.virial_test \
            = self.cond_load_mat (nframe, 9, 
                                  os.path.join(set_name, "virial.npy"))
        # shuffle data
        idx = np.arange (nframe)
        np.random.shuffle (idx)
        self.energy_test = self.energy_test[idx]
        self.force_test = self.force_test[idx]
        self.virial_test = self.virial_test[idx]
        self.atom_ener_test = self.atom_ener_test[idx]
        self.coord_test = self.coord_test[idx]
        self.box_test = self.box_test[idx]
        self.type_test = np.tile (self.atom_type, (nframe, 1))
        # sort according to type
        self.type_test = self.type_test[:, self.idx_map]
        self.atom_ener_test = self.atom_ener_test[:, self.idx_map]
        self.coord_test = self.coord_test[:, self.idx3_map]
        self.force_test = self.force_test[:, self.idx3_map]
        end_time = time.time()
        
    def reset_iter (self) :
        self.iterator = 0              
        self.set_count += 1
        
    def get_test (self) :
        """
        returned property prefector [4] in order: 
        energy, force, virial, atom_ener
        """
        return \
            self.prop_c_test.astype(np.float32), \
            self.energy_test.astype(global_np_float_precision), \
            self.force_test.astype(global_np_float_precision), \
            self.virial_test.astype(global_np_float_precision), \
            self.atom_ener_test.astype(global_np_float_precision), \
            self.coord_test.astype(global_np_float_precision), \
            self.box_test.astype(global_np_float_precision), \
            self.type_test
    
    def get_batch (self,
                   batch_size) :
        """
        returned property prefector [4] in order: 
        energy, force, virial, atom_ener
        """
        set_size = self.energy_batch.shape[0]
        # assert (batch_size <= set_size), "batch size should be no more than set size"
        if self.iterator + batch_size > set_size :
            self.load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
            set_size = self.energy_batch.shape[0]
        # print ("%d %d %d" % (self.iterator, self.iterator + batch_size, set_size))
        iterator_1 = self.iterator + batch_size
        if iterator_1 >= set_size :
            iterator_1 = set_size
        idx = np.arange (self.iterator, iterator_1)
        self.iterator += batch_size
        return \
            self.prop_c_batch.astype(np.float32), \
            self.energy_batch[idx].astype(global_np_float_precision), \
            self.force_batch[idx, :].astype(global_np_float_precision), \
            self.virial_batch[idx, :].astype(global_np_float_precision), \
            self.atom_ener_batch[idx, :].astype(global_np_float_precision), \
            self.coord_batch[idx, :].astype(global_np_float_precision), \
            self.box_batch[idx, :].astype(global_np_float_precision), \
            self.type_batch[idx, :]
    
    def get_natoms (self) :
        sample_type = self.type_batch[0]
        natoms = len(sample_type)
        return natoms

    def get_natoms_2 (self, ntypes) :
        sample_type = self.type_batch[0]
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
        return self.energy_batch.shape[0] // batch_size

    def get_sys_numb_batch (self, batch_size) :
        return self.set_numb_batch(batch_size) * self.get_numb_set()

    def get_ener (self) :
        return self.eavg

if __name__ == '__main__':
    data = DataSets (".", "set")
    prop_c, energy, force, virial, atom_ener, coord, box, ttype = data.get_batch(1)
    print (energy.shape)
    print (force.shape)
    print (coord.shape)
    print (box.shape)
    print (ttype.shape)
    # energy, force, coord, box, ttype = data.get_test()
    print (energy)
    
