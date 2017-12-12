#!/usr/bin/env python3

import time
import glob
import random
import numpy as np
import os.path

class DataScan (object) :
    def __init__ (self, 
                  sys_path,
                  set_prefix) :

        self.dirs = glob.glob (sys_path + "/" + set_prefix + ".*")
        self.dirs.sort()        
        if os.path.isfile (sys_path + "/ncopies.raw") :
            self.ncopies = np.loadtxt (sys_path + "/ncopies.raw", dtype=np.int32)
        else :
            self.ncopies = [1]
        e = np.load (self.dirs[0] + "/energy.npy")
        self.set_ndata = e.shape[0]        

    def get_set_numb_batch (self, batch_size) :
        return self.set_ndata // batch_size

    def get_numb_train_set (self) :
        return ( len(self.dirs) - 1 )
    
    def get_sys_numb_batch (self, batch_size) :
        return self.get_set_numb_batch(batch_size) * self.get_numb_train_set()

    def get_ncopies (self) :
        return self.ncopies                           

class DataSets (object):
    def __init__ (self, 
                  sys_path,
                  set_prefix,
                  do_norm = False,
                  seed = None) :

        np.random.seed (seed)
        self.dirs = glob.glob (sys_path + "/" + set_prefix + ".*")
        self.dirs.sort()
        if os.path.isfile (sys_path + "/ncopies.raw") :
            self.ncopies = np.loadtxt (sys_path + "/ncopies.raw", dtype=np.int32)
        else :
            self.ncopies = [1]

        # load atom type
        self.atom_type, self.idx_map, self.idx3_map = self.load_type (sys_path)
        
        self.do_norm = do_norm
        self.eavg = self.stats_energy()
        # print ("avg is ",self.eavg)

        self.test_dir   = self.dirs[-1]
        if len(self.dirs) == 1 :
            self.train_dirs = self.dirs
        else :
            self.train_dirs = self.dirs[:-1]
        self.set_count = 0
        self.load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
        self.load_test_set (self.test_dir)

    def load_type (self, sys_path) :
        atom_type = np.loadtxt (sys_path + "/type.raw", dtype=np.int32)
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
        eners = []
        for ii in self.dirs:
            ei = np.load (ii + "/energy.npy")
            eners.append (np.average(ei))
        return np.average (eners)
    
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

    def load_batch_set (self,
                        set_name) :
        start_time = time.time()
        self.coord_batch = np.load (set_name + "/coord.npy")
        self.box_batch = np.load (set_name + "/box.npy")
        nframe = self.box_batch.shape[0]
        ncoord = self.coord_batch.shape[1]
        self.prop_c_batch = np.zeros (3)
        self.prop_c_batch[0], self.energy_batch = self.cond_load_vec (nframe, set_name + "/energy.npy")
        self.prop_c_batch[1], self.force_batch = self.cond_load_mat (nframe, ncoord, set_name + "/force.npy")
        self.prop_c_batch[2], self.virial_batch = self.cond_load_mat (nframe, 9, set_name + "/virial.npy")
        # shuffle data
        idx = np.arange (nframe)
        np.random.shuffle (idx)
        self.energy_batch = self.energy_batch[idx]
        self.force_batch = self.force_batch[idx]
        self.virial_batch = self.virial_batch[idx]
        self.coord_batch = self.coord_batch[idx]
        self.box_batch = self.box_batch[idx]
        self.type_batch = np.tile (self.atom_type, (nframe, 1))
        self.reset_iter ()
        if self.do_norm: self.normalization (self.energy_batch)
        # sort according to type
        self.type_batch = self.type_batch[:, self.idx_map]
        self.coord_batch = self.coord_batch[:, self.idx3_map]
        self.force_batch = self.force_batch[:, self.idx3_map]
        end_time = time.time()

    def load_test_set (self,
                       set_name) :
        start_time = time.time()
        self.coord_test = np.load (set_name + "/coord.npy")
        self.box_test = np.load (set_name + "/box.npy")
        nframe = self.box_test.shape[0]
        ncoord = self.coord_test.shape[1]
        self.prop_c_test = np.zeros (3)
        self.prop_c_test[0], self.energy_test = self.cond_load_vec (nframe, set_name + "/energy.npy")
        self.prop_c_test[1], self.force_test = self.cond_load_mat (nframe, ncoord, set_name + "/force.npy")
        self.prop_c_test[2], self.virial_test = self.cond_load_mat (nframe, 9, set_name + "/virial.npy")
        # shuffle data
        idx = np.arange (nframe)
        np.random.shuffle (idx)
        self.energy_test = self.energy_test[idx]
        self.force_test = self.force_test[idx]
        self.virial_test = self.virial_test[idx]
        self.coord_test = self.coord_test[idx]
        self.box_test = self.box_test[idx]
        self.type_test = np.tile (self.atom_type, (nframe, 1))
        if self.do_norm: self.normalization (self.energy_test)
        self.type_test = self.type_test[:, self.idx_map]
        self.coord_test = self.coord_test[:, self.idx3_map]
        self.force_test = self.force_test[:, self.idx3_map]
        end_time = time.time()
        
    def reset_iter (self) :
        self.iterator = 0              
        self.set_count += 1
        
    def get_test (self) :
        return self.prop_c_test.astype(np.float32), self.energy_test.astype(np.float64), self.force_test.astype(np.float64), self.virial_test.astype(np.float64), self.coord_test.astype(np.float64), self.box_test.astype(np.float64), self.type_test
    
    def get_batch (self,
                   batch_size) :
        set_size = self.energy_batch.shape[0]
        assert (batch_size <= set_size), "batch size should be no more than set size"
        if self.iterator + batch_size > set_size :
            self.load_batch_set (self.train_dirs[self.set_count % self.get_numb_set()])
        # print ("%d %d %d" % (self.iterator, self.iterator + batch_size, set_size))
        idx = np.arange (self.iterator, self.iterator + batch_size)
        self.iterator += batch_size
        return self.prop_c_batch.astype(np.float32), self.energy_batch[idx].astype(np.float64), self.force_batch[idx, :].astype(np.float64), self.virial_batch[idx, :].astype(np.float64), self.coord_batch[idx, :].astype(np.float64), self.box_batch[idx, :].astype(np.float64), self.type_batch[idx, :]
    
    def normalization (self, 
                       energy) :
        if self.do_norm : 
            for ii in range (energy.shape[0]) :
                energy[ii] -= self.eavg

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

    def get_ncopies (self) :
        return self.ncopies

    def set_numb_batch (self, 
                        batch_size) :
        return self.energy_batch.shape[0] // batch_size

    def get_sys_numb_batch (self, batch_size) :
        return self.set_numb_batch(batch_size) * self.get_numb_set()

    def get_bias_atom_e (self) :
        natoms = self.get_natoms ()
        return self.eavg / natoms

if __name__ == '__main__':
    data = DataSets (".", "set", do_norm = False)
    energy, force, virial, coord, box, ttype = data.get_batch(1)
    print (energy.shape)
    print (force.shape)
    print (coord.shape)
    print (box.shape)
    print (ttype.shape)
    # energy, force, coord, box, ttype = data.get_test()
    print (energy)
    
