import os,sys,shutil,copy
import numpy as np
import unittest

from deepmd.utils.data_system import DeepmdDataSystem,DeepmdDataDocker
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION

if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    places = 6
else:
    places = 12

class TestDataDocker (unittest.TestCase) :
    def setUp (self) :
        self.nmethod = 2
        self.dsmt_list = []
        self.nframes = [3, 6, 5, 4]
        self.natoms = [3, 4, 6, 5]
        self.atom_type = [[1, 0, 0], 
                          [2, 1, 0, 2],
                          [0, 0, 1, 1, 2, 1],
                          [0, 2, 2, 0, 0]]
        self.test_ndof = 2
        self.nsys = 4
        self.nset = 3
        for method in range(self.nmethod):
            nname = 'method_%d' % method
            os.makedirs(nname, exist_ok = True)
            nbatch_size = 1
            ntest_size = 1
            sys_name_total = []
            for ii in range(self.nsys) :
                sys_name = os.path.join(nname, 'sys_%d' % ii)
                sys_name_total.append(sys_name)
                os.makedirs(sys_name, exist_ok = True)
                np.savetxt(os.path.join(sys_name, 'type.raw'), 
                       self.atom_type[ii], 
                       fmt = '%d')
                for jj in range(self.nset):
                    set_name = os.path.join(sys_name, 'set.%03d' % jj)
                    os.makedirs(set_name, exist_ok = True)
                    path = os.path.join(set_name, 'coord.npy')
                    val = np.random.random([self.nframes[ii]+jj, self.natoms[ii]*3])
                    np.save(path, val)
                    path = os.path.join(set_name, 'box.npy')
                    val = np.random.random([self.nframes[ii]+jj, 9]) * 10
                    np.save(path, val)
                    path = os.path.join(set_name, 'test.npy')
                    val = np.random.random([self.nframes[ii]+jj, self.natoms[ii]*self.test_ndof])
                    np.save(path, val)
            ds = DeepmdDataSystem(sys_name_total, nbatch_size, ntest_size, 2.0, name = nname)
            self.dsmt_list.append(ds)
        
    def tearDown(self):
        for method in range(self.nmethod):
            nname = 'method_%d' % method
            shutil.rmtree(nname)    
                      

    def test_ntypes(self) :
        batch_size = 2
        ds = DeepmdDataDocker(self.dsmt_list, batch_size)
        self.assertEqual(ds.get_nmethod(), self.nmethod)
        self.assertEqual(ds.get_nbatches(), [7, 13, 11, 9, 7, 13, 11, 9])
        self.assertEqual(ds.get_name(), ['method_0','method_1'])
        self.assertEqual(list(ds.get_batch_size()), [1, 1, 1, 1, 1, 1, 1, 1])
        
    def test_get_data_system(self):
        batch_size = 2
        ds = DeepmdDataDocker(self.dsmt_list, batch_size)
        pick_idx = 0
        method_name = 'method_%d' % pick_idx
        self.assertEqual(ds.get_data_system_idx(pick_idx).get_name(), method_name)
        self.assertEqual(ds.get_data_system(method_name).get_name(), method_name)

    def test_get_batch(self):
        batch_size = 2
        ds = DeepmdDataDocker(self.dsmt_list, batch_size)
        for i in range(self.nmethod):
            ds.get_data_system_idx(i).add('test', self.test_ndof, atomic = True, must = True)
            ds.get_data_system_idx(i).add('null', self.test_ndof, atomic = True, must = False)
        method_idx = 0
        sys_idx = 0
        data = ds.get_batch(method_idx = method_idx,sys_idx=sys_idx)
        self.assertEqual(list(data['type'][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(np.load('method_0/sys_0/set.000/coord.npy'),
                       ds.get_data_system_idx(method_idx).get_sys(sys_idx).idx_map,
                       3, 
                       data['coord'])
        self._in_array(np.load('method_0/sys_0/set.000/test.npy'),
                       ds.get_data_system_idx(method_idx).get_sys(sys_idx).idx_map,
                       self.test_ndof,
                       data['test'])
        self.assertAlmostEqual(np.linalg.norm(np.zeros([batch_size,
                                                        self.natoms[sys_idx]*self.test_ndof])
                                              -
                                              data['null']
        ), 0.0)
        sys_idx = 2
        data = ds.get_batch(method_idx = method_idx,sys_idx=sys_idx)
        self.assertEqual(list(data['type'][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(np.load('method_0/sys_2/set.000/coord.npy'),
                       ds.get_data_system_idx(method_idx).get_sys(sys_idx).idx_map,
                       3, 
                       data['coord'])
        self._in_array(np.load('method_0/sys_2/set.000/test.npy'),
                       ds.get_data_system_idx(method_idx).get_sys(sys_idx).idx_map,
                       self.test_ndof,
                       data['test'])
        self.assertAlmostEqual(np.linalg.norm(np.zeros([batch_size, 
                                                        self.natoms[sys_idx]*self.test_ndof])
                                              -
                                              data['null']
        ), 0.0)




    def _idx_map(self, target, idx_map, ndof):
        natoms = len(idx_map)
        target = target.reshape([-1, natoms, ndof])
        target = target[:,idx_map,:]
        target = target.reshape([-1, natoms * ndof])
        return target        

    def _in_array(self, target, idx_map, ndof, array):
        target = self._idx_map(target, idx_map, ndof)
        all_find = []
        for ii in array :
            find = False
            for jj in target :
                if np.linalg.norm(ii - jj) < 1e-5 :
                    find = True
            all_find.append(find)
        for idx,ii in enumerate(all_find) :
            self.assertTrue(ii, msg = 'does not find frame %d in array' % idx)



                
