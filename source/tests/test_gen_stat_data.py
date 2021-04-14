import os,sys,platform,json,shutil
import numpy as np
import unittest
import dpdata

from deepmd.utils.data_system import DeepmdDataSystem
from deepmd.fit import EnerFitting
from deepmd.model.model_stat import make_stat_input, merge_sys_stat, _make_all_stat_ref

def gen_sys(nframes, atom_types):
    natoms = len(atom_types)
    data = {}
    data['coords'] = np.random.random([nframes, natoms, 3])
    data['forces'] = np.random.random([nframes, natoms, 3])
    data['cells'] = np.random.random([nframes, 9])
    data['energies'] = np.random.random([nframes, 1])
    types = list(set(list(atom_types)))
    types.sort()
    data['atom_names'] = []
    data['atom_numbs'] = []
    for ii in range(len(types)):
        data['atom_names'] .append( 'TYPE_%d' % ii )
        data['atom_numbs'] .append(np.sum(atom_types == ii))
    data['atom_types'] = np.array(atom_types, dtype = int)
    return data

class TestGenStatData(unittest.TestCase) :
    def setUp(self):
        data0 = gen_sys(20, [0, 1, 0, 2, 1])
        data1 = gen_sys(30, [0, 1, 0, 0])
        sys0 = dpdata.LabeledSystem()
        sys1 = dpdata.LabeledSystem()
        sys0.data = data0
        sys1.data = data1
        sys0.to_deepmd_npy('system_0', set_size = 10)
        sys1.to_deepmd_npy('system_1', set_size = 10)
        
    def tearDown(self):
        shutil.rmtree('system_0')
        shutil.rmtree('system_1')

    def _comp_data(self, d0, d1) :
        for ii in range(d0.shape[0]):
            for jj in range(d0.shape[1]):
                for kk in range(d0.shape[2]):
                    self.assertAlmostEqual(d0[ii][jj][kk], d1[ii][jj][kk])

    def test_merge_all_stat(self):
        np.random.seed(0)
        data0 = DeepmdDataSystem(['system_0', 'system_1'], 
                                5, 
                                10, 
                                1.0)
        data0.add('energy', 1, must = True)
        np.random.seed(0)
        data1 = DeepmdDataSystem(['system_0', 'system_1'], 
                                5, 
                                10, 
                                1.0)
        data1.add('force', 3, atomic = True, must = True)
        np.random.seed(0)
        data2 = DeepmdDataSystem(['system_0', 'system_1'], 
                                5, 
                                10, 
                                1.0)
        data2.add('force', 3, atomic = True, must = True)
        
        np.random.seed(0)
        all_stat_0 = make_stat_input(data0, 10, merge_sys = False)
        np.random.seed(0)
        all_stat_1 = make_stat_input(data1, 10, merge_sys = True)
        all_stat_2 = merge_sys_stat(all_stat_0)
        np.random.seed(0)
        all_stat_3 = _make_all_stat_ref(data2, 10)
        
        ####################################
        # only check if the energy is concatenated correctly
        ####################################
        dd = 'energy'
            # if 'find_' in dd: continue
            # if 'natoms_vec' in dd: continue
            # if 'default_mesh' in dd: continue
            # print(all_stat_2[dd])
            # print(dd, all_stat_1[dd])
        d1 = np.array(all_stat_1[dd])
        d2 = np.array(all_stat_2[dd])
        d3 = np.array(all_stat_3[dd])
        # print(dd)
        # print(d1.shape)
        # print(d2.shape)            
        # self.assertEqual(all_stat_2[dd], all_stat_1[dd])
        self._comp_data(d1, d2)
        self._comp_data(d1, d3)


class TestEnerShift(unittest.TestCase):
    def setUp(self):
        data0 = gen_sys(30, [0, 1, 0, 2, 1])
        data1 = gen_sys(30, [0, 1, 0, 0])    
        sys0 = dpdata.LabeledSystem()
        sys1 = dpdata.LabeledSystem()
        sys0.data = data0
        sys1.data = data1
        sys0.to_deepmd_npy('system_0', set_size = 10)
        sys1.to_deepmd_npy('system_1', set_size = 10)
        
    def tearDown(self):
        shutil.rmtree('system_0')
        shutil.rmtree('system_1')

    def test_ener_shift(self):
        np.random.seed(0)
        data = DeepmdDataSystem(['system_0', 'system_1'], 
                                5, 
                                10, 
                                1.0)
        data.add('energy', 1, must = True)
        ener_shift0 = data.compute_energy_shift(rcond = 1)
        all_stat = make_stat_input(data, 4, merge_sys = False)
        ener_shift1 = EnerFitting._compute_output_stats(all_stat, rcond = 1)        
        for ii in range(len(ener_shift0)):
            self.assertAlmostEqual(ener_shift0[ii], ener_shift1[ii])
