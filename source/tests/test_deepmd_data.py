import os,sys,shutil,copy
import numpy as np
import unittest

from deepmd.utils.data import DeepmdData
from deepmd.env import GLOBAL_NP_FLOAT_PRECISION

if GLOBAL_NP_FLOAT_PRECISION == np.float32 :
    places = 6
else:
    places = 12

class TestDataTypeSel(unittest.TestCase):
    def setUp(self):
        self.data_name = 'test_data'
        os.makedirs(self.data_name, exist_ok = True)
        os.makedirs(os.path.join(self.data_name,'set.foo'), exist_ok = True)
        np.savetxt(os.path.join(self.data_name, 'type.raw'), 
                   np.array([0, 1, 1, 0, 1, 1]), 
                   fmt = '%d')
        self.nframes = 3
        self.natoms = 6
        # coord
        path = os.path.join(self.data_name, 'set.foo', 'coord.npy')
        self.coord = np.random.random([self.nframes, self.natoms, 3])
        np.save(path, np.reshape(self.coord, [self.nframes, -1]))
        self.coord = self.coord[:,[0,3,1,2,4,5],:]
        self.coord = self.coord.reshape([self.nframes, -1])
        # box
        path = os.path.join(self.data_name, 'set.foo', 'box.npy')
        self.box = np.random.random([self.nframes, 9])
        np.save(path, self.box)
        # value
        path = os.path.join(self.data_name, 'set.foo', 'value_1.npy')
        self.value_1 = np.arange(self.nframes * 2)
        self.value_1 = np.reshape(self.value_1, [self.nframes, 2])
        np.save(path, self.value_1)
        # value
        path = os.path.join(self.data_name, 'set.foo', 'value_2.npy')
        self.value_2 = np.arange(self.nframes * 4)
        self.value_2 = np.reshape(self.value_2, [self.nframes, 4])
        np.save(path, self.value_2)

    def tearDown(self) :
        shutil.rmtree(self.data_name)

    def test_load_set_1(self) :
        dd = DeepmdData(self.data_name)\
             .add('value_1', 1, atomic=True, must=True, type_sel = [0])
        data = dd._load_set(os.path.join(self.data_name, 'set.foo'))
        self.assertEqual(data['value_1'].shape, (self.nframes, 2))
        for ii in range(self.nframes):
            for jj in range(2):
                self.assertAlmostEqual(data['value_1'][ii][jj],
                                       self.value_1[ii][jj])
                

    def test_load_set_2(self) :
        dd = DeepmdData(self.data_name)\
             .add('value_2', 1, atomic=True, must=True, type_sel = [1])
        data = dd._load_set(os.path.join(self.data_name, 'set.foo'))
        self.assertEqual(data['value_2'].shape, (self.nframes, 4))
        for ii in range(self.nframes):
            for jj in range(4):
                self.assertAlmostEqual(data['value_2'][ii][jj],
                                       self.value_2[ii][jj])                


class TestData (unittest.TestCase) :
    def setUp (self) :
        self.data_name = 'test_data'
        os.makedirs(self.data_name, exist_ok = True)
        os.makedirs(os.path.join(self.data_name,'set.foo'), exist_ok = True)
        os.makedirs(os.path.join(self.data_name,'set.bar'), exist_ok = True)
        os.makedirs(os.path.join(self.data_name,'set.tar'), exist_ok = True)
        np.savetxt(os.path.join(self.data_name, 'type.raw'), 
                   np.array([1, 0]), 
                   fmt = '%d')
        np.savetxt(os.path.join(self.data_name, 'type_map.raw'), 
                   np.array(['foo', 'bar']), 
                   fmt = '%s')
        self.nframes = 5
        self.natoms = 2
        # coord
        path = os.path.join(self.data_name, 'set.foo', 'coord.npy')
        self.coord = np.random.random([self.nframes, self.natoms, 3])
        np.save(path, np.reshape(self.coord, [self.nframes, -1]))
        self.coord = self.coord[:,[1,0],:]
        self.coord = self.coord.reshape([self.nframes, -1])
        # coord bar
        path = os.path.join(self.data_name, 'set.bar', 'coord.npy')
        self.coord_bar = np.random.random([self.nframes, 3 * self.natoms])
        np.save(path, self.coord_bar)
        self.coord_bar = self.coord_bar.reshape([self.nframes, self.natoms, 3])
        self.coord_bar = self.coord_bar[:,[1,0],:]
        self.coord_bar = self.coord_bar.reshape([self.nframes, -1])
        # coord tar
        path = os.path.join(self.data_name, 'set.tar', 'coord.npy')
        self.coord_tar = np.random.random([2, 3 * self.natoms])
        np.save(path, self.coord_tar)
        self.coord_tar = self.coord_tar.reshape([2, self.natoms, 3])
        self.coord_tar = self.coord_tar[:,[1,0],:]
        self.coord_tar = self.coord_tar.reshape([2, -1])
        # box
        path = os.path.join(self.data_name, 'set.foo', 'box.npy')
        self.box = np.random.random([self.nframes, 9])
        np.save(path, self.box)
        # box bar
        path = os.path.join(self.data_name, 'set.bar', 'box.npy')
        self.box_bar = np.random.random([self.nframes, 9])
        np.save(path, self.box_bar)
        # box tar
        path = os.path.join(self.data_name, 'set.tar', 'box.npy')
        self.box_tar = np.random.random([2, 9])
        np.save(path, self.box_tar)
        # t a
        path = os.path.join(self.data_name, 'set.foo', 'test_atomic.npy')
        self.test_atomic = np.random.random([self.nframes, self.natoms, 7])
        self.redu_atomic = np.sum(self.test_atomic, axis = 1)
        np.save(path, np.reshape(self.test_atomic, [self.nframes, -1]))
        self.test_atomic = self.test_atomic[:,[1,0],:]        
        self.test_atomic = self.test_atomic.reshape([self.nframes, -1])
        # t f
        path = os.path.join(self.data_name, 'set.foo', 'test_frame.npy')
        self.test_frame = np.random.random([self.nframes, 5])
        np.save(path, self.test_frame)
        path = os.path.join(self.data_name, 'set.bar', 'test_frame.npy')
        self.test_frame_bar = np.random.random([self.nframes, 5])
        np.save(path, self.test_frame_bar)
        # t n
        self.test_null = np.zeros([self.nframes, 2 * self.natoms])
    
    def tearDown(self) :
        shutil.rmtree(self.data_name)

    def test_init (self) :
        dd = DeepmdData(self.data_name)
        self.assertEqual(dd.idx_map[0], 1)
        self.assertEqual(dd.idx_map[1], 0)
        self.assertEqual(dd.type_map, ['foo', 'bar'])
        self.assertEqual(dd.test_dir, 'test_data/set.tar')
        self.assertEqual(dd.train_dirs, ['test_data/set.bar', 'test_data/set.foo'])

    def test_init_type_map (self) :
        dd = DeepmdData(self.data_name, type_map = ['bar', 'foo', 'tar'])
        self.assertEqual(dd.idx_map[0], 0)
        self.assertEqual(dd.idx_map[1], 1)
        self.assertEqual(dd.atom_type[0], 0)
        self.assertEqual(dd.atom_type[1], 1)
        self.assertEqual(dd.type_map, ['bar', 'foo'])

    def test_load_set(self) :
        dd = DeepmdData(self.data_name)\
             .add('test_atomic', 7, atomic=True, must=True)\
             .add('test_frame', 5, atomic=False, must=True)\
             .add('test_null', 2, atomic=True, must=False)
        data = dd._load_set(os.path.join(self.data_name, 'set.foo'))
        nframes = data['coord'].shape[0]
        self.assertEqual(dd.get_numb_set(), 2)
        self.assertEqual(dd.get_type_map(), ['foo', 'bar'])
        self.assertEqual(dd.get_natoms(), 2)
        self.assertEqual(list(dd.get_natoms_vec(3)), [2,2,1,1,0])
        for ii in range(nframes) :
            self.assertEqual(data['type'][ii][0], 0)
            self.assertEqual(data['type'][ii][1], 1)
        self.assertEqual(data['find_coord'], 1)
        self._comp_np_mat2(data['coord'], self.coord)
        self.assertEqual(data['find_test_atomic'], 1)
        self._comp_np_mat2(data['test_atomic'], self.test_atomic)
        self.assertEqual(data['find_test_frame'], 1)
        self._comp_np_mat2(data['test_frame'], self.test_frame)
        self.assertEqual(data['find_test_null'], 0)
        self._comp_np_mat2(data['test_null'], self.test_null)

    def test_shuffle(self) :
        dd = DeepmdData(self.data_name)\
             .add('test_atomic', 7, atomic=True, must=True)\
             .add('test_frame', 5, atomic=False, must=True)
        data = dd._load_set(os.path.join(self.data_name, 'set.foo'))
        data_bk = copy.deepcopy(data)
        data, idx = dd._shuffle_data(data)
        self._comp_np_mat2(data_bk['coord'][idx,:], 
                           data['coord'])
        self._comp_np_mat2(data_bk['test_atomic'][idx,:], 
                           data['test_atomic'])
        self._comp_np_mat2(data_bk['test_frame'][idx,:], 
                           data['test_frame'])

    def test_reduce(self) :
        dd = DeepmdData(self.data_name)\
             .add('test_atomic', 7, atomic=True, must=True)
        dd.reduce('redu', 'test_atomic')        
        data = dd._load_set(os.path.join(self.data_name, 'set.foo'))
        self.assertEqual(data['find_test_atomic'], 1)
        self._comp_np_mat2(data['test_atomic'], self.test_atomic)
        self.assertEqual(data['find_redu'], 1)
        self._comp_np_mat2(data['redu'], self.redu_atomic)
        
    def test_reduce_null(self) :
        dd = DeepmdData(self.data_name)\
             .add('test_atomic_1', 7, atomic=True, must=False)
        dd.reduce('redu', 'test_atomic_1')
        data = dd._load_set(os.path.join(self.data_name, 'set.foo'))
        self.assertEqual(data['find_test_atomic_1'], 0)
        self._comp_np_mat2(data['test_atomic_1'], np.zeros([self.nframes, self.natoms * 7]))
        self.assertEqual(data['find_redu'], 0)
        self._comp_np_mat2(data['redu'], np.zeros([self.nframes, 7]))
    
    def test_load_null_must(self):
        dd = DeepmdData(self.data_name)\
             .add('test_atomic_1', 7, atomic=True, must=True)
        with self.assertRaises(RuntimeError) :
            data = dd._load_set(os.path.join(self.data_name, 'set.foo'))

    def test_avg(self) :
        dd = DeepmdData(self.data_name)\
             .add('test_frame', 5, atomic=False, must=True)
        favg = dd.avg('test_frame')
        fcmp = np.average(np.concatenate((self.test_frame, self.test_frame_bar), axis = 0), axis = 0)
        for ii in range(favg.size) :
            self.assertAlmostEqual((favg[ii]), (fcmp[ii]), places = places)

    def test_check_batch_size(self) :
        dd = DeepmdData(self.data_name)
        ret = dd.check_batch_size(10)
        self.assertEqual(ret, (os.path.join(self.data_name,'set.bar'), 5))
        ret = dd.check_batch_size(5)
        self.assertEqual(ret, None)

    def test_check_test_size(self):
        dd = DeepmdData(self.data_name)
        ret = dd.check_test_size(10)
        self.assertEqual(ret, (os.path.join(self.data_name,'set.tar'), 2))
        ret = dd.check_test_size(2)
        self.assertEqual(ret, None)

    def test_get_batch(self) :
        dd = DeepmdData(self.data_name)
        data = dd.get_batch(5)
        self._comp_np_mat2(np.sort(data['coord'], axis = 0), 
                           np.sort(self.coord_bar, axis = 0))
        data = dd.get_batch(5)
        self._comp_np_mat2(np.sort(data['coord'], axis = 0), 
                           np.sort(self.coord, axis = 0))
        data = dd.get_batch(5)
        self._comp_np_mat2(np.sort(data['coord'], axis = 0), 
                           np.sort(self.coord_bar, axis = 0))
        data = dd.get_batch(5)
        self._comp_np_mat2(np.sort(data['coord'], axis = 0), 
                           np.sort(self.coord, axis = 0))

    def test_get_test(self) :
        dd = DeepmdData(self.data_name)
        data = dd.get_test()
        self._comp_np_mat2(np.sort(data['coord'], axis = 0), 
                           np.sort(self.coord_tar, axis = 0))

    def test_get_nbatch(self):
        dd = DeepmdData(self.data_name)
        nb = dd.get_numb_batch(1, 0)
        self.assertEqual(nb, 5)
        nb = dd.get_numb_batch(2, 0)
        self.assertEqual(nb, 2)
        
    def _comp_np_mat2(self, first, second) :
        for ii in range(first.shape[0]) :
            for jj in range(first.shape[1]) :
                self.assertAlmostEqual(first[ii][jj], second[ii][jj], 
                                       msg = 'item [%d][%d] does not match' % (ii,jj), 
                                       places = places)
