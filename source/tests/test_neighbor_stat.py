import shutil
import numpy as np
import unittest
import dpdata

from deepmd.entrypoints.neighbor_stat import neighbor_stat

def gen_sys(nframes):
    natoms = 1000
    data = {}
    X, Y, Z = np.mgrid[0:9:10j, 0:9:10j, 0:9:10j]
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T #+ 0.1
    data['coords'] = np.repeat(positions[np.newaxis, :, :], nframes, axis=0)
    data['forces'] = np.random.random([nframes, natoms, 3])
    data['cells'] = np.array([10., 0., 0., 0., 10., 0., 0., 0., 10.]).reshape(1,3,3)
    data['energies'] = np.random.random([nframes, 1])
    data['atom_names'] = ['TYPE']
    data['atom_numbs'] = [1000]
    data['atom_types'] = np.repeat(0, 1000)
    return data


class TestNeighborStat(unittest.TestCase):
    def setUp(self):
        data0 = gen_sys(1)
        sys0 = dpdata.LabeledSystem()
        sys0.data = data0
        sys0.to_deepmd_npy('system_0', set_size = 10)
        
    def tearDown(self):
        shutil.rmtree('system_0')

    def test_neighbor_stat(self):
        # set rcut to 0. will cause a core dumped
        # TODO: check what is wrong
        for rcut in (3., 6., 11.):
            with self.subTest():
                rcut += 1e-3 # prevent numerical errors
                min_nbor_dist, max_nbor_size = neighbor_stat(system="system_0", rcut=rcut, type_map=["TYPE"])
                upper = np.ceil(rcut)+1
                X, Y, Z = np.mgrid[-upper:upper, -upper:upper, -upper:upper]
                positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                # distance to (0,0,0)
                distance = np.linalg.norm(positions, axis=1)
                expected_neighbors = np.count_nonzero(np.logical_and(distance > 0, distance <= rcut))
                self.assertAlmostEqual(min_nbor_dist, 1.0, 6)
                self.assertEqual(max_nbor_size, [expected_neighbors])
