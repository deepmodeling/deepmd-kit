import os
import shutil
import unittest

import numpy as np

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.utils import (
    random,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    places = 6
else:
    places = 12


class TestDataSystem(unittest.TestCase):
    def setUp(self):
        self.nsys = 4
        self.nframes = [3, 6, 5, 4]
        self.natoms = [3, 4, 6, 5]
        self.atom_type = [[1, 0, 0], [2, 1, 0, 2], [0, 0, 1, 1, 2, 1], [0, 2, 2, 0, 0]]
        self.test_ndof = 2
        self.sys_name = []
        self.nset = 3
        for ii in range(self.nsys):
            sys_name = "sys_%d" % ii
            self.sys_name.append(sys_name)
            os.makedirs(sys_name, exist_ok=True)
            np.savetxt(os.path.join(sys_name, "type.raw"), self.atom_type[ii], fmt="%d")
            for jj in range(self.nset):
                set_name = os.path.join(sys_name, "set.%03d" % jj)
                os.makedirs(set_name, exist_ok=True)
                path = os.path.join(set_name, "coord.npy")
                val = np.random.random([self.nframes[ii] + jj, self.natoms[ii] * 3])
                np.save(path, val)
                path = os.path.join(set_name, "box.npy")
                val = np.random.random([self.nframes[ii] + jj, 9]) * 10
                np.save(path, val)
                path = os.path.join(set_name, "test.npy")
                val = np.random.random(
                    [self.nframes[ii] + jj, self.natoms[ii] * self.test_ndof]
                )
                np.save(path, val)

    def tearDown(self):
        for ii in range(self.nsys):
            sys_name = "sys_%d" % ii
            shutil.rmtree(sys_name)

    def test_ntypes(self):
        batch_size = 3
        test_size = 2
        ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)
        ds.add("test", self.test_ndof, atomic=True, must=True)
        ds.add("null", self.test_ndof, atomic=True, must=False)
        self.assertEqual(ds.get_ntypes(), 3)
        self.assertEqual(ds.get_nbatches(), [2, 4, 3, 2])
        self.assertEqual(ds.get_nsystems(), self.nsys)
        self.assertEqual(list(ds.get_batch_size()), [batch_size] * 4)

    def test_batch_size_5(self):
        batch_size = "auto:5"
        test_size = 2
        ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)
        self.assertEqual(ds.batch_size, [2, 2, 1, 1])

    def test_batch_size_null(self):
        batch_size = "auto:3"
        test_size = 2
        ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)
        self.assertEqual(ds.batch_size, [1, 1, 1, 1])

    def test_batch_size_raise(self):
        batch_size = "foo"
        test_size = 2
        with self.assertRaises(RuntimeError):
            ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)

    def test_get_test(self):
        batch_size = 3
        test_size = 2
        ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)
        ds.add("test", self.test_ndof, atomic=True, must=True)
        ds.add("null", self.test_ndof, atomic=True, must=False)
        ds.add("ones", self.test_ndof, atomic=True, must=False, default=1.0)
        sys_idx = 0
        data = ds.get_test(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_0/set.002/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_0/set.002/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros(
                    [self.nframes[sys_idx] + 2, self.natoms[sys_idx] * self.test_ndof]
                )
                - data["null"]
            ),
            0.0,
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.ones(
                    [self.nframes[sys_idx] + 2, self.natoms[sys_idx] * self.test_ndof]
                )
                - data["ones"]
            ),
            0.0,
        )

        sys_idx = 2
        data = ds.get_test(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_2/set.002/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_2/set.002/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros(
                    [self.nframes[sys_idx] + 2, self.natoms[sys_idx] * self.test_ndof]
                )
                - data["null"]
            ),
            0.0,
        )

    def test_get_batch(self):
        batch_size = 3
        test_size = 2
        ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)
        ds.add("test", self.test_ndof, atomic=True, must=True)
        ds.add("null", self.test_ndof, atomic=True, must=False)
        sys_idx = 0
        data = ds.get_batch(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_0/set.000/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_0/set.000/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros([batch_size, self.natoms[sys_idx] * self.test_ndof])
                - data["null"]
            ),
            0.0,
        )
        data = ds.get_batch(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_0/set.001/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_0/set.001/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros([batch_size, self.natoms[sys_idx] * self.test_ndof])
                - data["null"]
            ),
            0.0,
        )
        data = ds.get_batch(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_0/set.000/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_0/set.000/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros([batch_size, self.natoms[sys_idx] * self.test_ndof])
                - data["null"]
            ),
            0.0,
        )
        sys_idx = 2
        data = ds.get_batch(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_2/set.000/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_2/set.000/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros([batch_size, self.natoms[sys_idx] * self.test_ndof])
                - data["null"]
            ),
            0.0,
        )
        data = ds.get_batch(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_2/set.001/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_2/set.001/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros([batch_size, self.natoms[sys_idx] * self.test_ndof])
                - data["null"]
            ),
            0.0,
        )
        data = ds.get_batch(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_2/set.001/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_2/set.001/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros([batch_size, self.natoms[sys_idx] * self.test_ndof])
                - data["null"]
            ),
            0.0,
        )
        data = ds.get_batch(sys_idx=sys_idx)
        self.assertEqual(list(data["type"][0]), list(np.sort(self.atom_type[sys_idx])))
        self._in_array(
            np.load("sys_2/set.000/coord.npy"),
            ds.get_sys(sys_idx).idx_map,
            3,
            data["coord"],
        )
        self._in_array(
            np.load("sys_2/set.000/test.npy"),
            ds.get_sys(sys_idx).idx_map,
            self.test_ndof,
            data["test"],
        )
        self.assertAlmostEqual(
            np.linalg.norm(
                np.zeros([batch_size, self.natoms[sys_idx] * self.test_ndof])
                - data["null"]
            ),
            0.0,
        )

    def test_prob_sys_size_1(self):
        batch_size = 1
        test_size = 1
        ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)
        prob = ds._prob_sys_size_ext("prob_sys_size; 0:2:2; 2:4:8")
        self.assertAlmostEqual(np.sum(prob), 1)
        self.assertAlmostEqual(np.sum(prob[0:2]), 0.2)
        self.assertAlmostEqual(np.sum(prob[2:4]), 0.8)
        # number of training set is self.nset-1
        # shift is the total number of set size shift...
        shift = np.sum(np.arange(self.nset - 1))
        self.assertAlmostEqual(
            prob[1] / prob[0],
            float(self.nframes[1] * (self.nset - 1) + shift)
            / float(self.nframes[0] * (self.nset - 1) + shift),
        )
        self.assertAlmostEqual(
            prob[3] / prob[2],
            float(self.nframes[3] * (self.nset - 1) + shift)
            / float(self.nframes[2] * (self.nset - 1) + shift),
        )

    def test_prob_sys_size_2(self):
        batch_size = 1
        test_size = 1
        ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)
        prob = ds._prob_sys_size_ext("prob_sys_size; 1:2:0.4; 2:4:1.6")
        self.assertAlmostEqual(np.sum(prob), 1)
        self.assertAlmostEqual(np.sum(prob[1:2]), 0.2)
        self.assertAlmostEqual(np.sum(prob[2:4]), 0.8)
        # number of training set is self.nset-1
        # shift is the total number of set size shift...
        shift = np.sum(np.arange(self.nset - 1))
        self.assertAlmostEqual(prob[0], 0.0)
        self.assertAlmostEqual(prob[1], 0.2)
        self.assertAlmostEqual(
            prob[3] / prob[2],
            float(self.nframes[3] * (self.nset - 1) + shift)
            / float(self.nframes[2] * (self.nset - 1) + shift),
        )

    def _idx_map(self, target, idx_map, ndof):
        natoms = len(idx_map)
        target = target.reshape([-1, natoms, ndof])
        target = target[:, idx_map, :]
        target = target.reshape([-1, natoms * ndof])
        return target

    def _in_array(self, target, idx_map, ndof, array):
        target = self._idx_map(target, idx_map, ndof)
        all_find = []
        for ii in array:
            find = False
            for jj in target:
                if np.linalg.norm(ii - jj) < 1e-5:
                    find = True
            all_find.append(find)
        for idx, ii in enumerate(all_find):
            self.assertTrue(ii, msg="does not find frame %d in array" % idx)

    def test_sys_prob_floating_point_error(self):
        # test floating point error; See #1917
        sys_probs = [
            0.010,
            0.010,
            0.010,
            0.010,
            0.010,
            0.010,
            0.010,
            0.010,
            0.010,
            0.150,
            0.100,
            0.100,
            0.050,
            0.050,
            0.020,
            0.015,
            0.015,
            0.050,
            0.020,
            0.015,
            0.040,
            0.055,
            0.025,
            0.025,
            0.015,
            0.025,
            0.055,
            0.040,
            0.040,
            0.005,
        ]
        ds = DeepmdDataSystem(self.sys_name, 3, 2, 2.0, sys_probs=sys_probs)
        self.assertEqual(ds.sys_probs.size, len(sys_probs))

    def test_get_mixed_batch(self):
        """Test get_batch with mixed system."""
        batch_size = "mixed:3"
        test_size = 2

        ds = DeepmdDataSystem(self.sys_name, batch_size, test_size, 2.0)
        ds.add("test", self.test_ndof, atomic=True, must=True)
        ds.add("null", self.test_ndof, atomic=True, must=False)
        random.seed(114514)
        # with this seed, the batch is fixed, with natoms 3, 6, 6
        data = ds.get_batch()
        np.testing.assert_equal(data["natoms_vec"], np.array([6, 6, 6, 0, 0]))
        np.testing.assert_equal(data["real_natoms_vec"][:, 0], np.array([3, 6, 6]))
        np.testing.assert_equal(data["type"][0, 3:6], np.array([-1, -1, -1]))
        np.testing.assert_equal(data["coord"][0, 9:18], np.zeros(9))
        for kk in ("test", "null"):
            np.testing.assert_equal(
                data[kk][0, 3 * self.test_ndof : 6 * self.test_ndof],
                np.zeros(3 * self.test_ndof),
            )
