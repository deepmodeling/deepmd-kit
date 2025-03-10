# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import os
import shutil
import unittest

import numpy as np

from deepmd.tf.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.tf.utils.data import (
    DeepmdData,
)

from ..seed import (
    GLOBAL_SEED,
)
from .common import (
    tests_path,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    places = 6
else:
    places = 12


class TestDataTypeSel(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        self.data_name = "test_data"
        os.makedirs(self.data_name, exist_ok=True)
        os.makedirs(os.path.join(self.data_name, "set.foo"), exist_ok=True)
        np.savetxt(
            os.path.join(self.data_name, "type.raw"),
            np.array([0, 1, 1, 0, 1, 1]),
            fmt="%d",
        )
        self.nframes = 3
        self.natoms = 6
        # coord
        path = os.path.join(self.data_name, "set.foo", "coord.npy")
        self.coord = rng.random([self.nframes, self.natoms, 3])
        np.save(path, np.reshape(self.coord, [self.nframes, -1]))
        self.coord = self.coord[:, [0, 3, 1, 2, 4, 5], :]
        self.coord = self.coord.reshape([self.nframes, -1])
        # box
        path = os.path.join(self.data_name, "set.foo", "box.npy")
        self.box = rng.random([self.nframes, 9])
        np.save(path, self.box)
        # value
        path = os.path.join(self.data_name, "set.foo", "value_1.npy")
        self.value_1 = np.arange(self.nframes * 2)
        self.value_1 = np.reshape(self.value_1, [self.nframes, 2])
        np.save(path, self.value_1)
        # value
        path = os.path.join(self.data_name, "set.foo", "value_2.npy")
        self.value_2 = np.arange(self.nframes * 4)
        self.value_2 = np.reshape(self.value_2, [self.nframes, 4])
        np.save(path, self.value_2)

    def tearDown(self) -> None:
        shutil.rmtree(self.data_name)

    def test_load_set_1(self) -> None:
        dd = DeepmdData(self.data_name).add(
            "value_1", 1, atomic=True, must=True, type_sel=[0]
        )
        data = dd._load_set(os.path.join(self.data_name, "set.foo"))
        self.assertEqual(data["value_1"].shape, (self.nframes, 2))
        np.testing.assert_almost_equal(data["value_1"], self.value_1)

    def test_load_set_2(self) -> None:
        dd = DeepmdData(self.data_name).add(
            "value_2", 1, atomic=True, must=True, type_sel=[1]
        )
        data = dd._load_set(os.path.join(self.data_name, "set.foo"))
        self.assertEqual(data["value_2"].shape, (self.nframes, 4))
        np.testing.assert_almost_equal(data["value_2"], self.value_2)


class TestData(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        self.data_name = "test_data"
        os.makedirs(self.data_name, exist_ok=True)
        os.makedirs(os.path.join(self.data_name, "set.foo"), exist_ok=True)
        os.makedirs(os.path.join(self.data_name, "set.bar"), exist_ok=True)
        os.makedirs(os.path.join(self.data_name, "set.tar"), exist_ok=True)
        os.makedirs(os.path.join(self.data_name, "set.foo"), exist_ok=True)
        np.savetxt(os.path.join(self.data_name, "type.raw"), np.array([1, 0]), fmt="%d")
        np.savetxt(
            os.path.join(self.data_name, "type_map.raw"),
            np.array(["foo", "bar"]),
            fmt="%s",
        )
        self.nframes = 5
        self.natoms = 2
        # coord
        path = os.path.join(self.data_name, "set.foo", "coord.npy")
        self.coord = rng.random([self.nframes, self.natoms, 3])
        np.save(path, np.reshape(self.coord, [self.nframes, -1]))
        self.coord = self.coord[:, [1, 0], :]
        self.coord = self.coord.reshape([self.nframes, -1])
        # coord bar
        path = os.path.join(self.data_name, "set.bar", "coord.npy")
        self.coord_bar = rng.random([self.nframes, 3 * self.natoms])
        np.save(path, self.coord_bar)
        self.coord_bar = self.coord_bar.reshape([self.nframes, self.natoms, 3])
        self.coord_bar = self.coord_bar[:, [1, 0], :]
        self.coord_bar = self.coord_bar.reshape([self.nframes, -1])
        # coord tar
        path = os.path.join(self.data_name, "set.tar", "coord.npy")
        self.coord_tar = rng.random([2, 3 * self.natoms])
        np.save(path, self.coord_tar)
        self.coord_tar = self.coord_tar.reshape([2, self.natoms, 3])
        self.coord_tar = self.coord_tar[:, [1, 0], :]
        self.coord_tar = self.coord_tar.reshape([2, -1])
        # box
        path = os.path.join(self.data_name, "set.foo", "box.npy")
        self.box = rng.random([self.nframes, 9])
        np.save(path, self.box)
        # box bar
        path = os.path.join(self.data_name, "set.bar", "box.npy")
        self.box_bar = rng.random([self.nframes, 9])
        np.save(path, self.box_bar)
        # box tar
        path = os.path.join(self.data_name, "set.tar", "box.npy")
        self.box_tar = rng.random([2, 9])
        np.save(path, self.box_tar)
        # t a
        path = os.path.join(self.data_name, "set.foo", "test_atomic.npy")
        self.test_atomic = rng.random([self.nframes, self.natoms, 7])
        self.redu_atomic = np.sum(self.test_atomic, axis=1)
        np.save(path, np.reshape(self.test_atomic, [self.nframes, -1]))
        self.test_atomic = self.test_atomic[:, [1, 0], :]
        self.test_atomic = self.test_atomic.reshape([self.nframes, -1])
        # t f
        path = os.path.join(self.data_name, "set.foo", "test_frame.npy")
        self.test_frame = rng.random([self.nframes, 5])
        np.save(path, self.test_frame)
        path = os.path.join(self.data_name, "set.bar", "test_frame.npy")
        self.test_frame_bar = rng.random([self.nframes, 5])
        np.save(path, self.test_frame_bar)
        path = os.path.join(self.data_name, "set.tar", "test_frame.npy")
        self.test_frame_tar = rng.random([2, 5])
        np.save(path, self.test_frame_tar)
        # t n
        self.test_null = np.zeros([self.nframes, 2 * self.natoms])
        # tensor shape
        path = os.path.join(self.data_name, "set.foo", "tensor_natoms.npy")
        self.tensor_natoms = rng.random([self.nframes, self.natoms, 6])
        self.tensor_natoms[:, 0, :] = 0
        np.save(path, self.tensor_natoms)
        path = os.path.join(self.data_name, "set.foo", "tensor_nsel.npy")
        self.tensor_nsel = self.tensor_natoms[:, 1, :]
        np.save(path, self.tensor_nsel)

    def tearDown(self) -> None:
        shutil.rmtree(self.data_name)

    def test_init(self) -> None:
        dd = DeepmdData(self.data_name)
        self.assertEqual(dd.idx_map[0], 1)
        self.assertEqual(dd.idx_map[1], 0)
        self.assertEqual(dd.type_map, ["foo", "bar"])
        self.assertEqual(
            dd.dirs, ["test_data/set.bar", "test_data/set.foo", "test_data/set.tar"]
        )

    def test_init_type_map(self) -> None:
        dd = DeepmdData(self.data_name, type_map=["bar", "foo", "tar"])
        self.assertEqual(dd.idx_map[0], 0)
        self.assertEqual(dd.idx_map[1], 1)
        self.assertEqual(dd.atom_type[0], 0)
        self.assertEqual(dd.atom_type[1], 1)
        self.assertEqual(dd.type_map, ["bar", "foo", "tar"])

    def test_init_type_map_error(self) -> None:
        with self.assertRaises(ValueError):
            DeepmdData(self.data_name, type_map=["bar"])

    def test_load_set(self) -> None:
        dd = (
            DeepmdData(self.data_name)
            .add("test_atomic", 7, atomic=True, must=True)
            .add("test_frame", 5, atomic=False, must=True)
            .add("test_null", 2, atomic=True, must=False)
        )
        data = dd._load_set(os.path.join(self.data_name, "set.foo"))
        nframes = data["coord"].shape[0]
        self.assertEqual(dd.get_numb_set(), 3)
        self.assertEqual(dd.get_type_map(), ["foo", "bar"])
        self.assertEqual(dd.get_natoms(), 2)
        self.assertEqual(list(dd.get_natoms_vec(3)), [2, 2, 1, 1, 0])
        for ii in range(nframes):
            self.assertEqual(data["type"][ii][0], 0)
            self.assertEqual(data["type"][ii][1], 1)
        self.assertEqual(data["find_coord"], 1)
        self._comp_np_mat2(data["coord"], self.coord)
        self.assertEqual(data["find_test_atom"], 1)
        self._comp_np_mat2(data["test_atom"], self.test_atomic)
        self.assertEqual(data["find_test_frame"], 1)
        self._comp_np_mat2(data["test_frame"], self.test_frame)
        self.assertEqual(data["find_test_null"], 0)
        self._comp_np_mat2(data["test_null"], self.test_null)

    def test_shuffle(self) -> None:
        dd = (
            DeepmdData(self.data_name)
            .add("test_atomic", 7, atomic=True, must=True)
            .add("test_frame", 5, atomic=False, must=True)
        )
        data = dd._load_set(os.path.join(self.data_name, "set.foo"))
        data_bk = copy.deepcopy(data)
        data, idx = dd._shuffle_data(data)
        self._comp_np_mat2(data_bk["coord"][idx, :], data["coord"])
        self._comp_np_mat2(data_bk["test_atom"][idx, :], data["test_atom"])
        self._comp_np_mat2(data_bk["test_frame"][idx, :], data["test_frame"])

    def test_shuffle_with_numb_copy(self) -> None:
        path = os.path.join(self.data_name, "set.foo", "numb_copy.npy")
        prob = np.arange(self.nframes)
        np.save(path, prob)
        dd = (
            DeepmdData(self.data_name)
            .add("test_atomic", 7, atomic=True, must=True)
            .add("test_frame", 5, atomic=False, must=True)
        )
        data = dd._load_set(os.path.join(self.data_name, "set.foo"))
        data_bk = copy.deepcopy(data)
        data, idx = dd._shuffle_data(data)
        assert idx.size == np.sum(prob)
        self._comp_np_mat2(data_bk["coord"][idx, :], data["coord"])
        self._comp_np_mat2(data_bk["test_atom"][idx, :], data["test_atom"])
        self._comp_np_mat2(data_bk["test_frame"][idx, :], data["test_frame"])

    def test_reduce(self) -> None:
        dd = DeepmdData(self.data_name).add("test_atomic", 7, atomic=True, must=True)
        dd.reduce("redu", "test_atomic")
        data = dd._load_set(os.path.join(self.data_name, "set.foo"))
        self.assertEqual(data["find_test_atom"], 1)
        self._comp_np_mat2(data["test_atom"], self.test_atomic)
        self.assertEqual(data["find_redu"], 1)
        self._comp_np_mat2(data["redu"], self.redu_atomic)

    def test_reduce_null(self) -> None:
        dd = DeepmdData(self.data_name).add("test_atomic_1", 7, atomic=True, must=False)
        dd.reduce("redu", "test_atomic_1")
        data = dd._load_set(os.path.join(self.data_name, "set.foo"))
        self.assertEqual(data["find_test_atom_1"], 0)
        self._comp_np_mat2(
            data["test_atom_1"], np.zeros([self.nframes, self.natoms * 7])
        )
        self.assertEqual(data["find_redu"], 0)
        self._comp_np_mat2(data["redu"], np.zeros([self.nframes, 7]))

    def test_load_null_must(self) -> None:
        dd = DeepmdData(self.data_name).add("test_atomic_1", 7, atomic=True, must=True)
        with self.assertRaises(RuntimeError):
            data = dd._load_set(os.path.join(self.data_name, "set.foo"))

    def test_avg(self) -> None:
        dd = DeepmdData(self.data_name).add("test_frame", 5, atomic=False, must=True)
        favg = dd.avg("test_frame")
        fcmp = np.average(
            np.concatenate(
                (self.test_frame, self.test_frame_bar, self.test_frame_tar), axis=0
            ),
            axis=0,
        )
        np.testing.assert_almost_equal(favg, fcmp, places)

    def test_check_batch_size(self) -> None:
        dd = DeepmdData(self.data_name)
        ret = dd.check_batch_size(10)
        self.assertEqual(ret, (os.path.join(self.data_name, "set.bar"), 5))
        ret = dd.check_batch_size(5)
        self.assertEqual(ret, (os.path.join(self.data_name, "set.tar"), 2))
        ret = dd.check_batch_size(1)
        self.assertEqual(ret, None)

    def test_check_test_size(self) -> None:
        dd = DeepmdData(self.data_name)
        ret = dd.check_test_size(10)
        self.assertEqual(ret, (os.path.join(self.data_name, "set.bar"), 5))
        ret = dd.check_test_size(5)
        self.assertEqual(ret, (os.path.join(self.data_name, "set.tar"), 2))
        ret = dd.check_test_size(1)
        self.assertEqual(ret, None)

    def test_get_batch(self) -> None:
        dd = DeepmdData(self.data_name)
        data = dd.get_batch(5)
        self._comp_np_mat2(
            np.sort(data["coord"], axis=0), np.sort(self.coord_bar, axis=0)
        )
        data = dd.get_batch(5)
        self._comp_np_mat2(np.sort(data["coord"], axis=0), np.sort(self.coord, axis=0))
        data = dd.get_batch(5)
        self._comp_np_mat2(
            np.sort(data["coord"], axis=0), np.sort(self.coord_tar, axis=0)
        )
        data = dd.get_batch(5)
        self._comp_np_mat2(
            np.sort(data["coord"], axis=0), np.sort(self.coord_bar, axis=0)
        )
        data = dd.get_batch(5)
        self._comp_np_mat2(np.sort(data["coord"], axis=0), np.sort(self.coord, axis=0))

    def test_get_test(self) -> None:
        dd = DeepmdData(self.data_name)
        data = dd.get_test()
        expected_coord = np.concatenate(
            (self.coord_bar, self.coord, self.coord_tar), axis=0
        )
        self._comp_np_mat2(
            np.sort(data["coord"], axis=0), np.sort(expected_coord, axis=0)
        )

    def test_get_nbatch(self) -> None:
        dd = DeepmdData(self.data_name)
        nb = dd.get_numb_batch(1, 0)
        self.assertEqual(nb, 5)
        nb = dd.get_numb_batch(2, 0)
        self.assertEqual(nb, 2)

    def test_get_tensor(self) -> None:
        dd_natoms = (
            DeepmdData(self.data_name)
            .add(
                "tensor_nsel",
                6,
                atomic=True,
                must=True,
                type_sel=[0],
                output_natoms_for_type_sel=True,
            )
            .add(
                "tensor_natoms",
                6,
                atomic=True,
                must=True,
                type_sel=[0],
                output_natoms_for_type_sel=True,
            )
        )
        data_natoms = dd_natoms._load_set(os.path.join(self.data_name, "set.foo"))
        dd_nsel = (
            DeepmdData(self.data_name)
            .add(
                "tensor_nsel",
                6,
                atomic=True,
                must=True,
                type_sel=[0],
                output_natoms_for_type_sel=False,
            )
            .add(
                "tensor_natoms",
                6,
                atomic=True,
                must=True,
                type_sel=[0],
                output_natoms_for_type_sel=False,
            )
        )
        data_nsel = dd_nsel._load_set(os.path.join(self.data_name, "set.foo"))
        np.testing.assert_allclose(
            data_natoms["tensor_natoms"], data_natoms["tensor_nsel"]
        )
        np.testing.assert_allclose(data_nsel["tensor_natoms"], data_nsel["tensor_nsel"])
        np.testing.assert_allclose(
            data_natoms["tensor_natoms"].reshape(self.nframes, self.natoms, -1)[
                :, 0, :
            ],
            data_nsel["tensor_natoms"],
        )

    def _comp_np_mat2(self, first, second) -> None:
        np.testing.assert_almost_equal(first, second, places)


class TestDataMixType(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        self.data_name = "test_data"
        os.makedirs(self.data_name, exist_ok=True)
        os.makedirs(os.path.join(self.data_name, "set.foo"), exist_ok=True)
        os.makedirs(os.path.join(self.data_name, "set.bar"), exist_ok=True)
        os.makedirs(os.path.join(self.data_name, "set.tar"), exist_ok=True)
        np.savetxt(os.path.join(self.data_name, "type.raw"), np.array([0, 0]), fmt="%d")
        np.savetxt(
            os.path.join(self.data_name, "type_map.raw"),
            np.array(["foo", "bar"]),
            fmt="%s",
        )
        self.nframes = 5
        self.natoms = 2
        # coord
        path = os.path.join(self.data_name, "set.foo", "coord.npy")
        self.coord = rng.random([self.nframes, self.natoms, 3])
        np.save(path, np.reshape(self.coord, [self.nframes, -1]))
        self.coord = self.coord[:, [1, 0], :]
        self.coord = self.coord.reshape([self.nframes, -1])
        # coord bar
        path = os.path.join(self.data_name, "set.bar", "coord.npy")
        self.coord_bar = rng.random([self.nframes, 3 * self.natoms])
        np.save(path, self.coord_bar)
        self.coord_bar = self.coord_bar.reshape([self.nframes, self.natoms, 3])
        self.coord_bar = self.coord_bar[:, [1, 0], :]
        self.coord_bar = self.coord_bar.reshape([self.nframes, -1])
        # coord tar
        path = os.path.join(self.data_name, "set.tar", "coord.npy")
        self.coord_tar = rng.random([2, 3 * self.natoms])
        np.save(path, self.coord_tar)
        self.coord_tar = self.coord_tar.reshape([2, self.natoms, 3])
        self.coord_tar = self.coord_tar[:, [1, 0], :]
        self.coord_tar = self.coord_tar.reshape([2, -1])
        # box
        path = os.path.join(self.data_name, "set.foo", "box.npy")
        self.box = rng.random([self.nframes, 9])
        np.save(path, self.box)
        # box bar
        path = os.path.join(self.data_name, "set.bar", "box.npy")
        self.box_bar = rng.random([self.nframes, 9])
        np.save(path, self.box_bar)
        # box tar
        path = os.path.join(self.data_name, "set.tar", "box.npy")
        self.box_tar = rng.random([2, 9])
        np.save(path, self.box_tar)
        # real_atom_types
        path = os.path.join(self.data_name, "set.foo", "real_atom_types.npy")
        self.real_atom_types = rng.integers(0, 2, size=[self.nframes, self.natoms])
        np.save(path, self.real_atom_types)
        # real_atom_types bar
        path = os.path.join(self.data_name, "set.bar", "real_atom_types.npy")
        self.real_atom_types_bar = rng.integers(0, 2, size=[self.nframes, self.natoms])
        np.save(path, self.real_atom_types_bar)
        # real_atom_types tar
        path = os.path.join(self.data_name, "set.tar", "real_atom_types.npy")
        self.real_atom_types_tar = rng.integers(0, 2, size=[2, self.natoms])
        np.save(path, self.real_atom_types_tar)

    def test_init_type_map(self) -> None:
        dd = DeepmdData(self.data_name, type_map=["bar", "foo", "tar"])
        self.assertEqual(dd.enforce_type_map, True)
        self.assertEqual(dd.type_map, ["bar", "foo", "tar"])
        self.assertEqual(dd.mixed_type, True)
        self.assertEqual(dd.type_idx_map[0], 1)
        self.assertEqual(dd.type_idx_map[1], 0)
        self.assertEqual(dd.type_idx_map[2], -1)

    def test_init_type_map_error(self) -> None:
        with self.assertRaises(ValueError):
            DeepmdData(self.data_name, type_map=["foo"])

    def tearDown(self) -> None:
        shutil.rmtree(self.data_name)


class TestH5Data(unittest.TestCase):
    def setUp(self) -> None:
        self.data_name = str(tests_path / "test.hdf5")

    def test_init(self) -> None:
        dd = DeepmdData(self.data_name)
        self.assertEqual(dd.idx_map[0], 0)
        self.assertEqual(dd.type_map, ["X"])
        self.assertEqual(dd.dirs[0], self.data_name + "#/set.000")

    def test_get_batch(self) -> None:
        dd = DeepmdData(self.data_name)
        data = dd.get_batch(5)
