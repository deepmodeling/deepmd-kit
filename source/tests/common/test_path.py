# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)

import h5py
import numpy as np

from deepmd.utils.data_system import (
    process_systems,
)
from deepmd.utils.path import (
    DPH5Path,
    DPPath,
)


class PathTest:
    path: DPPath

    def test_numpy(self) -> None:
        numpy_path = self.path / "testcase"
        arr1 = np.ones(3)
        self.assertFalse(numpy_path.is_file())
        numpy_path.save_numpy(arr1)
        self.assertTrue(numpy_path.is_file())
        arr2 = numpy_path.load_numpy()
        np.testing.assert_array_equal(arr1, arr2)

    def test_dir(self) -> None:
        dir_path = self.path / "testcase"
        self.assertFalse(dir_path.is_dir())
        dir_path.mkdir()
        self.assertTrue(dir_path.is_dir())


class TestOSPath(PathTest, unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.path = DPPath(self.tempdir.name, "a")

    def tearDown(self) -> None:
        self.tempdir.cleanup()


class TestH5Path(PathTest, unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w") as f:
            pass
        self.path = DPPath(h5file, "a")

    def tearDown(self) -> None:
        self.tempdir.cleanup()


class TestH5PathReadOnly(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        h5file = str((Path(self.tempdir.name) / "testcase.h5").resolve())
        with h5py.File(h5file, "w"):
            pass
        self.path = DPPath(h5file, "r")

    def tearDown(self) -> None:
        assert isinstance(self.path, DPH5Path)
        self.path.root.close()
        DPH5Path._load_h5py.cache_clear()
        DPH5Path._file_keys.cache_clear()
        self.tempdir.cleanup()

    def test_write_operations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "read-only"):
            (self.path / "value").save_numpy(np.ones(1))
        with self.assertRaisesRegex(ValueError, "read-only"):
            (self.path / "group").mkdir()


class TestSystemDiscovery(unittest.TestCase):
    def test_lmdb_paths_and_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "direct.lmdb"
            direct.mkdir()
            parent = root / "collection"
            nested = parent / "nested" / "frames"
            nested.mkdir(parents=True)
            (nested / "data.mdb").touch()
            npy = parent / "npy"
            npy.mkdir()
            (npy / "type.raw").touch()

            found = process_systems([str(direct), str(parent)])

            self.assertEqual(found[0], str(direct))
            self.assertCountEqual(found, [str(direct), str(nested), str(npy)])

    def test_lmdb_pattern_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "train.lmdb"
            first.mkdir()
            second = root / "test.lmdb"
            second.mkdir()

            self.assertEqual(
                process_systems(str(root), patterns=["train*.lmdb"]),
                [str(first)],
            )

    def test_lmdb_is_terminal_even_when_patterns_exclude_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            internal = root / "frames.lmdb" / "nested"
            internal.mkdir(parents=True)
            (internal / "type.raw").touch()
            self.assertEqual(process_systems(str(root), patterns=["nested"]), [])

    def test_hdf5_paths_expand_inside_containers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            container = Path(directory) / "systems.h5"
            with h5py.File(container, "w") as handle:
                handle.create_dataset("first/type.raw", data=[0])
                handle.create_dataset("nested/second/type.raw", data=[0, 1])
            expected = [f"{container}#/first", f"{container}#/nested/second"]
            self.assertEqual(process_systems(str(container)), expected)
            self.assertEqual(
                process_systems([expected[1], expected[0]]), expected[::-1]
            )
            self.assertEqual(
                process_systems("systems.h5#/nested", root_dir=directory),
                [expected[1]],
            )
