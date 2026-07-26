# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)

import h5py
import numpy as np

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
