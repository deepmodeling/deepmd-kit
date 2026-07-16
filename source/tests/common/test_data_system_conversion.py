# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import sys
import tempfile
import types
import unittest
from pathlib import (
    Path,
)
from typing import (
    ClassVar,
)
from unittest.mock import (
    patch,
)

import h5py
import lmdb
import msgpack
import numpy as np

from deepmd.dpmodel.utils.lmdb_data import (
    is_lmdb,
)
from deepmd.utils import (
    data_system,
)
from deepmd.utils.data_system import (
    LmdbDataSystem,
    get_data,
    process_systems,
    validate_lmdb_systems,
)


def _write_minimal_deepmd_hdf5(file_name: str) -> None:
    with h5py.File(file_name, "w") as fp:
        system = fp.create_group("H")
        system.create_dataset("type.raw", data=np.array([0], dtype=np.int32))
        string_dtype = h5py.string_dtype(encoding="utf-8")
        system.create_dataset("type_map.raw", data=np.array(["H"], dtype=string_dtype))
        set_dir = system.create_group("set.000")
        set_dir.create_dataset("coord.npy", data=np.zeros((1, 3), dtype=np.float32))
        set_dir.create_dataset(
            "box.npy", data=np.eye(3, dtype=np.float32).reshape(1, 9)
        )


def _encode_array(arr: np.ndarray) -> dict:
    return {
        "type": str(arr.dtype),
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def _write_minimal_lmdb(path: str) -> None:
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    frame = {
        "atom_names": ["H"],
        "atom_numbs": [1],
        "atom_types": _encode_array(np.array([0], dtype=np.int64)),
        "cells": _encode_array(np.eye(3, dtype=np.float64) * 8.0),
        "coords": _encode_array(np.zeros((1, 3), dtype=np.float64)),
        "energies": _encode_array(np.array([0.0], dtype=np.float64)),
        "forces": _encode_array(np.zeros((1, 3), dtype=np.float64)),
    }
    metadata = {
        "nframes": 1,
        "frame_idx_fmt": "012d",
        "type_map": ["H"],
        "system_info": {
            "formula": "H",
            "natoms": [1],
            "nframes": 1,
        },
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        txn.put(b"000000000000", msgpack.packb(frame, use_bin_type=True))
    env.close()


class _FakeMultiSystems:
    write_count = 0
    load_calls: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, *systems) -> None:
        self.systems = list(systems)
        self.loaded = False

    def load_systems_from_file(self, file_name: str, fmt: str):
        self.load_calls.append((file_name, fmt))
        self.loaded = True
        return self

    def __len__(self) -> int:
        return 1 if self.loaded or self.systems else 0

    def to(self, fmt: str, file_name: str) -> None:
        type(self).write_count += 1
        if fmt == "deepmd/hdf5":
            _write_minimal_deepmd_hdf5(file_name)
        elif fmt == "lmdb":
            _write_minimal_lmdb(file_name)
        else:
            raise AssertionError(fmt)


class _FakeLabeledSystem:
    def __init__(self, file_name: str, fmt: str) -> None:
        self.file_name = file_name
        self.fmt = fmt


class TestDpdataFormatConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.source = self.root / "data.extxyz"
        self.source.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")
        _FakeMultiSystems.write_count = 0
        _FakeMultiSystems.load_calls = []
        data_system._DPDATA_CONVERSION_CACHE.clear()
        self.fake_dpdata = types.SimpleNamespace(
            MultiSystems=_FakeMultiSystems,
            LabeledSystem=_FakeLabeledSystem,
        )

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        self.tmpdir.cleanup()
        data_system._DPDATA_CONVERSION_CACHE.clear()

    def test_process_systems_defaults_to_lmdb_and_reuses_cache(self) -> None:
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            systems = process_systems(str(self.source), fmt="extxyz")
            systems_again = process_systems(str(self.source), fmt="extxyz")

        self.assertEqual(systems, systems_again)
        self.assertEqual(_FakeMultiSystems.write_count, 1)
        self.assertEqual(_FakeMultiSystems.load_calls, [(str(self.source), "extxyz")])
        self.assertEqual(len(systems), 1)
        self.assertTrue(systems[0].endswith(".lmdb"))
        self.assertTrue(is_lmdb(systems[0]))
        self.assertTrue(Path(systems[0]).is_relative_to(self.root))
        self.assertEqual(Path(systems[0]).parent, self.root / ".deepmd_dpdata_cache")

    def test_process_systems_cache_is_scoped_to_cwd(self) -> None:
        other_cwd = self.root / "run2"
        other_cwd.mkdir()

        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            systems = process_systems(str(self.source), fmt="extxyz")
            os.chdir(other_cwd)
            systems_other = process_systems(str(self.source), fmt="extxyz")

        self.assertNotEqual(systems, systems_other)
        self.assertEqual(_FakeMultiSystems.write_count, 2)
        self.assertEqual(Path(systems[0]).parent, self.root / ".deepmd_dpdata_cache")
        self.assertEqual(
            Path(systems_other[0]).parent,
            other_cwd / ".deepmd_dpdata_cache",
        )

    def test_process_systems_converts_to_explicit_hdf5(self) -> None:
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            systems = process_systems(
                str(self.source), fmt="extxyz", out_fmt="deepmd/hdf5"
            )

        self.assertEqual(_FakeMultiSystems.write_count, 1)
        self.assertEqual(_FakeMultiSystems.load_calls, [(str(self.source), "extxyz")])
        self.assertEqual(len(systems), 1)
        self.assertTrue(systems[0].endswith("#/H"))

    def test_get_data_uses_format_conversion(self) -> None:
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            data = get_data(
                {
                    "systems": str(self.source),
                    "format": "auto",
                    "batch_size": 1,
                },
                0.0,
                ["H"],
                None,
            )

        self.assertEqual(data.get_nsystems(), 1)
        self.assertIsInstance(data, LmdbDataSystem)
        self.assertEqual(_FakeMultiSystems.load_calls, [(str(self.source), "extxyz")])
        batch = data.get_batch()
        self.assertIn("type", batch)
        self.assertIn("natoms_vec", batch)
        self.assertEqual(batch["coord"].shape, (1, 3))
        self.assertEqual(data.data_systems, [data])
        stat_set = data._load_set(data.dirs[0])
        self.assertEqual(stat_set["coord"].shape, (1, 3))
        self.assertEqual(stat_set["type"].shape, (1, 1))

    def test_multiple_lmdb_paths_are_rejected(self) -> None:
        lmdb_a = self.root / "a.lmdb"
        lmdb_b = self.root / "b.lmdb"
        _write_minimal_lmdb(str(lmdb_a))
        _write_minimal_lmdb(str(lmdb_b))

        with self.assertRaisesRegex(ValueError, "exactly one path"):
            get_data(
                {
                    "systems": [str(lmdb_a), str(lmdb_b)],
                    "batch_size": 1,
                },
                0.0,
                ["H"],
                None,
            )

    def test_backend_without_lmdb_support_rejects_any_resolved_path(self) -> None:
        lmdb_path = self.root / "unsupported.lmdb"
        _write_minimal_lmdb(str(lmdb_path))

        with self.assertRaisesRegex(NotImplementedError, "Paddle backend"):
            validate_lmdb_systems(
                [str(lmdb_path)], backend_name="Paddle", supported=False
            )

    def test_lmdb_stack_frames_rejects_empty_batch(self) -> None:
        lmdb_path = self.root / "empty-batch.lmdb"
        _write_minimal_lmdb(str(lmdb_path))
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size=1)

        with self.assertRaisesRegex(ValueError, "empty LMDB frame batch"):
            data._stack_frames([])


if __name__ == "__main__":
    unittest.main()
