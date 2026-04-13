# SPDX-License-Identifier: LGPL-3.0-or-later
"""Consistency tests: LmdbDataReader (dpmodel) vs LmdbDataset (pt).

Verifies that the framework-agnostic reader and the PyTorch wrapper
produce identical outputs for the same LMDB data.
"""

import tempfile
import unittest

import lmdb
import msgpack
import numpy as np

from deepmd.dpmodel.utils.lmdb_data import (
    LmdbDataReader,
)

try:
    from deepmd.pt.utils.lmdb_dataset import (
        LmdbDataset,
    )

    INSTALLED_PT = True
except ImportError:
    INSTALLED_PT = False


def _make_frame(natoms: int = 6, seed: int = 0) -> dict:
    """Create a synthetic frame dict for testing."""
    rng = np.random.RandomState(seed)
    n_type0 = max(1, natoms // 3)
    n_type1 = natoms - n_type0
    atype = np.array([0] * n_type0 + [1] * n_type1, dtype=np.int64)
    return {
        "atom_names": ["O", "H"],
        "atom_numbs": [
            {
                "type": "<i8",
                "shape": (1,),
                "data": np.array([n_type0], dtype=np.int64).tobytes(),
            },
            {
                "type": "<i8",
                "shape": (1,),
                "data": np.array([n_type1], dtype=np.int64).tobytes(),
            },
        ],
        "atom_types": {
            "type": "<i8",
            "shape": (natoms,),
            "data": atype.tobytes(),
        },
        "coords": {
            "type": "<f8",
            "shape": (natoms, 3),
            "data": rng.randn(natoms, 3).astype(np.float64).tobytes(),
        },
        "cells": {
            "type": "<f8",
            "shape": (3, 3),
            "data": (np.eye(3) * 10.0).astype(np.float64).tobytes(),
        },
        "energies": {
            "type": "<f8",
            "shape": (1,),
            "data": rng.randn(1).astype(np.float64).tobytes(),
        },
        "forces": {
            "type": "<f8",
            "shape": (natoms, 3),
            "data": rng.randn(natoms, 3).astype(np.float64).tobytes(),
        },
    }


def _create_lmdb(path: str, nframes: int = 10, natoms: int = 6) -> str:
    """Create a test LMDB database with uniform nloc."""
    n_type0 = max(1, natoms // 3)
    n_type1 = natoms - n_type0
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": nframes,
            "frame_idx_fmt": "012d",
            "system_info": {
                "natoms": [n_type0, n_type1],
                "formula": "test",
            },
        }
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        for i in range(nframes):
            key = format(i, "012d").encode()
            frame = _make_frame(natoms=natoms, seed=i)
            txn.put(key, msgpack.packb(frame, use_bin_type=True))
    env.close()
    return path


def _create_mixed_nloc_lmdb(path: str) -> str:
    """Create an LMDB with frames of different atom counts."""
    frames_spec = [(6, 4), (9, 4), (12, 2)]
    total = sum(c for _, c in frames_spec)
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": total,
            "frame_idx_fmt": "012d",
            "system_info": {"natoms": [2, 4], "formula": "mixed"},
        }
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        idx = 0
        for natoms, count in frames_spec:
            for j in range(count):
                txn.put(
                    format(idx, "012d").encode(),
                    msgpack.packb(
                        _make_frame(natoms=natoms, seed=idx), use_bin_type=True
                    ),
                )
                idx += 1
    env.close()
    return path


def _assert_frames_equal(test_case, frame_dp, frame_pt, frame_idx):
    """Assert two frames (from reader and dataset) are identical."""
    test_case.assertEqual(
        set(frame_dp.keys()),
        set(frame_pt.keys()),
        msg=f"frame={frame_idx}",
    )
    for key in frame_dp:
        dp_val = frame_dp[key]
        pt_val = frame_pt[key]
        if isinstance(dp_val, np.ndarray):
            np.testing.assert_array_equal(
                dp_val, pt_val, err_msg=f"key={key}, frame={frame_idx}"
            )
        else:
            test_case.assertEqual(dp_val, pt_val, msg=f"key={key}, frame={frame_idx}")


@unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
class TestLmdbDataConsistency(unittest.TestCase):
    """Verify LmdbDataReader (dpmodel) and LmdbDataset (pt) produce identical outputs."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._lmdb_path = _create_lmdb(
            f"{cls._tmpdir.name}/test.lmdb", nframes=10, natoms=6
        )
        cls._type_map = ["O", "H"]
        cls._reader = LmdbDataReader(cls._lmdb_path, cls._type_map, batch_size=2)
        cls._ds = LmdbDataset(cls._lmdb_path, cls._type_map, batch_size=2)

    @classmethod
    def tearDownClass(cls):
        del cls._ds, cls._reader
        cls._tmpdir.cleanup()

    def test_same_len(self):
        self.assertEqual(len(self._reader), len(self._ds))

    def test_same_frame_data(self):
        for i in range(len(self._reader)):
            _assert_frames_equal(self, self._reader[i], self._ds[i], i)

    def test_same_batch_size(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="auto")
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size="auto")
        self.assertEqual(reader.batch_size, ds.batch_size)

    def test_same_properties(self):
        self.assertEqual(self._reader.index, self._ds.index)
        self.assertEqual(self._reader.total_batch, self._ds.total_batch)
        self.assertEqual(self._reader.batch_sizes, self._ds.batch_sizes)
        self.assertEqual(self._reader.nframes, self._ds.nframes)
        self.assertEqual(self._reader.mixed_type, self._ds.mixed_type)
        self.assertEqual(self._reader.mixed_batch, self._ds.mixed_batch)

    def test_data_requirement(self):
        req = [
            {
                "key": "virial",
                "ndof": 9,
                "atomic": False,
                "must": False,
                "high_prec": False,
                "repeat": 1,
                "default": 0.0,
            }
        ]
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        reader.add_data_requirement(req)
        ds.add_data_requirement(req)
        frame_dp = reader[0]
        frame_pt = ds[0]
        np.testing.assert_array_equal(frame_dp["virial"], frame_pt["virial"])
        self.assertEqual(frame_dp["find_virial"], frame_pt["find_virial"])

    def test_mixed_nloc_same_frame_data(self):
        """Reader and dataset produce identical frames for mixed atom counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_mixed_nloc_lmdb(f"{tmpdir}/mixed.lmdb")
            reader = LmdbDataReader(path, self._type_map, batch_size=2)
            ds = LmdbDataset(path, self._type_map, batch_size=2)
            self.assertEqual(len(reader), len(ds))
            for i in range(len(reader)):
                _assert_frames_equal(self, reader[i], ds[i], i)

    def test_mixed_nloc_same_properties(self):
        """Reader and dataset agree on properties for mixed-nloc LMDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _create_mixed_nloc_lmdb(f"{tmpdir}/mixed.lmdb")
            reader = LmdbDataReader(path, self._type_map, batch_size=2)
            ds = LmdbDataset(path, self._type_map, batch_size=2)
            self.assertEqual(reader.nframes, ds.nframes)
            self.assertEqual(reader.batch_sizes, ds.batch_sizes)
            self.assertEqual(reader.mixed_batch, ds.mixed_batch)
            self.assertFalse(reader.mixed_batch)


if __name__ == "__main__":
    unittest.main()
