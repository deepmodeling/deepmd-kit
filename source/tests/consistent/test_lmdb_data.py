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
        _collate_lmdb_batch,
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

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_same_len(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(len(reader), len(ds))

    def test_same_frame_data(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        for i in range(len(reader)):
            frame_dp = reader[i]
            frame_pt = ds[i]
            self.assertEqual(set(frame_dp.keys()), set(frame_pt.keys()))
            for key in frame_dp:
                dp_val = frame_dp[key]
                pt_val = frame_pt[key]
                if isinstance(dp_val, np.ndarray):
                    np.testing.assert_array_equal(
                        dp_val, pt_val, err_msg=f"key={key}, frame={i}"
                    )
                else:
                    self.assertEqual(dp_val, pt_val, msg=f"key={key}, frame={i}")

    def test_same_batch_size(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="auto")
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size="auto")
        self.assertEqual(reader.batch_size, ds.batch_size)

    def test_same_properties(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(reader.index, ds.index)
        self.assertEqual(reader.total_batch, ds.total_batch)
        self.assertEqual(reader.batch_sizes, ds.batch_sizes)

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


@unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
class TestMixedNlocConsistency(unittest.TestCase):
    """Consistency tests for mixed-nloc LMDB: collate, LmdbDataset iteration."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._lmdb_path = _create_mixed_nloc_lmdb(f"{cls._tmpdir.name}/mixed.lmdb")
        cls._type_map = ["O", "H"]

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_collate_mixed_nloc_raises(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        with self.assertRaises(NotImplementedError):
            _collate_lmdb_batch([reader[0], reader[4]])

    def test_collate_same_nloc_ok(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        batch = _collate_lmdb_batch([reader[0], reader[1]])
        self.assertEqual(batch["coord"].shape[0], 2)

    def test_mixed_batch_true_raises(self):
        with self.assertRaises(NotImplementedError):
            LmdbDataset(self._lmdb_path, self._type_map, batch_size=2, mixed_batch=True)

    def test_pt_dataset_mixed_batch_flag(self):
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        self.assertFalse(ds.mixed_batch)

    def test_pt_full_epoch_mixed_nloc(self):
        import torch

        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        all_fids = []
        with torch.device("cpu"):
            for dl in ds.dataloaders:
                for batch in dl:
                    atype = batch["atype"]
                    nloc = atype.shape[1]
                    for i in range(atype.shape[0]):
                        self.assertEqual(atype[i].shape[0], nloc)
                    all_fids.extend(batch["fid"])
        self.assertEqual(sorted(all_fids), list(range(10)))

    def test_pt_batch_shapes_consistent(self):
        import torch

        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=3)
        with torch.device("cpu"):
            for batch in ds.dataloaders[0]:
                bs = batch["atype"].shape[0]
                nloc = batch["atype"].shape[1]
                self.assertEqual(batch["coord"].shape, (bs, nloc, 3))
                self.assertEqual(batch["force"].shape, (bs, nloc, 3))
                self.assertEqual(batch["natoms"].shape, (bs, 4))


@unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
class TestLmdbNeighborStatConsistency(unittest.TestCase):
    """Test neighbor stat values from LMDB match expected geometry."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        X, Y, Z = np.mgrid[0:2:3j, 0:2:3j, 0:2:3j]
        positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
        natoms = 27
        cell = np.array([3.0, 0, 0, 0, 3.0, 0, 0, 0, 3.0], dtype=np.float64)
        atype = np.zeros(natoms, dtype=np.int64)
        path = f"{cls._tmpdir.name}/grid.lmdb"
        env = lmdb.open(path, map_size=10 * 1024 * 1024)
        with env.begin(write=True) as txn:
            meta = {
                "nframes": 3,
                "frame_idx_fmt": "012d",
                "type_map": ["TYPE"],
                "system_info": {"natoms": [natoms], "formula": "grid"},
            }
            txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
            for i in range(3):
                frame = {
                    "atom_types": {
                        "type": "<i8",
                        "shape": (natoms,),
                        "data": atype.tobytes(),
                    },
                    "coords": {
                        "type": "<f8",
                        "shape": (natoms, 3),
                        "data": positions.astype(np.float64).tobytes(),
                    },
                    "cells": {
                        "type": "<f8",
                        "shape": (3, 3),
                        "data": cell.reshape(3, 3).tobytes(),
                    },
                    "energies": {
                        "type": "<f8",
                        "shape": (1,),
                        "data": np.array([0.0], dtype=np.float64).tobytes(),
                    },
                    "forces": {
                        "type": "<f8",
                        "shape": (natoms, 3),
                        "data": np.zeros((natoms, 3), dtype=np.float64).tobytes(),
                    },
                    "atom_names": ["TYPE"],
                    "atom_numbs": [
                        {
                            "type": "<i8",
                            "shape": (1,),
                            "data": np.array([natoms], dtype=np.int64).tobytes(),
                        }
                    ],
                }
                txn.put(
                    format(i, "012d").encode(), msgpack.packb(frame, use_bin_type=True)
                )
        env.close()
        cls._lmdb_path = path

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_neighbor_stat_values(self):
        """Neighbor stat from LMDB matches expected values for grid geometry."""
        from deepmd.dpmodel.utils.lmdb_data import (
            make_neighbor_stat_data,
        )
        from deepmd.pt.utils.neighbor_stat import (
            NeighborStat,
        )

        type_map = ["TYPE", "NO_THIS_TYPE"]
        data = make_neighbor_stat_data(self._lmdb_path, type_map)

        for rcut in (1.0, 2.0, 4.0):
            for mixed_type in (True, False):
                with self.subTest(rcut=rcut, mixed_type=mixed_type):
                    rcut_eps = rcut + 1e-3
                    nei = NeighborStat(len(type_map), rcut_eps, mixed_type=mixed_type)
                    min_nbor_dist, max_nbor_size = nei.get_stat(data)

                    upper = int(np.ceil(rcut_eps)) + 1
                    X, Y, Z = np.mgrid[-upper:upper, -upper:upper, -upper:upper]
                    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                    distance = np.linalg.norm(positions, axis=1)
                    expected_neighbors = np.count_nonzero(
                        np.logical_and(distance > 0, distance <= rcut_eps)
                    )

                    self.assertAlmostEqual(min_nbor_dist, 1.0, places=6)
                    expected = [expected_neighbors]
                    if not mixed_type:
                        expected.append(0)
                    np.testing.assert_array_equal(max_nbor_size, expected)


if __name__ == "__main__":
    unittest.main()
