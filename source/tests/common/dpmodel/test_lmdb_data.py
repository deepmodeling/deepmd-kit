# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for LmdbDataReader, LmdbTestData, SameNlocBatchSampler, etc.

Pure dpmodel (NumPy/lmdb) tests — no PyTorch dependency.
"""

import tempfile
import unittest

import lmdb
import msgpack
import numpy as np

from deepmd.dpmodel.utils.lmdb_data import (
    LmdbDataReader,
    LmdbTestData,
    SameNlocBatchSampler,
    _expand_indices_by_blocks,
    compute_block_targets,
    is_lmdb,
    make_neighbor_stat_data,
)

# ============================================================
# LMDB creation helpers
# ============================================================


def _make_frame(natoms: int = 6, seed: int = 0) -> dict:
    """Create a synthetic frame dict for testing.

    Generates atom_types with roughly 1/3 type-0 and 2/3 type-1.
    """
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
    """Create an LMDB with frames of different atom counts.

    Frames 0-3: 6 atoms, Frames 4-7: 9 atoms, Frames 8-9: 12 atoms.
    """
    frames_spec = [(6, 4), (9, 4), (12, 2)]  # (natoms, count)
    total = sum(c for _, c in frames_spec)
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": total,
            "frame_idx_fmt": "012d",
            "system_info": {
                "natoms": [2, 4],  # first frame's type counts
                "formula": "mixed",
            },
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


def _create_lmdb_with_type_map(
    path: str,
    nframes: int = 6,
    natoms: int = 6,
    lmdb_type_map: list[str] | None = None,
) -> str:
    """Create a test LMDB with type_map stored in metadata."""
    n_type0 = max(1, natoms // 3)
    n_type1 = natoms - n_type0
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": nframes,
            "frame_idx_fmt": "012d",
            "system_info": {
                "natoms": [n_type0, n_type1],
            },
        }
        if lmdb_type_map is not None:
            meta["type_map"] = lmdb_type_map
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        for i in range(nframes):
            key = format(i, "012d").encode()
            frame = _make_frame(natoms=natoms, seed=i)
            txn.put(key, msgpack.packb(frame, use_bin_type=True))
    env.close()
    return path


def _create_lmdb_with_system_ids(
    path: str,
    system_frames: list[int],
    natoms: int = 6,
    type_map: list[str] | None = None,
) -> str:
    """Create a test LMDB with frame_system_ids in metadata."""
    total = sum(system_frames)
    n_type0 = max(1, natoms // 3)
    n_type1 = natoms - n_type0
    frame_system_ids = []
    for sid, nf in enumerate(system_frames):
        frame_system_ids.extend([sid] * nf)

    env = lmdb.open(path, map_size=50 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": total,
            "frame_idx_fmt": "012d",
            "system_info": {"natoms": [n_type0, n_type1]},
            "frame_system_ids": frame_system_ids,
            "frame_nlocs": [natoms] * total,
        }
        if type_map is not None:
            meta["type_map"] = type_map
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        for i in range(total):
            key = format(i, "012d").encode()
            frame = _make_frame(natoms=natoms, seed=i % 100)
            txn.put(key, msgpack.packb(frame, use_bin_type=True))
    env.close()
    return path


def _create_grid_lmdb(path: str, nframes: int = 3) -> str:
    """Create a test LMDB with 3x3x3 grid of atoms (27 atoms, cell=3A).

    Same geometry as test_neighbor_stat.py: positions at integer coords
    (0,1,2)^3, so min_nbor_dist = 1.0.
    """
    X, Y, Z = np.mgrid[0:2:3j, 0:2:3j, 0:2:3j]
    positions = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T  # (27, 3)
    natoms = 27
    cell = np.array([3.0, 0, 0, 0, 3.0, 0, 0, 0, 3.0], dtype=np.float64)
    atype = np.zeros(natoms, dtype=np.int64)

    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": nframes,
            "frame_idx_fmt": "012d",
            "type_map": ["TYPE"],
            "system_info": {"natoms": [natoms], "formula": "grid"},
        }
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        for i in range(nframes):
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
                format(i, "012d").encode(),
                msgpack.packb(frame, use_bin_type=True),
            )
    env.close()
    return path


# ============================================================
# LmdbDataReader basic tests
# ============================================================


class TestLmdbDataReader(unittest.TestCase):
    """Test LmdbDataReader (dpmodel) functionality."""

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

    def test_reader_standalone(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        frame = reader[0]
        self.assertIn("coord", frame)
        self.assertIn("energy", frame)
        self.assertIn("force", frame)
        self.assertIn("atype", frame)
        self.assertIn("box", frame)
        self.assertIn("natoms", frame)
        self.assertIn("real_natoms_vec", frame)
        self.assertIn("find_energy", frame)
        self.assertEqual(frame["coord"].dtype, np.float64)
        self.assertEqual(frame["atype"].dtype, np.int64)

    def test_len(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(len(reader), 10)

    def test_uniform_nloc_single_group(self):
        """Uniform-nloc LMDB has exactly one nloc group."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(len(reader.nloc_groups), 1)
        self.assertIn(6, reader.nloc_groups)
        self.assertEqual(len(reader.nloc_groups[6]), 10)

    def test_is_lmdb(self):
        self.assertTrue(is_lmdb(self._lmdb_path))
        self.assertTrue(is_lmdb("something.lmdb"))
        self.assertFalse(is_lmdb("/some/npy/system"))
        self.assertFalse(is_lmdb(["list", "of", "systems"]))

    def test_lmdb_test_data(self):
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        td.add("force", 3, atomic=True, must=False, high_prec=False)
        result = td.get_test()
        self.assertEqual(result["coord"].shape, (10, 18))
        self.assertEqual(result["box"].shape, (10, 9))
        self.assertEqual(result["type"].shape, (10, 6))
        self.assertEqual(result["energy"].shape, (10, 1))
        self.assertEqual(result["force"].shape, (10, 18))
        self.assertEqual(result["find_energy"], 1.0)
        self.assertEqual(result["find_force"], 1.0)


# ============================================================
# Mixed nloc tests
# ============================================================


class TestMixedNloc(unittest.TestCase):
    """Tests for mixed-nloc datasets and SameNlocBatchSampler."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._lmdb_path = _create_mixed_nloc_lmdb(f"{cls._tmpdir.name}/mixed.lmdb")
        cls._type_map = ["O", "H"]

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_nloc_groups_detected(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(set(reader.nloc_groups.keys()), {6, 9, 12})
        self.assertEqual(len(reader.nloc_groups[6]), 4)
        self.assertEqual(len(reader.nloc_groups[9]), 4)
        self.assertEqual(len(reader.nloc_groups[12]), 2)

    def test_per_frame_natoms_vec(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(reader[0]["natoms"][0], 6)
        self.assertEqual(reader[4]["natoms"][0], 9)
        self.assertEqual(reader[8]["natoms"][0], 12)
        np.testing.assert_array_equal(reader[0]["real_natoms_vec"], reader[0]["natoms"])

    def test_per_frame_shapes(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        f0, f4 = reader[0], reader[4]
        self.assertEqual(f0["coord"].shape, (6, 3))
        self.assertEqual(f0["atype"].shape, (6,))
        self.assertEqual(f4["coord"].shape, (9, 3))
        self.assertEqual(f4["atype"].shape, (9,))

    def test_frame_nlocs(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(reader.frame_nlocs, [6, 6, 6, 6, 9, 9, 9, 9, 12, 12])

    def test_sampler_all_batches_same_nloc(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = SameNlocBatchSampler(reader, shuffle=False, seed=42)
        for batch_indices in sampler:
            nlocs = [reader.frame_nlocs[i] for i in batch_indices]
            self.assertTrue(all(n == nlocs[0] for n in nlocs))

    def test_sampler_covers_all_frames(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = SameNlocBatchSampler(reader, shuffle=False, seed=42)
        all_indices = [i for batch in sampler for i in batch]
        self.assertEqual(sorted(all_indices), list(range(10)))

    def test_sampler_auto_batch_size_per_nloc(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="auto")
        bs_6 = reader.get_batch_size_for_nloc(6)
        bs_12 = reader.get_batch_size_for_nloc(12)
        self.assertGreaterEqual(bs_6, bs_12)

    def test_sampler_shuffle_deterministic(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        s1 = SameNlocBatchSampler(reader, shuffle=True, seed=123)
        s2 = SameNlocBatchSampler(reader, shuffle=True, seed=123)
        self.assertEqual(list(s1), list(s2))

    def test_sampler_len(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = SameNlocBatchSampler(reader, shuffle=False)
        self.assertEqual(len(sampler), len(list(sampler)))

    # --- LmdbTestData mixed-nloc tests ---

    def test_test_data_nloc_groups(self):
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        self.assertEqual(set(td.nloc_groups.keys()), {6, 9, 12})

    def test_test_data_get_test_specific_nloc(self):
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        td.add("force", 3, atomic=True, must=False, high_prec=False)
        r6 = td.get_test(nloc=6)
        self.assertEqual(r6["coord"].shape, (4, 6 * 3))
        r9 = td.get_test(nloc=9)
        self.assertEqual(r9["coord"].shape, (4, 9 * 3))
        r12 = td.get_test(nloc=12)
        self.assertEqual(r12["coord"].shape, (2, 12 * 3))

    def test_test_data_get_test_default_mixed(self):
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        result = td.get_test()
        self.assertEqual(result["coord"].shape[0], 4)

    def test_test_data_get_test_invalid_nloc(self):
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        with self.assertRaises(ValueError):
            td.get_test(nloc=999)

    def test_test_data_uniform_nloc_no_warning(self):
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb(f"{tmpdir.name}/uniform.lmdb", nframes=5, natoms=6)
        td = LmdbTestData(path, type_map=self._type_map, shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        self.assertEqual(td.get_test()["coord"].shape, (5, 18))
        tmpdir.cleanup()


# ============================================================
# Type map remapping tests
# ============================================================


class TestTypeMapRemapping(unittest.TestCase):
    """Test type_map remapping in LmdbDataReader and LmdbTestData."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._lmdb_path = _create_lmdb_with_type_map(
            f"{cls._tmpdir.name}/remap.lmdb",
            nframes=6,
            natoms=6,
            lmdb_type_map=["O", "H"],
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_reader_no_remap_when_match(self):
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        self.assertIsNone(reader._type_remap)

    def test_reader_remap_when_reversed(self):
        reader = LmdbDataReader(self._lmdb_path, ["H", "O"])
        np.testing.assert_array_equal(reader._type_remap, [1, 0])
        atype = reader[0]["atype"]
        n0 = max(1, 6 // 3)
        for i in range(n0):
            self.assertEqual(atype[i], 1)  # O -> 1
        for i in range(n0, 6):
            self.assertEqual(atype[i], 0)  # H -> 0

    def test_reader_remap_superset(self):
        reader = LmdbDataReader(self._lmdb_path, ["C", "O", "H"])
        np.testing.assert_array_equal(reader._type_remap, [1, 2])

    def test_reader_natoms_vec_after_remap(self):
        reader = LmdbDataReader(self._lmdb_path, ["H", "O"])
        natoms = reader[0]["natoms"]
        self.assertEqual(natoms[0], 6)
        self.assertEqual(natoms[2], 4)  # H count
        self.assertEqual(natoms[3], 2)  # O count

    def test_reader_missing_element_raises(self):
        with self.assertRaises(ValueError):
            LmdbDataReader(self._lmdb_path, ["O"])

    def test_reader_no_type_map_in_metadata(self):
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb_with_type_map(
            f"{tmpdir.name}/old.lmdb", nframes=3, natoms=6, lmdb_type_map=None
        )
        reader = LmdbDataReader(path, ["H", "O"])
        self.assertIsNone(reader._type_remap)
        tmpdir.cleanup()

    def test_testdata_no_remap_when_match(self):
        td = LmdbTestData(self._lmdb_path, type_map=["O", "H"], shuffle_test=False)
        self.assertIsNone(td._type_remap)

    def test_testdata_remap_when_reversed(self):
        td = LmdbTestData(self._lmdb_path, type_map=["H", "O"], shuffle_test=False)
        self.assertIsNotNone(td._type_remap)
        data = td.get_test()
        n0 = max(1, 6 // 3)
        for i in range(n0):
            self.assertEqual(data["type"][0, i], 1)
        for i in range(n0, 6):
            self.assertEqual(data["type"][0, i], 0)

    def test_testdata_remap_superset(self):
        td = LmdbTestData(self._lmdb_path, type_map=["C", "O", "H"], shuffle_test=False)
        self.assertIsNotNone(td._type_remap)

    def test_testdata_missing_element_raises(self):
        with self.assertRaises(ValueError):
            LmdbTestData(self._lmdb_path, type_map=["O"], shuffle_test=False)

    def test_testdata_no_type_map_in_metadata(self):
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb_with_type_map(
            f"{tmpdir.name}/old.lmdb", nframes=3, natoms=6, lmdb_type_map=None
        )
        td = LmdbTestData(path, type_map=["H", "O"], shuffle_test=False)
        self.assertIsNone(td._type_remap)
        tmpdir.cleanup()


# ============================================================
# auto_prob / frame_system_ids tests
# ============================================================


class TestAutoProb(unittest.TestCase):
    """Test auto_prob support: frame_system_ids, compute_block_targets,
    _expand_indices_by_blocks, and SameNlocBatchSampler with block_targets.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._lmdb_path = _create_lmdb_with_system_ids(
            f"{cls._tmpdir.name}/auto_prob.lmdb",
            system_frames=[100, 200, 300],
            natoms=6,
            type_map=["O", "H"],
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_reader_system_groups(self):
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        self.assertEqual(reader.nsystems, 3)
        self.assertEqual(reader.system_nframes, [100, 200, 300])
        self.assertEqual(len(reader.system_groups[0]), 100)
        self.assertEqual(len(reader.system_groups[1]), 200)
        self.assertEqual(len(reader.system_groups[2]), 300)

    def test_reader_no_system_ids_backward_compat(self):
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb(f"{tmpdir.name}/old.lmdb", nframes=10, natoms=6)
        reader = LmdbDataReader(path, ["O", "H"])
        self.assertEqual(reader.nsystems, 1)
        self.assertIsNone(reader.frame_system_ids)
        tmpdir.cleanup()

    def test_compute_block_targets_equal_weight(self):
        result = compute_block_targets(
            "prob_sys_size;0:1:0.5;1:2:0.5", nsystems=2, system_nframes=[100, 100]
        )
        self.assertEqual(result, [])

    def test_compute_block_targets_unequal(self):
        result = compute_block_targets(
            "prob_sys_size;0:1:0.5;1:2:0.5", nsystems=2, system_nframes=[100, 500]
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], ([0], 500))
        self.assertEqual(result[1], ([1], 500))

    def test_compute_block_targets_multi_sys_block(self):
        result = compute_block_targets(
            "prob_sys_size;0:2:0.5;2:3:0.5",
            nsystems=3,
            system_nframes=[100, 200, 300],
        )
        self.assertEqual(result, [])

    def test_compute_block_targets_asymmetric(self):
        result = compute_block_targets(
            "prob_sys_size;0:2:0.5;2:3:0.5",
            nsystems=3,
            system_nframes=[50, 50, 400],
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], [0, 1])
        self.assertEqual(result[0][1], 400)

    def test_expand_indices_basic(self):
        frame_system_ids = [0] * 5 + [1] * 5
        block_targets = [([0], 25), ([1], 25)]
        rng = np.random.default_rng(42)
        expanded = _expand_indices_by_blocks(
            list(range(10)), frame_system_ids, block_targets, rng
        )
        sys0 = [i for i in expanded if frame_system_ids[i] == 0]
        sys1 = [i for i in expanded if frame_system_ids[i] == 1]
        self.assertEqual(len(sys0), 25)
        self.assertEqual(len(sys1), 25)

    def test_expand_indices_no_expansion(self):
        frame_system_ids = [0] * 5 + [1] * 5
        block_targets = [([0], 5), ([1], 5)]
        rng = np.random.default_rng(42)
        expanded = _expand_indices_by_blocks(
            list(range(10)), frame_system_ids, block_targets, rng
        )
        self.assertEqual(sorted(expanded), list(range(10)))

    def test_expand_indices_remainder_sampling(self):
        from collections import (
            Counter,
        )

        frame_system_ids = [0] * 10
        block_targets = [([0], 23)]
        rng = np.random.default_rng(42)
        expanded = _expand_indices_by_blocks(
            list(range(10)), frame_system_ids, block_targets, rng
        )
        self.assertEqual(len(expanded), 23)
        counts = Counter(expanded)
        n_three = sum(1 for c in counts.values() if c == 3)
        self.assertEqual(n_three, 3)

    def test_expand_epoch_diversity(self):
        frame_system_ids = [0] * 10
        block_targets = [([0], 15)]
        results = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            expanded = _expand_indices_by_blocks(
                list(range(10)), frame_system_ids, block_targets, rng
            )
            results.append(sorted(expanded[10:]))
        unique = {tuple(r) for r in results}
        self.assertGreater(len(unique), 1)

    def test_sampler_with_block_targets(self):
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        block_targets = compute_block_targets(
            "prob_sys_size;0:1:0.5;1:3:0.5",
            nsystems=3,
            system_nframes=[100, 200, 300],
        )
        sampler = SameNlocBatchSampler(
            reader, shuffle=True, block_targets=block_targets
        )
        all_indices = [i for batch in sampler for i in batch]
        self.assertGreater(len(all_indices), 600)
        self.assertEqual(len(set(all_indices)), 600)

    def test_sampler_without_block_targets(self):
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        sampler = SameNlocBatchSampler(reader, shuffle=False)
        all_indices = [i for batch in sampler for i in batch]
        self.assertEqual(sorted(all_indices), list(range(600)))


# ============================================================
# Neighbor stat from LMDB tests
# ============================================================


class TestLmdbNeighborStat(unittest.TestCase):
    """Test make_neighbor_stat_data interface and sampling."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._lmdb_path = _create_grid_lmdb(f"{cls._tmpdir.name}/grid.lmdb", nframes=3)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_make_neighbor_stat_data_interface(self):
        data = make_neighbor_stat_data(self._lmdb_path, ["TYPE", "NO_TYPE"])
        self.assertIsInstance(data.system_dirs, list)
        self.assertGreater(len(data.system_dirs), 0)
        self.assertEqual(data.get_ntypes(), 2)
        data.get_batch()  # no-op
        sys0 = data.data_systems[0]
        self.assertIsInstance(sys0.pbc, bool)
        set_data = sys0._load_set(sys0.dirs[0])
        self.assertEqual(set_data["coord"].ndim, 2)
        self.assertEqual(set_data["coord"].shape[1], sys0.get_natoms() * 3)

    def test_sampling_large_dataset(self):
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_grid_lmdb(f"{tmpdir.name}/large.lmdb", nframes=50)
        data = make_neighbor_stat_data(path, ["TYPE"], max_frames=10)
        total = sum(s._load_set(s.dirs[0])["coord"].shape[0] for s in data.data_systems)
        self.assertEqual(total, 10)
        tmpdir.cleanup()


def _create_lmdb_with_extra_keys(
    path: str, nframes: int = 5, natoms: int = 6, extra_keys: dict | None = None
) -> str:
    """Create a test LMDB with extra per-frame keys (e.g. atom_pref, fparam).

    Parameters
    ----------
    extra_keys : dict
        key -> (shape_fn, dtype) where shape_fn(natoms) returns the array shape.
        Example: {"atom_pref": (lambda n: (n,), np.float64)}
    """
    n_type0 = max(1, natoms // 3)
    n_type1 = natoms - n_type0
    extra_keys = extra_keys or {}
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": nframes,
            "frame_idx_fmt": "012d",
            "type_map": ["O", "H"],
            "system_info": {"natoms": [n_type0, n_type1]},
        }
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        rng = np.random.RandomState(0)
        for i in range(nframes):
            frame = _make_frame(natoms=natoms, seed=i)
            for ek, (shape_fn, dtype) in extra_keys.items():
                arr = rng.rand(*shape_fn(natoms)).astype(dtype)
                frame[ek] = {
                    "type": str(arr.dtype),
                    "shape": list(arr.shape),
                    "data": arr.tobytes(),
                }
            txn.put(
                format(i, "012d").encode(),
                msgpack.packb(frame, use_bin_type=True),
            )
    env.close()
    return path


# ============================================================
# Dynamic find_* and repeat tests
# ============================================================


class TestDynamicKeysAndRepeat(unittest.TestCase):
    """Test auto-discovery of find_* flags and repeat handling."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._natoms = 6
        cls._nframes = 5
        cls._lmdb_path = _create_lmdb_with_extra_keys(
            f"{cls._tmpdir.name}/extra.lmdb",
            nframes=cls._nframes,
            natoms=cls._natoms,
            extra_keys={
                "atom_pref": (lambda n: (n,), np.float64),
                "fparam": (lambda n: (3,), np.float64),
            },
        )
        cls._type_map = ["O", "H"]

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    # --- LmdbDataReader ---

    def test_reader_find_flags_auto_detected(self):
        """Extra keys in frame get find_*=1.0 automatically."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map)
        frame = reader[0]
        self.assertEqual(frame["find_atom_pref"], np.float32(1.0))
        self.assertEqual(frame["find_fparam"], np.float32(1.0))
        self.assertEqual(frame["find_energy"], np.float32(1.0))
        # Keys not in frame get find_*=0.0
        self.assertEqual(frame["find_aparam"], np.float32(0.0))
        self.assertEqual(frame["find_spin"], np.float32(0.0))

    def test_reader_repeat_applied(self):
        """DataRequirementItem with repeat=3 expands atom_pref from (natoms,) to (natoms*3,)."""
        from deepmd.utils.data import (
            DataRequirementItem,
        )

        reader = LmdbDataReader(self._lmdb_path, self._type_map)
        reader.add_data_requirement(
            [
                DataRequirementItem(
                    "atom_pref",
                    ndof=1,
                    atomic=True,
                    must=False,
                    high_prec=False,
                    repeat=3,
                ),
            ]
        )
        frame = reader[0]
        self.assertEqual(frame["atom_pref"].shape, (self._natoms * 3,))

    def test_reader_repeat_default_fill(self):
        """Missing key with repeat fills correct shape."""
        from deepmd.utils.data import (
            DataRequirementItem,
        )

        reader = LmdbDataReader(self._lmdb_path, self._type_map)
        reader.add_data_requirement(
            [
                DataRequirementItem(
                    "drdq", ndof=6, atomic=True, must=False, high_prec=False, repeat=2
                ),
            ]
        )
        frame = reader[0]
        self.assertEqual(frame["find_drdq"], np.float32(0.0))
        self.assertEqual(frame["drdq"].shape, (self._natoms * 6 * 2,))

    # --- LmdbTestData ---

    def test_testdata_find_flags_auto_detected(self):
        """LmdbTestData.get_test() discovers extra keys dynamically."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        result = td.get_test()
        self.assertEqual(result["find_atom_pref"], 1.0)
        self.assertEqual(result["find_fparam"], 1.0)
        self.assertIn("atom_pref", result)
        self.assertIn("fparam", result)

    def test_testdata_repeat_applied(self):
        """LmdbTestData respects repeat=3 for atom_pref."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        td.add("atom_pref", 1, atomic=True, must=False, high_prec=False, repeat=3)
        result = td.get_test()
        self.assertEqual(
            result["atom_pref"].shape,
            (self._nframes, self._natoms * 3),
        )

    def test_testdata_missing_key_not_found(self):
        """Keys absent from LMDB frames get find_*=0.0 in get_test()."""
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb(f"{tmpdir.name}/plain.lmdb", nframes=3, natoms=6)
        td = LmdbTestData(path, type_map=["O", "H"], shuffle_test=False)
        result = td.get_test()
        # atom_pref is not in the plain LMDB
        self.assertEqual(result.get("find_atom_pref", 0.0), 0.0)
        tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
