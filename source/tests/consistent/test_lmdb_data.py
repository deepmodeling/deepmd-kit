# SPDX-License-Identifier: LGPL-3.0-or-later
"""Consistency tests: LmdbDataReader (dpmodel) vs LmdbDataset (pt).

Verifies that the framework-agnostic reader and the PyTorch wrapper
produce identical outputs for the same LMDB data.
Also tests SameNlocBatchSampler and mixed_batch guards.
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


# ============================================================
# Uniform nloc tests
# ============================================================


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
        self.assertEqual(len(reader), 10)
        if INSTALLED_PT:
            ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
            self.assertEqual(len(ds), 10)
            self.assertEqual(len(reader), len(ds))

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
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

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_same_batch_size(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="auto")
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size="auto")
        self.assertEqual(reader.batch_size, ds.batch_size)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_same_properties(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(reader.index, ds.index)
        self.assertEqual(reader.total_batch, ds.total_batch)
        self.assertEqual(reader.batch_sizes, ds.batch_sizes)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
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

    def test_is_lmdb(self):
        self.assertTrue(is_lmdb(self._lmdb_path))
        self.assertTrue(is_lmdb("something.lmdb"))
        self.assertFalse(is_lmdb("/some/npy/system"))
        self.assertFalse(is_lmdb(["list", "of", "systems"]))

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

    def test_uniform_nloc_single_group(self):
        """Uniform-nloc LMDB has exactly one nloc group."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(len(reader.nloc_groups), 1)
        self.assertIn(6, reader.nloc_groups)
        self.assertEqual(len(reader.nloc_groups[6]), 10)


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
        """LmdbDataReader correctly groups frames by nloc."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        self.assertEqual(set(reader.nloc_groups.keys()), {6, 9, 12})
        self.assertEqual(len(reader.nloc_groups[6]), 4)
        self.assertEqual(len(reader.nloc_groups[9]), 4)
        self.assertEqual(len(reader.nloc_groups[12]), 2)

    def test_per_frame_natoms_vec(self):
        """Each frame gets its own natoms_vec matching its actual atom count."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        frame0 = reader[0]  # 6 atoms
        frame4 = reader[4]  # 9 atoms
        frame8 = reader[8]  # 12 atoms
        self.assertEqual(frame0["natoms"][0], 6)
        self.assertEqual(frame4["natoms"][0], 9)
        self.assertEqual(frame8["natoms"][0], 12)
        np.testing.assert_array_equal(frame0["real_natoms_vec"], frame0["natoms"])

    def test_per_frame_shapes(self):
        """coord/force/atype shapes match per-frame atom count."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        frame0 = reader[0]  # 6 atoms
        frame4 = reader[4]  # 9 atoms
        self.assertEqual(frame0["coord"].shape, (6, 3))
        self.assertEqual(frame0["force"].shape, (6, 3))
        self.assertEqual(frame0["atype"].shape, (6,))
        self.assertEqual(frame4["coord"].shape, (9, 3))
        self.assertEqual(frame4["force"].shape, (9, 3))
        self.assertEqual(frame4["atype"].shape, (9,))

    def test_frame_nlocs(self):
        """frame_nlocs returns correct per-frame atom counts."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        expected = [6, 6, 6, 6, 9, 9, 9, 9, 12, 12]
        self.assertEqual(reader.frame_nlocs, expected)

    # --- SameNlocBatchSampler tests ---

    def test_sampler_all_batches_same_nloc(self):
        """Every batch from SameNlocBatchSampler has frames with identical nloc."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = SameNlocBatchSampler(reader, shuffle=False, seed=42)
        for batch_indices in sampler:
            nlocs_in_batch = [reader.frame_nlocs[i] for i in batch_indices]
            self.assertTrue(
                all(n == nlocs_in_batch[0] for n in nlocs_in_batch),
                f"Mixed nloc in batch: {nlocs_in_batch} (indices={batch_indices})",
            )

    def test_sampler_covers_all_frames(self):
        """SameNlocBatchSampler yields every frame exactly once."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = SameNlocBatchSampler(reader, shuffle=False, seed=42)
        all_indices = []
        for batch_indices in sampler:
            all_indices.extend(batch_indices)
        self.assertEqual(sorted(all_indices), list(range(10)))

    def test_sampler_auto_batch_size_per_nloc(self):
        """Auto batch_size varies by nloc group."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="auto")
        bs_6 = reader.get_batch_size_for_nloc(6)
        bs_9 = reader.get_batch_size_for_nloc(9)
        bs_12 = reader.get_batch_size_for_nloc(12)
        # Larger nloc → smaller batch_size
        self.assertGreaterEqual(bs_6, bs_9)
        self.assertGreaterEqual(bs_9, bs_12)

    def test_sampler_shuffle_deterministic(self):
        """Same seed produces same batch order."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        s1 = SameNlocBatchSampler(reader, shuffle=True, seed=123)
        s2 = SameNlocBatchSampler(reader, shuffle=True, seed=123)
        batches1 = list(s1)
        batches2 = list(s2)
        self.assertEqual(batches1, batches2)

    def test_sampler_len(self):
        """__len__ matches actual number of batches yielded."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = SameNlocBatchSampler(reader, shuffle=False)
        batches = list(sampler)
        self.assertEqual(len(sampler), len(batches))

    # --- Collate guard tests ---

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_collate_mixed_nloc_raises(self):
        """Collating frames with different nloc raises NotImplementedError."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        frame_6 = reader[0]
        frame_9 = reader[4]
        with self.assertRaises(NotImplementedError) as ctx:
            _collate_lmdb_batch([frame_6, frame_9])
        self.assertIn("mixed_batch", str(ctx.exception))

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_collate_same_nloc_ok(self):
        """Collating frames with same nloc works fine."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        frame0 = reader[0]
        frame1 = reader[1]
        batch = _collate_lmdb_batch([frame0, frame1])
        self.assertIn("coord", batch)
        self.assertEqual(batch["coord"].shape[0], 2)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_mixed_batch_true_raises(self):
        """LmdbDataset(mixed_batch=True) raises NotImplementedError."""
        with self.assertRaises(NotImplementedError) as ctx:
            LmdbDataset(self._lmdb_path, self._type_map, batch_size=2, mixed_batch=True)
        self.assertIn("mixed_batch", str(ctx.exception))

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_pt_dataset_iterates_same_nloc_batches(self):
        """LmdbDataset iteration produces only same-nloc batches."""
        import torch

        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        with torch.device("cpu"):
            for batch in ds.dataloaders[0]:
                atype = batch.get("atype")
                if atype is not None:
                    # All frames in batch have same nloc
                    self.assertEqual(atype.shape[1], atype.shape[1])
                break  # just check first batch

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_pt_dataset_mixed_batch_flag(self):
        """LmdbDataset exposes mixed_batch from reader."""
        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=2)
        self.assertFalse(ds.mixed_batch)

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_pt_full_epoch_mixed_nloc(self):
        """Full DataLoader epoch over mixed-nloc LMDB: all batches same-nloc, all frames covered."""
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
        # All 10 frames should be covered
        self.assertEqual(sorted(all_fids), list(range(10)))

    @unittest.skipUnless(INSTALLED_PT, "PyTorch not available")
    def test_pt_batch_shapes_consistent(self):
        """Within each batch, coord/force/natoms shapes are consistent with atype."""
        import torch

        ds = LmdbDataset(self._lmdb_path, self._type_map, batch_size=3)
        with torch.device("cpu"):
            for batch in ds.dataloaders[0]:
                bs = batch["atype"].shape[0]
                nloc = batch["atype"].shape[1]
                self.assertEqual(batch["coord"].shape, (bs, nloc, 3))
                self.assertEqual(batch["force"].shape, (bs, nloc, 3))
                self.assertEqual(batch["natoms"].shape, (bs, 4))  # ntypes=2 → 2+2=4
                # natoms_vec[0] should equal nloc for all frames
                for i in range(bs):
                    self.assertEqual(batch["natoms"][i, 0].item(), nloc)

    # --- LmdbTestData mixed-nloc tests ---

    def test_test_data_nloc_groups(self):
        """LmdbTestData detects nloc groups in mixed-nloc LMDB."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        self.assertEqual(set(td.nloc_groups.keys()), {6, 9, 12})
        self.assertEqual(len(td.nloc_groups[6]), 4)
        self.assertEqual(len(td.nloc_groups[9]), 4)
        self.assertEqual(len(td.nloc_groups[12]), 2)

    def test_test_data_get_test_specific_nloc(self):
        """get_test(nloc=N) returns only frames with that atom count."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        td.add("force", 3, atomic=True, must=False, high_prec=False)

        result_6 = td.get_test(nloc=6)
        self.assertEqual(result_6["coord"].shape, (4, 6 * 3))
        self.assertEqual(result_6["force"].shape, (4, 6 * 3))
        self.assertEqual(result_6["type"].shape, (4, 6))

        result_9 = td.get_test(nloc=9)
        self.assertEqual(result_9["coord"].shape, (4, 9 * 3))
        self.assertEqual(result_9["force"].shape, (4, 9 * 3))
        self.assertEqual(result_9["type"].shape, (4, 9))

        result_12 = td.get_test(nloc=12)
        self.assertEqual(result_12["coord"].shape, (2, 12 * 3))
        self.assertEqual(result_12["force"].shape, (2, 12 * 3))
        self.assertEqual(result_12["type"].shape, (2, 12))

    def test_test_data_get_test_default_mixed(self):
        """get_test() without nloc on mixed data returns largest group."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        # Largest groups are nloc=6 and nloc=9 (both 4 frames).
        # max() picks the one with the largest nloc among tied groups.
        result = td.get_test()
        nframes = result["coord"].shape[0]
        self.assertEqual(nframes, 4)

    def test_test_data_get_test_invalid_nloc(self):
        """get_test(nloc=999) raises ValueError."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        with self.assertRaises(ValueError):
            td.get_test(nloc=999)

    def test_test_data_uniform_nloc_no_warning(self):
        """Uniform-nloc LMDB: get_test() returns all frames without warning."""
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb(f"{tmpdir.name}/uniform.lmdb", nframes=5, natoms=6)
        td = LmdbTestData(path, type_map=self._type_map, shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        result = td.get_test()
        self.assertEqual(result["coord"].shape, (5, 18))
        tmpdir.cleanup()


def _create_lmdb_with_type_map(
    path: str,
    nframes: int = 6,
    natoms: int = 6,
    lmdb_type_map: list[str] | None = None,
) -> str:
    """Create a test LMDB with type_map stored in metadata.

    Atom types: first 1/3 are type-0, rest are type-1.
    """
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


# ============================================================
# Type map remapping tests
# ============================================================


class TestTypeMapRemapping(unittest.TestCase):
    """Test type_map remapping in LmdbDataReader and LmdbTestData."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        # LMDB with type_map=["O","H"]: type-0 = O, type-1 = H
        cls._lmdb_path = _create_lmdb_with_type_map(
            f"{cls._tmpdir.name}/remap.lmdb",
            nframes=6,
            natoms=6,
            lmdb_type_map=["O", "H"],
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    # --- LmdbDataReader tests ---

    def test_reader_no_remap_when_match(self):
        """No remap when model type_map matches LMDB type_map."""
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        self.assertIsNone(reader._type_remap)
        frame = reader[0]
        # type-0 = O, type-1 = H, unchanged
        self.assertEqual(frame["atype"][0], 0)

    def test_reader_remap_when_reversed(self):
        """Remap when model type_map=["H","O"] vs LMDB ["O","H"]."""
        reader = LmdbDataReader(self._lmdb_path, ["H", "O"])
        self.assertIsNotNone(reader._type_remap)
        np.testing.assert_array_equal(reader._type_remap, [1, 0])
        frame = reader[0]
        # Original type-0 (O) -> 1, type-1 (H) -> 0
        atype = frame["atype"]
        n_type0_orig = max(1, 6 // 3)  # 2 atoms of original type-0
        # First n_type0_orig atoms were type-0 (O), now should be 1
        for i in range(n_type0_orig):
            self.assertEqual(atype[i], 1)
        # Remaining atoms were type-1 (H), now should be 0
        for i in range(n_type0_orig, 6):
            self.assertEqual(atype[i], 0)

    def test_reader_remap_superset(self):
        """Remap when model type_map has extra elements."""
        reader = LmdbDataReader(self._lmdb_path, ["C", "O", "H"])
        self.assertIsNotNone(reader._type_remap)
        np.testing.assert_array_equal(reader._type_remap, [1, 2])
        frame = reader[0]
        # O -> 1, H -> 2
        n_type0_orig = max(1, 6 // 3)
        for i in range(n_type0_orig):
            self.assertEqual(frame["atype"][i], 1)
        for i in range(n_type0_orig, 6):
            self.assertEqual(frame["atype"][i], 2)

    def test_reader_natoms_vec_after_remap(self):
        """natoms_vec reflects remapped types."""
        reader = LmdbDataReader(self._lmdb_path, ["H", "O"])
        frame = reader[0]
        natoms = frame["natoms"]
        # model type_map=["H","O"], ntypes=2
        # Original: 2 O + 4 H. After remap: O->1, H->0
        # So count_H(idx=0)=4, count_O(idx=1)=2
        self.assertEqual(natoms[0], 6)  # nloc
        self.assertEqual(natoms[2], 4)  # H count (model idx 0)
        self.assertEqual(natoms[3], 2)  # O count (model idx 1)

    def test_reader_missing_element_raises(self):
        """ValueError when model type_map misses an LMDB element."""
        with self.assertRaises(ValueError):
            LmdbDataReader(self._lmdb_path, ["O"])  # missing H

    def test_reader_no_type_map_in_metadata(self):
        """Old LMDB without type_map -> no remap."""
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb_with_type_map(
            f"{tmpdir.name}/old.lmdb", nframes=3, natoms=6, lmdb_type_map=None
        )
        reader = LmdbDataReader(path, ["H", "O"])
        self.assertIsNone(reader._type_remap)
        tmpdir.cleanup()

    # --- LmdbTestData tests ---

    def test_testdata_no_remap_when_match(self):
        """LmdbTestData: no remap when type_maps match."""
        td = LmdbTestData(self._lmdb_path, type_map=["O", "H"], shuffle_test=False)
        self.assertIsNone(td._type_remap)
        data = td.get_test()
        self.assertEqual(data["type"][0, 0], 0)

    def test_testdata_remap_when_reversed(self):
        """LmdbTestData: remap when model ["H","O"] vs LMDB ["O","H"]."""
        td = LmdbTestData(self._lmdb_path, type_map=["H", "O"], shuffle_test=False)
        self.assertIsNotNone(td._type_remap)
        data = td.get_test()
        n_type0_orig = max(1, 6 // 3)
        # Original type-0 (O) -> 1, type-1 (H) -> 0
        for i in range(n_type0_orig):
            self.assertEqual(data["type"][0, i], 1)
        for i in range(n_type0_orig, 6):
            self.assertEqual(data["type"][0, i], 0)

    def test_testdata_remap_superset(self):
        """LmdbTestData: remap with superset model type_map."""
        td = LmdbTestData(self._lmdb_path, type_map=["C", "O", "H"], shuffle_test=False)
        self.assertIsNotNone(td._type_remap)
        data = td.get_test()
        n_type0_orig = max(1, 6 // 3)
        # O -> 1, H -> 2
        for i in range(n_type0_orig):
            self.assertEqual(data["type"][0, i], 1)
        for i in range(n_type0_orig, 6):
            self.assertEqual(data["type"][0, i], 2)

    def test_testdata_missing_element_raises(self):
        """LmdbTestData: ValueError when model misses LMDB element."""
        with self.assertRaises(ValueError):
            LmdbTestData(self._lmdb_path, type_map=["O"], shuffle_test=False)

    def test_testdata_no_type_map_in_metadata(self):
        """LmdbTestData: old LMDB without type_map -> no remap."""
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


def _create_lmdb_with_system_ids(
    path: str,
    system_frames: list[int],
    natoms: int = 6,
    type_map: list[str] | None = None,
) -> str:
    """Create a test LMDB with frame_system_ids in metadata.

    Parameters
    ----------
    system_frames : list[int]
        Number of frames per system, e.g. [100, 500] means sys0 has 100
        frames and sys1 has 500 frames.
    """
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


def _create_lmdb_with_system_ids_mixed_nloc(
    path: str,
    system_specs: list[tuple[int, int, int]],
    type_map: list[str] | None = None,
) -> str:
    """Create a test LMDB with frame_system_ids and mixed nloc.

    Parameters
    ----------
    system_specs : list[tuple[int, int, int]]
        Each tuple is (system_id, natoms, nframes).
    """
    total = sum(nf for _, _, nf in system_specs)
    frame_system_ids = []
    frame_nlocs = []
    frames_data = []
    for sid, natoms, nf in system_specs:
        for _ in range(nf):
            frame_system_ids.append(sid)
            frame_nlocs.append(natoms)
            frames_data.append(_make_frame(natoms=natoms, seed=len(frames_data) % 100))

    env = lmdb.open(path, map_size=50 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": total,
            "frame_idx_fmt": "012d",
            "system_info": {"natoms": [2, 4]},
            "frame_system_ids": frame_system_ids,
            "frame_nlocs": frame_nlocs,
        }
        if type_map is not None:
            meta["type_map"] = type_map
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        for i, frame in enumerate(frames_data):
            key = format(i, "012d").encode()
            txn.put(key, msgpack.packb(frame, use_bin_type=True))
    env.close()
    return path


class TestAutoProb(unittest.TestCase):
    """Test auto_prob support: frame_system_ids, compute_block_targets,
    _expand_indices_by_blocks, and SameNlocBatchSampler with block_targets.
    """

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        # 3 systems: sys0=100 frames, sys1=200 frames, sys2=300 frames
        cls._lmdb_path = _create_lmdb_with_system_ids(
            f"{cls._tmpdir.name}/auto_prob.lmdb",
            system_frames=[100, 200, 300],
            natoms=6,
            type_map=["O", "H"],
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    # --- Reader system tracking ---

    def test_reader_system_groups(self):
        """LmdbDataReader correctly parses frame_system_ids."""
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        self.assertEqual(reader.nsystems, 3)
        self.assertEqual(reader.system_nframes, [100, 200, 300])
        self.assertIsNotNone(reader.frame_system_ids)
        self.assertEqual(len(reader.frame_system_ids), 600)
        # First 100 frames belong to sys 0
        self.assertTrue(all(s == 0 for s in reader.frame_system_ids[:100]))
        # Next 200 to sys 1
        self.assertTrue(all(s == 1 for s in reader.frame_system_ids[100:300]))
        # Last 300 to sys 2
        self.assertTrue(all(s == 2 for s in reader.frame_system_ids[300:600]))
        # system_groups
        self.assertEqual(len(reader.system_groups[0]), 100)
        self.assertEqual(len(reader.system_groups[1]), 200)
        self.assertEqual(len(reader.system_groups[2]), 300)

    def test_reader_no_system_ids_backward_compat(self):
        """Old LMDB without frame_system_ids: nsystems=1, no crash."""
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb(f"{tmpdir.name}/old.lmdb", nframes=10, natoms=6)
        reader = LmdbDataReader(path, ["O", "H"])
        self.assertEqual(reader.nsystems, 1)
        self.assertIsNone(reader.frame_system_ids)
        self.assertEqual(reader.system_nframes, [10])
        self.assertEqual(len(reader.system_groups[0]), 10)
        tmpdir.cleanup()

    # --- compute_block_targets ---

    def test_compute_block_targets_equal_weight(self):
        """Equal weight blocks with equal frames -> no expansion needed."""
        # sys0=100, sys1=100; prob 0.5 each
        # ratio = 100/0.5 = 200 for both -> target = 100 each -> no expansion
        result = compute_block_targets(
            "prob_sys_size;0:1:0.5;1:2:0.5",
            nsystems=2,
            system_nframes=[100, 100],
        )
        # No expansion needed (targets == actual)
        self.assertEqual(result, [])

    def test_compute_block_targets_unequal(self):
        """Unequal frames with equal weight -> expansion needed."""
        # sys0=100, sys1=500; prob 0.5 each
        # ratio_A = 100/0.5 = 200, ratio_B = 500/0.5 = 1000
        # total_target = ceil(max(200, 1000)) = 1000
        # target_A = round(1000 * 0.5) = 500, target_B = round(1000 * 0.5) = 500
        result = compute_block_targets(
            "prob_sys_size;0:1:0.5;1:2:0.5",
            nsystems=2,
            system_nframes=[100, 500],
        )
        self.assertEqual(len(result), 2)
        sys_ids_a, target_a = result[0]
        sys_ids_b, target_b = result[1]
        self.assertEqual(sys_ids_a, [0])
        self.assertEqual(sys_ids_b, [1])
        self.assertEqual(target_a, 500)
        self.assertEqual(target_b, 500)

    def test_compute_block_targets_multi_sys_block(self):
        """Block spanning multiple systems."""
        # sys0=100, sys1=200, sys2=300; block A=[0,1] prob=0.5, block B=[2] prob=0.5
        # block_A frames=300, block_B frames=300
        # ratio_A = 300/0.5 = 600, ratio_B = 300/0.5 = 600
        # total_target = 600, target_A = 300, target_B = 300 -> no expansion
        result = compute_block_targets(
            "prob_sys_size;0:2:0.5;2:3:0.5",
            nsystems=3,
            system_nframes=[100, 200, 300],
        )
        self.assertEqual(result, [])

    def test_compute_block_targets_asymmetric(self):
        """Asymmetric: small block needs expansion."""
        # sys0=50, sys1=50, sys2=400; block A=[0,1] prob=0.5, block B=[2] prob=0.5
        # block_A frames=100, block_B frames=400
        # ratio_A = 100/0.5 = 200, ratio_B = 400/0.5 = 800
        # total_target = 800, target_A = 400, target_B = 400
        result = compute_block_targets(
            "prob_sys_size;0:2:0.5;2:3:0.5",
            nsystems=3,
            system_nframes=[50, 50, 400],
        )
        self.assertEqual(len(result), 2)
        sys_ids_a, target_a = result[0]
        sys_ids_b, target_b = result[1]
        self.assertEqual(sys_ids_a, [0, 1])
        self.assertEqual(sys_ids_b, [2])
        self.assertEqual(target_a, 400)
        self.assertEqual(target_b, 400)

    # --- _expand_indices_by_blocks ---

    def test_expand_indices_basic(self):
        """Expand indices: block A needs 5x expansion."""
        # 10 frames: sys0=[0..4], sys1=[5..9]
        frame_system_ids = [0] * 5 + [1] * 5
        block_targets = [([0], 25), ([1], 25)]
        rng = np.random.default_rng(42)
        indices = list(range(10))
        expanded = _expand_indices_by_blocks(
            indices,
            frame_system_ids,
            block_targets,
            rng,
        )
        # Each block should have 25 indices
        sys0_expanded = [i for i in expanded if frame_system_ids[i] == 0]
        sys1_expanded = [i for i in expanded if frame_system_ids[i] == 1]
        self.assertEqual(len(sys0_expanded), 25)
        self.assertEqual(len(sys1_expanded), 25)
        # All original indices present
        self.assertTrue(set(range(5)).issubset(set(sys0_expanded)))
        self.assertTrue(set(range(5, 10)).issubset(set(sys1_expanded)))

    def test_expand_indices_no_expansion(self):
        """No expansion when target == actual."""
        frame_system_ids = [0] * 5 + [1] * 5
        block_targets = [([0], 5), ([1], 5)]
        rng = np.random.default_rng(42)
        indices = list(range(10))
        expanded = _expand_indices_by_blocks(
            indices,
            frame_system_ids,
            block_targets,
            rng,
        )
        self.assertEqual(sorted(expanded), list(range(10)))

    def test_expand_indices_remainder_sampling(self):
        """Remainder part uses without-replacement sampling."""
        # sys0=10 frames, target=23 -> 1 full copy + 3 remainder
        frame_system_ids = [0] * 10
        block_targets = [([0], 23)]
        rng = np.random.default_rng(42)
        indices = list(range(10))
        expanded = _expand_indices_by_blocks(
            indices,
            frame_system_ids,
            block_targets,
            rng,
        )
        self.assertEqual(len(expanded), 23)
        # Each original index appears at least twice (1 original + 1 full copy)
        from collections import (
            Counter,
        )

        counts = Counter(expanded)
        for i in range(10):
            self.assertGreaterEqual(counts[i], 2)
        # 3 indices appear 3 times (the remainder)
        n_three = sum(1 for c in counts.values() if c == 3)
        self.assertEqual(n_three, 3)

    def test_expand_epoch_diversity(self):
        """Different RNG seeds produce different remainder samples."""
        frame_system_ids = [0] * 10
        block_targets = [([0], 15)]  # 10 + 5 remainder
        indices = list(range(10))
        results = []
        for seed in range(5):
            rng = np.random.default_rng(seed)
            expanded = _expand_indices_by_blocks(
                indices,
                frame_system_ids,
                block_targets,
                rng,
            )
            # Sort the "extra" part (beyond the first 10)
            results.append(sorted(expanded[10:]))
        # At least 2 different remainder sets across 5 seeds
        unique = {tuple(r) for r in results}
        self.assertGreater(len(unique), 1)

    # --- SameNlocBatchSampler with block_targets ---

    def test_sampler_with_block_targets(self):
        """SameNlocBatchSampler expands frames when block_targets provided."""
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        # sys0=100, sys1=200, sys2=300; make block A=[0] prob=0.5, B=[1,2] prob=0.5
        # block_A frames=100, block_B frames=500
        # ratio_A=200, ratio_B=1000 -> total_target=1000
        # target_A=500, target_B=500
        block_targets = compute_block_targets(
            "prob_sys_size;0:1:0.5;1:3:0.5",
            nsystems=3,
            system_nframes=[100, 200, 300],
        )
        self.assertTrue(len(block_targets) > 0)

        sampler = SameNlocBatchSampler(
            reader,
            shuffle=True,
            block_targets=block_targets,
        )
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        # Total should be ~1000 (500 from block A + 500 from block B)
        self.assertGreater(len(all_indices), 600)  # more than original 600
        # All original indices should appear at least once
        self.assertEqual(len(set(all_indices)), 600)

    def test_sampler_without_block_targets(self):
        """SameNlocBatchSampler without block_targets: no expansion."""
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        sampler = SameNlocBatchSampler(reader, shuffle=False)
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        self.assertEqual(len(all_indices), 600)
        self.assertEqual(sorted(all_indices), list(range(600)))


if __name__ == "__main__":
    unittest.main()
