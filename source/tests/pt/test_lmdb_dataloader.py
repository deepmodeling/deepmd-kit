# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for LmdbDataset."""

import lmdb
import msgpack
import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.lmdb_data import (
    DistributedSameNlocBatchSampler,
    LmdbDataReader,
    LmdbTestData,
    SameNlocBatchSampler,
    _decode_frame,
    _read_metadata,
    _remap_keys,
    is_lmdb,
)
from deepmd.pt.utils.lmdb_dataset import (
    LmdbDataset,
    _collate_lmdb_batch,
)
from deepmd.utils.data import (
    DataRequirementItem,
)


def _make_frame(natoms: int = 6, seed: int = 0) -> dict:
    """Create a synthetic frame dict as stored in LMDB."""
    rng = np.random.RandomState(seed)

    def _encode_array(arr: np.ndarray) -> dict:
        return {
            "nd": None,
            "type": str(arr.dtype),
            "kind": "",
            "shape": list(arr.shape),
            "data": arr.tobytes(),
        }

    return {
        "atom_numbs": [natoms // 2, natoms // 2],
        "atom_names": ["O", "H"],
        "atom_types": _encode_array(
            np.array([0] * (natoms // 2) + [1] * (natoms // 2), dtype=np.int64)
        ),
        "orig": _encode_array(np.zeros(3, dtype=np.float64)),
        "cells": _encode_array(rng.randn(3, 3).astype(np.float32)),
        "coords": _encode_array(rng.randn(natoms, 3).astype(np.float32)),
        "energies": _encode_array(np.array(rng.randn(), dtype=np.float32)),
        "forces": _encode_array(rng.randn(natoms, 3).astype(np.float32)),
    }


def _create_test_lmdb(path: str, nframes: int = 10, natoms: int = 6) -> None:
    """Create a minimal LMDB dataset for testing."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    metadata = {
        "nframes": nframes,
        "frame_idx_fmt": fmt,
        "system_info": {
            "formula": f"O{natoms // 2}H{natoms // 2}",
            "natoms": [natoms // 2, natoms // 2],
            "nframes": nframes,
        },
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for i in range(nframes):
            key = format(i, fmt).encode()
            frame = _make_frame(natoms=natoms, seed=i)
            txn.put(key, msgpack.packb(frame, use_bin_type=True))
    env.close()


@pytest.fixture
def lmdb_dir(tmp_path):
    """Create a temporary LMDB dataset."""
    lmdb_path = str(tmp_path / "test.lmdb")
    _create_test_lmdb(lmdb_path, nframes=10, natoms=6)
    return lmdb_path


class TestHelpers:
    """Test helper functions."""

    def test_read_metadata(self, lmdb_dir):
        env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            meta = _read_metadata(txn)
        assert meta["nframes"] == 10
        assert "system_info" in meta
        env.close()

    def test_read_metadata_missing(self, tmp_path):
        empty_path = str(tmp_path / "empty.lmdb")
        env = lmdb.open(empty_path, map_size=1024 * 1024)
        env.close()
        env = lmdb.open(empty_path, readonly=True, lock=False)
        with env.begin() as txn:
            with pytest.raises(ValueError, match="missing __metadata__"):
                _read_metadata(txn)
        env.close()

    def test_decode_frame(self, lmdb_dir):
        env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            raw = txn.get(format(0, "012d").encode())
        frame = _decode_frame(raw)
        assert "coords" in frame
        assert "forces" in frame
        assert isinstance(frame["coords"], np.ndarray)
        assert frame["coords"].shape == (6, 3)
        env.close()

    def test_remap_keys(self):
        frame = {
            "coords": np.zeros((3, 3)),
            "cells": np.zeros((3, 3)),
            "energies": np.array(1.0),
            "forces": np.zeros((3, 3)),
            "atom_types": np.array([0, 1, 0]),
            "custom_key": np.array([42.0]),
        }
        remapped = _remap_keys(frame)
        assert "coord" in remapped
        assert "box" in remapped
        assert "energy" in remapped
        assert "force" in remapped
        assert "atype" in remapped
        assert "custom_key" in remapped  # pass-through
        assert "coords" not in remapped

    def test_is_lmdb(self, lmdb_dir, tmp_path):
        assert is_lmdb(lmdb_dir) is True
        assert is_lmdb(str(tmp_path / "nonexistent.lmdb")) is True  # ends with .lmdb
        assert is_lmdb(str(tmp_path / "nope")) is False
        assert is_lmdb(["a", "b"]) is False
        assert is_lmdb(42) is False


class TestLmdbDataset:
    """Test LmdbDataset class."""

    def test_len(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert len(ds) == 10

    def test_getitem_keys(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        frame = ds[0]
        # Required keys
        assert "coord" in frame
        assert "box" in frame
        assert "energy" in frame
        assert "force" in frame
        assert "atype" in frame
        assert "natoms" in frame
        assert "fid" in frame
        # find_* flags
        assert "find_energy" in frame
        assert "find_force" in frame
        assert frame["find_energy"] == 1.0
        assert frame["find_force"] == 1.0
        assert frame["find_virial"] == 0.0
        # Metadata keys removed
        assert "atom_numbs" not in frame
        assert "atom_names" not in frame
        assert "orig" not in frame

    def test_getitem_shapes(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        frame = ds[0]
        assert frame["coord"].shape == (6, 3)
        assert frame["box"].shape == (9,)
        assert frame["energy"].shape == (1,)
        assert frame["force"].shape == (6, 3)
        assert frame["atype"].shape == (6,)
        assert frame["natoms"].shape == (4,)  # [natoms, natoms, nO, nH]

    def test_getitem_dtypes(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        frame = ds[0]
        assert frame["coord"].dtype == np.float64
        assert frame["box"].dtype == np.float64
        assert frame["energy"].dtype == np.float64
        assert frame["force"].dtype == np.float64
        assert frame["atype"].dtype == np.int64

    def test_getitem_out_of_range(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        with pytest.raises(IndexError):
            ds[999]

    def test_natoms_vec(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        frame = ds[0]
        natoms = frame["natoms"]
        assert natoms[0] == 6  # total natoms
        assert natoms[1] == 6  # total natoms (repeated)
        assert natoms[2] == 3  # O count
        assert natoms[3] == 3  # H count

    def test_auto_batch_size(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size="auto")
        # rule=32, natoms=6, 32//6=5, 5*6=30<32 → 6
        assert ds.batch_size == 6

    def test_auto_batch_size_with_rule(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size="auto:12")
        # rule=12, natoms=6, 12//6=2, 2*6=12 → 2
        assert ds.batch_size == 2

    def test_int_batch_size(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=3)
        assert ds.batch_size == 3


class TestTrainerInterface:
    """Test Trainer compatibility interface."""

    def test_systems(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert len(ds.systems) == 1
        assert ds.systems[0] is ds

    def test_dataloaders(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert len(ds.dataloaders) == 1

    def test_index(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert ds.index == [5]  # 10 frames / 2 batch_size

    def test_total_batch(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert ds.total_batch == 5

    def test_batch_sizes(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert ds.batch_sizes == [2]

    def test_sampler_list(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert len(ds.sampler_list) == 1

    def test_add_data_requirement(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        req = [
            DataRequirementItem("virial", 9, atomic=False, must=False, default=0.0),
        ]
        ds.add_data_requirement(req)
        frame = ds[0]
        assert frame["find_virial"] == 0.0
        assert frame["virial"].shape == (9,)
        assert np.allclose(frame["virial"], 0.0)

    def test_add_data_requirement_existing_key(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        req = [
            DataRequirementItem("energy", 1, atomic=False, must=True),
        ]
        ds.add_data_requirement(req)
        frame = ds[0]
        assert frame["find_energy"] == 1.0

    def test_preload_noop(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        ds.preload_and_modify_all_data_torch()  # should not raise

    def test_set_noise_noop(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        ds.set_noise({})  # should not raise


class TestDataLoaderIteration:
    """Test DataLoader iteration with LmdbDataset."""

    def test_batch_iteration(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        from torch.utils.data import (
            DataLoader,
        )

        with torch.device("cpu"):
            dl = DataLoader(
                ds,
                batch_size=2,
                shuffle=False,
                collate_fn=_collate_lmdb_batch,
            )
            batch = next(iter(dl))
        assert "coord" in batch
        assert "sid" in batch
        assert batch["sid"] == 0
        assert batch["coord"].shape == (2, 6, 3)
        assert batch["energy"].shape == (2, 1)
        assert batch["force"].shape == (2, 6, 3)
        assert batch["atype"].shape == (2, 6)
        assert isinstance(batch["fid"], list)
        assert len(batch["fid"]) == 2
        assert isinstance(batch["find_energy"], (float, np.floating))

    def test_inner_dataloader(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        dl = ds.dataloaders[0]
        with torch.device("cpu"):
            batch = next(iter(dl))
        assert "coord" in batch
        assert batch["coord"].shape[0] == 2

    def test_full_epoch(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=3)
        from torch.utils.data import (
            DataLoader,
        )

        with torch.device("cpu"):
            dl = DataLoader(
                ds,
                batch_size=3,
                shuffle=False,
                collate_fn=_collate_lmdb_batch,
            )
            total_frames = 0
            for batch in dl:
                total_frames += batch["coord"].shape[0]
        # 10 frames, batch_size=3 → 3+3+3+1 = 10
        assert total_frames == 10


class TestCollate:
    """Test collate function."""

    def test_collate_basic(self):
        rng = np.random.default_rng(42)
        frames = [
            {
                "coord": rng.standard_normal((4, 3)),
                "energy": np.array([1.0]),
                "find_energy": 1.0,
                "fid": 0,
            },
            {
                "coord": rng.standard_normal((4, 3)),
                "energy": np.array([2.0]),
                "find_energy": 1.0,
                "fid": 1,
            },
        ]
        batch = _collate_lmdb_batch(frames)
        assert batch["coord"].shape == (2, 4, 3)
        assert batch["energy"].shape == (2, 1)
        assert batch["find_energy"] == 1.0
        assert batch["fid"] == [0, 1]
        assert batch["sid"] == 0

    def test_collate_skips_type(self):
        frames = [
            {"coord": np.zeros((2, 3)), "type": np.array([0, 1])},
            {"coord": np.zeros((2, 3)), "type": np.array([0, 1])},
        ]
        batch = _collate_lmdb_batch(frames)
        assert "type" not in batch

    def test_collate_none_values(self):
        frames = [
            {"coord": np.zeros((2, 3)), "box": None},
            {"coord": np.zeros((2, 3)), "box": None},
        ]
        batch = _collate_lmdb_batch(frames)
        assert batch["box"] is None


class TestLmdbTestData:
    """Test LmdbTestData for dp test support."""

    def test_get_test_keys(self, lmdb_dir):
        td = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        td.add("force", 3, atomic=True, must=False, high_prec=False)
        td.add("virial", 9, atomic=False, must=False, high_prec=False)
        result = td.get_test()
        assert "coord" in result
        assert "box" in result
        assert "type" in result
        assert "energy" in result
        assert "force" in result
        assert "find_energy" in result
        assert "find_force" in result
        assert "find_virial" in result

    def test_get_test_shapes(self, lmdb_dir):
        td = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        td.add("force", 3, atomic=True, must=False, high_prec=False)
        td.add("virial", 9, atomic=False, must=False, high_prec=False)
        result = td.get_test()
        nframes = 10
        natoms = 6
        assert result["coord"].shape == (nframes, natoms * 3)
        assert result["box"].shape == (nframes, 9)
        assert result["type"].shape == (nframes, natoms)
        assert result["energy"].shape == (nframes, 1)
        assert result["force"].shape == (nframes, natoms * 3)

    def test_get_test_dtypes(self, lmdb_dir):
        td = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        td.add("force", 3, atomic=True, must=False, high_prec=False)
        result = td.get_test()
        assert result["coord"].dtype == np.float64
        assert result["box"].dtype == np.float64
        assert result["type"].dtype == np.int64
        assert result["energy"].dtype == np.float64
        assert result["force"].dtype == np.float64

    def test_get_test_find_flags(self, lmdb_dir):
        td = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        td.add("force", 3, atomic=True, must=False, high_prec=False)
        td.add("virial", 9, atomic=False, must=False, high_prec=False)
        result = td.get_test()
        assert result["find_energy"] == 1.0
        assert result["find_force"] == 1.0
        assert result["find_virial"] == 0.0  # not in test LMDB data

    def test_get_test_missing_key_default(self, lmdb_dir):
        td = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        td.add("virial", 9, atomic=False, must=False, high_prec=False, default=0.0)
        result = td.get_test()
        assert result["find_virial"] == 0.0
        assert result["virial"].shape == (10, 9)
        assert np.allclose(result["virial"], 0.0)

    def test_get_test_missing_atomic_key(self, lmdb_dir):
        td = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        td.add("atom_ener", 1, atomic=True, must=False, high_prec=False, default=0.0)
        result = td.get_test()
        assert result["find_atom_ener"] == 0.0
        assert result["atom_ener"].shape == (10, 6)  # natoms=6
        assert np.allclose(result["atom_ener"], 0.0)

    def test_pbc(self, lmdb_dir):
        td = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        assert td.pbc is True

    def test_mixed_type(self, lmdb_dir):
        td = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        assert td.mixed_type is True

    def test_shuffle(self, lmdb_dir):
        td1 = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        td2 = LmdbTestData(lmdb_dir, type_map=["O", "H"], shuffle_test=False)
        r1 = td1.get_test()
        r2 = td2.get_test()
        # Without shuffle, results should be identical
        np.testing.assert_array_equal(r1["coord"], r2["coord"])

    def test_type_map_global(self, lmdb_dir):
        """Test with a larger global type_map than LMDB data."""
        td = LmdbTestData(lmdb_dir, type_map=["O", "H", "C"], shuffle_test=False)
        result = td.get_test()
        # type indices should still be 0 and 1
        assert result["type"].max() <= 1


def _create_multi_nloc_lmdb(path: str) -> None:
    """Create an LMDB with frames of varying nloc for distributed tests."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    # 30 frames: 10 with nloc=4, 10 with nloc=6, 10 with nloc=8
    nframes = 30
    frame_nlocs = []
    with env.begin(write=True) as txn:
        idx = 0
        for natoms in [4, 6, 8]:
            for i in range(10):
                key = format(idx, fmt).encode()
                frame = _make_frame(natoms=natoms, seed=idx * 100)
                txn.put(key, msgpack.packb(frame, use_bin_type=True))
                frame_nlocs.append(natoms)
                idx += 1
        metadata = {
            "nframes": nframes,
            "frame_idx_fmt": fmt,
            "frame_nlocs": frame_nlocs,
        }
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
    env.close()


@pytest.fixture
def multi_nloc_lmdb(tmp_path):
    """Create LMDB with multiple nloc groups for distributed tests."""
    lmdb_path = str(tmp_path / "multi_nloc.lmdb")
    _create_multi_nloc_lmdb(lmdb_path)
    return lmdb_path


class TestMixedTypeProperty:
    """Test mixed_type property on LMDB classes."""

    def test_lmdb_data_reader_mixed_type(self, lmdb_dir):
        reader = LmdbDataReader(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert reader.mixed_type is True

    def test_lmdb_dataset_mixed_type(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert ds.mixed_type is True


class TestDistributedSameNlocBatchSampler:
    """Test DistributedSameNlocBatchSampler (pure logic, no torch.distributed)."""

    def test_disjoint_batches(self, multi_nloc_lmdb):
        """Two ranks produce disjoint frame index sets."""
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s0 = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s1 = DistributedSameNlocBatchSampler(
            reader, rank=1, world_size=2, shuffle=True, seed=42
        )
        frames0 = set()
        for batch in s0:
            frames0.update(batch)
        frames1 = set()
        for batch in s1:
            frames1.update(batch)
        # No overlap
        assert frames0 & frames1 == set()

    def test_covers_all_frames(self, multi_nloc_lmdb):
        """Union of all ranks covers all frames."""
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s0 = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s1 = DistributedSameNlocBatchSampler(
            reader, rank=1, world_size=2, shuffle=True, seed=42
        )
        all_frames = set()
        for batch in s0:
            all_frames.update(batch)
        for batch in s1:
            all_frames.update(batch)
        assert all_frames == set(range(30))

    def test_len(self, multi_nloc_lmdb):
        """__len__ returns approximately total // world_size."""
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        single = SameNlocBatchSampler(reader, shuffle=False)
        total = len(single)
        dist_s = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=False, seed=0
        )
        import math

        assert len(dist_s) == math.ceil(total / 2)

    def test_deterministic(self, multi_nloc_lmdb):
        """Same parameters produce same batch sequence."""
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s1 = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s2 = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        batches1 = list(s1)
        batches2 = list(s2)
        assert batches1 == batches2

    def test_set_epoch_changes_order(self, multi_nloc_lmdb):
        """Different epochs produce different batch orderings."""
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s.set_epoch(0)
        batches_e0 = list(s)
        s.set_epoch(1)
        batches_e1 = list(s)
        # Should produce different orderings
        assert batches_e0 != batches_e1

    def test_single_gpu_fallback(self, multi_nloc_lmdb):
        """world_size=1 produces same frames as SameNlocBatchSampler."""
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        single = SameNlocBatchSampler(reader, shuffle=True, seed=42)
        dist_single = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=1, shuffle=True, seed=42
        )
        frames_single = set()
        for batch in single:
            frames_single.update(batch)
        frames_dist = set()
        for batch in dist_single:
            frames_dist.update(batch)
        # Both should cover all frames
        assert frames_single == frames_dist == set(range(30))

    def test_partition_batches_overridable(self, multi_nloc_lmdb):
        """Subclass can override _partition_batches for custom load balancing."""
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)

        class ReversePartition(DistributedSameNlocBatchSampler):
            def _partition_batches(self, all_batches):
                # Take the complementary slice
                return all_batches[
                    self._world_size - 1 - self._rank :: self._world_size
                ]

        s_default = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s_custom = ReversePartition(reader, rank=0, world_size=2, shuffle=True, seed=42)
        # Custom should get rank=1's batches (since it reverses)
        s_rank1 = DistributedSameNlocBatchSampler(
            reader, rank=1, world_size=2, shuffle=True, seed=42
        )
        frames_custom = set()
        for batch in s_custom:
            frames_custom.update(batch)
        frames_rank1 = set()
        for batch in s_rank1:
            frames_rank1.update(batch)
        assert frames_custom == frames_rank1

    def test_same_nloc_per_batch(self, multi_nloc_lmdb):
        """Each batch from distributed sampler has frames with same nloc."""
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        for batch in s:
            nlocs = [reader.frame_nlocs[idx] for idx in batch]
            assert len(set(nlocs)) == 1, f"Mixed nloc in batch: {nlocs}"
