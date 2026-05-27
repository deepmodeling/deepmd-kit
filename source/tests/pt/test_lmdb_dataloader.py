# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for LmdbDataset (PyTorch wrapper) and related PT-specific features.

Pure dpmodel tests (LmdbDataReader, LmdbTestData, SameNlocBatchSampler, type_map
remapping, auto_prob) live in source/tests/common/dpmodel/test_lmdb_data.py.
Consistency tests (dpmodel vs pt) live in source/tests/consistent/test_lmdb_data.py.
"""

import lmdb
import msgpack
import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.lmdb_data import (
    DistributedSameNlocBatchSampler,
    LmdbDataReader,
    SameNlocBatchSampler,
    _decode_frame,
    _read_metadata,
    _remap_keys,
    merge_lmdb,
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
        "cells": _encode_array((np.eye(3) * 10.0).astype(np.float64)),
        "coords": _encode_array((rng.rand(natoms, 3) * 10.0).astype(np.float64)),
        "energies": _encode_array(np.array(rng.randn(), dtype=np.float64)),
        "forces": _encode_array(rng.randn(natoms, 3).astype(np.float64)),
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


# ============================================================
# Internal helper functions
# ============================================================


class TestHelpers:
    """Test internal helper functions (dpmodel, but only tested here)."""

    def test_read_metadata(self, lmdb_dir):
        env = lmdb.open(lmdb_dir, readonly=True, lock=False)
        with env.begin() as txn:
            meta = _read_metadata(txn)
        assert meta["nframes"] == 10
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
        assert "custom_key" in remapped
        assert "coords" not in remapped


# ============================================================
# LmdbDataset (PT wrapper)
# ============================================================


class TestLmdbDataset:
    """Test LmdbDataset class."""

    def test_len(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert len(ds) == 10

    def test_getitem_keys(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        frame = ds[0]
        for key in ("coord", "box", "energy", "force", "atype", "natoms", "fid"):
            assert key in frame
        assert frame["find_energy"] == 1.0
        assert frame["find_force"] == 1.0
        # Metadata keys removed
        for key in ("atom_numbs", "atom_names", "orig"):
            assert key not in frame

    def test_getitem_shapes(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        frame = ds[0]
        assert frame["coord"].shape == (6, 3)
        assert frame["box"].shape == (9,)
        assert frame["energy"].shape == (1,)
        assert frame["force"].shape == (6, 3)
        assert frame["atype"].shape == (6,)
        assert frame["natoms"].shape == (4,)

    def test_getitem_dtypes(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        frame = ds[0]
        assert frame["coord"].dtype == np.float64
        assert frame["atype"].dtype == np.int64

    def test_getitem_out_of_range(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        with pytest.raises(IndexError):
            ds[999]

    def test_natoms_vec(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        natoms = ds[0]["natoms"]
        assert natoms[0] == 6
        assert natoms[2] == 3  # O count
        assert natoms[3] == 3  # H count

    def test_auto_batch_size(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size="auto")
        assert ds.batch_size == 6

    def test_auto_batch_size_with_rule(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size="auto:12")
        assert ds.batch_size == 2

    def test_int_batch_size(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=3)
        assert ds.batch_size == 3

    def test_mixed_type(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        assert ds.mixed_type is True


# ============================================================
# Trainer compatibility interface
# ============================================================


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
        assert ds.index == [5]

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
        req = [DataRequirementItem("virial", 9, atomic=False, must=False, default=0.0)]
        ds.add_data_requirement(req)
        frame = ds[0]
        assert frame["find_virial"] == 0.0
        assert frame["virial"].shape == (9,)

    def test_add_data_requirement_existing_key(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        req = [DataRequirementItem("energy", 1, atomic=False, must=True)]
        ds.add_data_requirement(req)
        assert ds[0]["find_energy"] == 1.0

    def test_preload_noop(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        ds.preload_and_modify_all_data_torch()

    def test_set_noise_noop(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        ds.set_noise({})


# ============================================================
# DataLoader iteration
# ============================================================


class TestDataLoaderIteration:
    """Test DataLoader iteration with LmdbDataset."""

    def test_batch_iteration(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        from torch.utils.data import (
            DataLoader,
        )

        with torch.device("cpu"):
            dl = DataLoader(
                ds, batch_size=2, shuffle=False, collate_fn=_collate_lmdb_batch
            )
            batch = next(iter(dl))
        assert batch["coord"].shape == (2, 6, 3)
        assert batch["energy"].shape == (2, 1)
        assert batch["atype"].shape == (2, 6)
        assert isinstance(batch["fid"], list)
        assert batch["sid"] == 0

    def test_inner_dataloader(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=2)
        with torch.device("cpu"):
            batch = next(iter(ds.dataloaders[0]))
        assert batch["coord"].shape[0] == 2

    def test_full_epoch(self, lmdb_dir):
        ds = LmdbDataset(lmdb_dir, type_map=["O", "H"], batch_size=3)
        from torch.utils.data import (
            DataLoader,
        )

        with torch.device("cpu"):
            dl = DataLoader(
                ds, batch_size=3, shuffle=False, collate_fn=_collate_lmdb_batch
            )
            total_frames = sum(batch["coord"].shape[0] for batch in dl)
        assert total_frames == 10


# ============================================================
# Collate function
# ============================================================


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
        assert batch["fid"] == [0, 1]
        assert batch["sid"] == 0

    def test_collate_skips_type(self):
        frames = [
            {"coord": np.zeros((2, 3)), "type": np.array([0, 1])},
            {"coord": np.zeros((2, 3)), "type": np.array([0, 1])},
        ]
        assert "type" not in _collate_lmdb_batch(frames)

    def test_collate_none_values(self):
        frames = [
            {"coord": np.zeros((2, 3)), "box": None},
            {"coord": np.zeros((2, 3)), "box": None},
        ]
        assert _collate_lmdb_batch(frames)["box"] is None


# ============================================================
# Type map remapping (PT-specific: LmdbDataset)
# ============================================================


def _create_test_lmdb_with_type_map(
    path: str,
    nframes: int = 10,
    natoms: int = 6,
    lmdb_type_map: list[str] | None = None,
) -> None:
    """Create a minimal LMDB dataset with type_map in metadata."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    metadata = {
        "nframes": nframes,
        "frame_idx_fmt": fmt,
        "system_info": {"natoms": [natoms // 2, natoms // 2]},
    }
    if lmdb_type_map is not None:
        metadata["type_map"] = lmdb_type_map
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for i in range(nframes):
            txn.put(
                format(i, fmt).encode(),
                msgpack.packb(_make_frame(natoms=natoms, seed=i), use_bin_type=True),
            )
    env.close()


@pytest.fixture
def lmdb_with_type_map(tmp_path):
    lmdb_path = str(tmp_path / "typed.lmdb")
    _create_test_lmdb_with_type_map(
        lmdb_path, nframes=10, natoms=6, lmdb_type_map=["O", "H"]
    )
    return lmdb_path


class TestTypeMapRemappingDataset:
    """Test type_map remapping in LmdbDataset (PT-specific)."""

    def test_dataset_remap_reversed(self, lmdb_with_type_map):
        ds = LmdbDataset(lmdb_with_type_map, type_map=["H", "O"], batch_size=2)
        frame = ds[0]
        np.testing.assert_array_equal(frame["atype"][:3], [1, 1, 1])
        np.testing.assert_array_equal(frame["atype"][3:], [0, 0, 0])

    def test_dataset_remap_batch(self, lmdb_with_type_map):
        ds = LmdbDataset(lmdb_with_type_map, type_map=["H", "O"], batch_size=2)
        with torch.device("cpu"):
            batch = next(iter(ds.dataloaders[0]))
        for i in range(batch["atype"].shape[0]):
            np.testing.assert_array_equal(batch["atype"][i, :3].numpy(), [1, 1, 1])
            np.testing.assert_array_equal(batch["atype"][i, 3:].numpy(), [0, 0, 0])

    def test_dataset_no_remap_when_match(self, lmdb_with_type_map):
        ds = LmdbDataset(lmdb_with_type_map, type_map=["O", "H"], batch_size=2)
        np.testing.assert_array_equal(ds[0]["atype"][:3], [0, 0, 0])


# ============================================================
# Distributed sampler
# ============================================================


def _create_multi_nloc_lmdb(path: str) -> None:
    """Create an LMDB with frames of varying nloc for distributed tests."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    nframes = 30
    frame_nlocs = []
    with env.begin(write=True) as txn:
        idx = 0
        for natoms in [4, 6, 8]:
            for i in range(10):
                txn.put(
                    format(idx, fmt).encode(),
                    msgpack.packb(
                        _make_frame(natoms=natoms, seed=idx * 100), use_bin_type=True
                    ),
                )
                frame_nlocs.append(natoms)
                idx += 1
        txn.put(
            b"__metadata__",
            msgpack.packb(
                {"nframes": nframes, "frame_idx_fmt": fmt, "frame_nlocs": frame_nlocs},
                use_bin_type=True,
            ),
        )
    env.close()


@pytest.fixture
def multi_nloc_lmdb(tmp_path):
    lmdb_path = str(tmp_path / "multi_nloc.lmdb")
    _create_multi_nloc_lmdb(lmdb_path)
    return lmdb_path


class TestDistributedSameNlocBatchSampler:
    """Test DistributedSameNlocBatchSampler (pure logic, no torch.distributed)."""

    def test_disjoint_batches(self, multi_nloc_lmdb):
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s0 = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s1 = DistributedSameNlocBatchSampler(
            reader, rank=1, world_size=2, shuffle=True, seed=42
        )
        frames0 = {i for batch in s0 for i in batch}
        frames1 = {i for batch in s1 for i in batch}
        assert frames0 & frames1 == set()

    def test_covers_all_frames(self, multi_nloc_lmdb):
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s0 = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s1 = DistributedSameNlocBatchSampler(
            reader, rank=1, world_size=2, shuffle=True, seed=42
        )
        all_frames = {i for batch in s0 for i in batch} | {
            i for batch in s1 for i in batch
        }
        assert all_frames == set(range(30))

    def test_len(self, multi_nloc_lmdb):
        import math

        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        total = len(SameNlocBatchSampler(reader, shuffle=False))
        dist_s = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=False, seed=0
        )
        assert len(dist_s) == math.ceil(total / 2)

    def test_deterministic(self, multi_nloc_lmdb):
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s1 = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s2 = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        assert list(s1) == list(s2)

    def test_set_epoch_changes_order(self, multi_nloc_lmdb):
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        s.set_epoch(0)
        e0 = list(s)
        s.set_epoch(1)
        e1 = list(s)
        assert e0 != e1

    def test_single_gpu_fallback(self, multi_nloc_lmdb):
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        single = {
            i
            for batch in SameNlocBatchSampler(reader, shuffle=True, seed=42)
            for i in batch
        }
        dist = {
            i
            for batch in DistributedSameNlocBatchSampler(
                reader, rank=0, world_size=1, shuffle=True, seed=42
            )
            for i in batch
        }
        assert single == dist == set(range(30))

    def test_same_nloc_per_batch(self, multi_nloc_lmdb):
        reader = LmdbDataReader(multi_nloc_lmdb, type_map=["O", "H"], batch_size=2)
        s = DistributedSameNlocBatchSampler(
            reader, rank=0, world_size=2, shuffle=True, seed=42
        )
        for batch in s:
            nlocs = {reader.frame_nlocs[idx] for idx in batch}
            assert len(nlocs) == 1


# ============================================================
# auto_prob / merge_lmdb (PT-specific: LmdbDataset integration)
# ============================================================


def _create_lmdb_with_system_ids(
    path: str,
    system_frames: list[int],
    natoms: int = 6,
    type_map: list[str] | None = None,
) -> str:
    total = sum(system_frames)
    frame_system_ids = []
    for sid, nf in enumerate(system_frames):
        frame_system_ids.extend([sid] * nf)
    env = lmdb.open(path, map_size=50 * 1024 * 1024)
    fmt = "012d"
    with env.begin(write=True) as txn:
        meta = {
            "nframes": total,
            "frame_idx_fmt": fmt,
            "system_info": {"natoms": [natoms // 2, natoms // 2]},
            "frame_system_ids": frame_system_ids,
            "frame_nlocs": [natoms] * total,
        }
        if type_map is not None:
            meta["type_map"] = type_map
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        for i in range(total):
            txn.put(
                format(i, fmt).encode(),
                msgpack.packb(
                    _make_frame(natoms=natoms, seed=i % 100), use_bin_type=True
                ),
            )
    env.close()
    return path


@pytest.fixture
def auto_prob_lmdb(tmp_path):
    path = str(tmp_path / "auto_prob.lmdb")
    _create_lmdb_with_system_ids(
        path, system_frames=[50, 100, 150], natoms=6, type_map=["O", "H"]
    )
    return path


class TestAutoProbDataset:
    """Test LmdbDataset with auto_prob_style."""

    def test_dataset_auto_prob_passthrough(self, auto_prob_lmdb):
        ds = LmdbDataset(
            auto_prob_lmdb,
            type_map=["O", "H"],
            batch_size=4,
            auto_prob_style="prob_sys_size;0:1:0.5;1:3:0.5",
        )
        assert ds._block_targets is not None

    def test_dataset_auto_prob_none(self, auto_prob_lmdb):
        ds = LmdbDataset(auto_prob_lmdb, type_map=["O", "H"], batch_size=4)
        assert ds._block_targets is None

    def test_dataset_auto_prob_no_system_ids(self, lmdb_dir):
        ds = LmdbDataset(
            lmdb_dir,
            type_map=["O", "H"],
            batch_size=4,
            auto_prob_style="prob_sys_size;0:1:1.0",
        )
        assert ds._block_targets is None

    def test_dataset_auto_prob_iteration(self, auto_prob_lmdb):
        ds = LmdbDataset(
            auto_prob_lmdb,
            type_map=["O", "H"],
            batch_size=4,
            auto_prob_style="prob_sys_size;0:1:0.5;1:3:0.5",
        )
        count = sum(len(batch) for batch in ds._batch_sampler)
        assert count > 300  # expanded


class TestMergeLmdbSystemIds:
    """Test merge_lmdb propagates frame_system_ids."""

    def test_merge_propagates_system_ids(self, tmp_path):
        src1, src2 = str(tmp_path / "src1.lmdb"), str(tmp_path / "src2.lmdb")
        _create_lmdb_with_system_ids(
            src1, system_frames=[5, 10], natoms=6, type_map=["O", "H"]
        )
        _create_lmdb_with_system_ids(
            src2, system_frames=[3, 7], natoms=6, type_map=["O", "H"]
        )
        dst = str(tmp_path / "merged.lmdb")
        merge_lmdb([src1, src2], dst)
        reader = LmdbDataReader(dst, ["O", "H"])
        assert reader.nframes == 25
        assert reader.nsystems == 4
        sids = list(reader.frame_system_ids)
        assert sids[:5] == [0] * 5
        assert sids[5:15] == [1] * 10
        assert sids[15:18] == [2] * 3
        assert sids[18:25] == [3] * 7

    def test_merge_old_lmdb_no_system_ids(self, tmp_path):
        src1, src2 = str(tmp_path / "old1.lmdb"), str(tmp_path / "old2.lmdb")
        _create_test_lmdb(src1, nframes=5, natoms=6)
        _create_test_lmdb(src2, nframes=3, natoms=6)
        dst = str(tmp_path / "merged_old.lmdb")
        merge_lmdb([src1, src2], dst)
        reader = LmdbDataReader(dst, ["O", "H"])
        assert reader.nsystems == 2
        assert list(reader.frame_system_ids[:5]) == [0] * 5
        assert list(reader.frame_system_ids[5:8]) == [1] * 3

    def test_merge_preserves_type_map(self, tmp_path):
        src1, src2 = str(tmp_path / "tm1.lmdb"), str(tmp_path / "tm2.lmdb")
        _create_lmdb_with_system_ids(
            src1, system_frames=[5], natoms=6, type_map=["O", "H"]
        )
        _create_lmdb_with_system_ids(
            src2, system_frames=[5], natoms=6, type_map=["O", "H"]
        )
        dst = str(tmp_path / "merged_tm.lmdb")
        merge_lmdb([src1, src2], dst)
        env = lmdb.open(dst, readonly=True, lock=False)
        with env.begin() as txn:
            meta = _read_metadata(txn)
        env.close()
        assert meta.get("type_map") == ["O", "H"]


# ============================================================
# Multitask LMDB training
# ============================================================


@pytest.fixture
def multitask_lmdb_setup(tmp_path):
    """Create two LMDB datasets and a multitask training config."""
    for name in ("task1_train", "task2_train", "task1_val", "task2_val"):
        nf = 20 if "train" in name else 10
        _create_test_lmdb_with_type_map(
            str(tmp_path / f"{name}.lmdb"),
            nframes=nf,
            natoms=6,
            lmdb_type_map=["O", "H"],
        )

    config = {
        "model": {
            "shared_dict": {
                "type_map_all": ["O", "H"],
                "my_descriptor": {
                    "type": "se_e2_a",
                    "sel": [4, 4],
                    "rcut_smth": 0.5,
                    "rcut": 4.0,
                    "neuron": [4, 8],
                    "axis_neuron": 4,
                    "precision": "float64",
                },
                "my_fitting": {"neuron": [8, 8], "precision": "float64", "seed": 1},
            },
            "model_dict": {
                "model_1": {
                    "type_map": "type_map_all",
                    "descriptor": "my_descriptor",
                    "fitting_net": "my_fitting",
                    "data_stat_nbatch": 1,
                },
                "model_2": {
                    "type_map": "type_map_all",
                    "descriptor": "my_descriptor",
                    "fitting_net": "my_fitting",
                    "data_stat_nbatch": 1,
                },
            },
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 50,
            "start_lr": 1e-3,
            "stop_lr": 1e-8,
        },
        "loss_dict": {
            "model_1": {
                "type": "ener",
                "start_pref_e": 0.2,
                "limit_pref_e": 1,
                "start_pref_f": 100,
                "limit_pref_f": 1,
                "start_pref_v": 0.0,
                "limit_pref_v": 0.0,
            },
            "model_2": {
                "type": "ener",
                "start_pref_e": 0.2,
                "limit_pref_e": 1,
                "start_pref_f": 100,
                "limit_pref_f": 1,
                "start_pref_v": 0.0,
                "limit_pref_v": 0.0,
            },
        },
        "training": {
            "model_prob": {"model_1": 0.5, "model_2": 0.5},
            "data_dict": {
                "model_1": {
                    "stat_file": str(tmp_path / "stat_model_1.hdf5"),
                    "training_data": {
                        "systems": str(tmp_path / "task1_train.lmdb"),
                        "batch_size": 4,
                    },
                    "validation_data": {
                        "systems": str(tmp_path / "task1_val.lmdb"),
                        "batch_size": 2,
                    },
                },
                "model_2": {
                    "stat_file": str(tmp_path / "stat_model_2.hdf5"),
                    "training_data": {
                        "systems": str(tmp_path / "task2_train.lmdb"),
                        "batch_size": 4,
                    },
                    "validation_data": {
                        "systems": str(tmp_path / "task2_val.lmdb"),
                        "batch_size": 2,
                    },
                },
            },
            "numb_steps": 5,
            "seed": 10,
            "disp_file": str(tmp_path / "lcurve.out"),
            "disp_freq": 2,
            "save_freq": 5,
        },
    }
    return config, tmp_path


class TestMultitaskLmdbTraining:
    """Test multitask training with LMDB datasets.

    Uses se_e2_a (not se_atten) to keep memory usage low on CI runners (~7 GB).
    All assertions are in a single test to avoid creating multiple trainers.
    """

    def test_multitask_lmdb_end_to_end(self, multitask_lmdb_setup, monkeypatch):
        from copy import (
            deepcopy,
        )

        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )
        from deepmd.pt.utils.multi_task import (
            preprocess_shared_params,
        )
        from deepmd.utils.argcheck import (
            normalize,
        )
        from deepmd.utils.compat import (
            update_deepmd_input,
        )

        config, tmp_path = multitask_lmdb_setup
        monkeypatch.chdir(tmp_path)
        config = update_deepmd_input(deepcopy(config), warning=True)
        config["model"], shared_links = preprocess_shared_params(config["model"])
        config = normalize(config, multi_task=True)
        trainer = get_trainer(config, shared_links=shared_links)

        # -- trainer init assertions --
        assert trainer.multi_task
        assert set(trainer.model_keys) == {"model_1", "model_2"}

        # -- shared params assertions --
        state_dict = trainer.wrapper.model.state_dict()
        for key in state_dict:
            if "model_1.atomic_model.descriptor" in key:
                key2 = key.replace("model_1", "model_2")
                assert key2 in state_dict
                torch.testing.assert_close(state_dict[key], state_dict[key2])

        # -- get_data assertions --
        for task_key in ["model_1", "model_2"]:
            input_dict, label_dict, log_dict = trainer.get_data(
                is_train=True, task_key=task_key
            )
            assert "coord" in input_dict
            assert "sid" in log_dict

        # -- training run assertions --
        trainer.run()
        assert len(list(tmp_path.glob("model.ckpt*.pt"))) > 0

        # Explicit cleanup to free memory on CI
        import gc

        del trainer
        gc.collect()
