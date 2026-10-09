# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for checkpointable training-data progress (#5814)."""

from __future__ import (
    annotations,
)

import tempfile
import unittest
from pathlib import (
    Path,
)

import lmdb
import msgpack
import numpy as np

from deepmd.dpmodel.train.data_progress import (
    DATA_PROGRESS_VERSION,
    collect_training_data_progress,
    restore_training_data_progress,
)
from deepmd.dpmodel.utils.lmdb_data import (
    LmdbBatchIterator,
    LmdbBatchSampler,
    LmdbDataReader,
)
from deepmd.utils import random as dp_random
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


def _encode_array(arr: np.ndarray) -> dict:
    return {
        "nd": None,
        "type": str(arr.dtype),
        "kind": "",
        "shape": list(arr.shape),
        "data": arr.tobytes(),
    }


def _make_frame(natoms: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    half = natoms // 2
    return {
        "atom_numbs": [half, natoms - half],
        "atom_names": ["O", "H"],
        "atom_types": _encode_array(
            np.array([0] * half + [1] * (natoms - half), dtype=np.int64)
        ),
        "orig": _encode_array(np.zeros(3, dtype=np.float64)),
        "cells": _encode_array((np.eye(3) * 10.0).astype(np.float64)),
        "coords": _encode_array((rng.rand(natoms, 3) * 10.0).astype(np.float64)),
        "energies": _encode_array(np.array(rng.randn(), dtype=np.float64)),
        "forces": _encode_array(rng.randn(natoms, 3).astype(np.float64)),
    }


def _create_test_lmdb(path: str, nframes: int, natoms: int) -> None:
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    fmt = "012d"
    metadata = {
        "nframes": nframes,
        "frame_idx_fmt": fmt,
        "system_info": {
            "formula": f"O{natoms // 2}H{natoms - natoms // 2}",
            "natoms": [natoms // 2, natoms - natoms // 2],
            "nframes": nframes,
        },
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for i in range(nframes):
            key = format(i, fmt).encode()
            txn.put(key, msgpack.packb(_make_frame(natoms, i), use_bin_type=True))
    env.close()


def _create_mixed_nloc_lmdb(
    path: str,
    frames_spec: list[tuple[int, int]] | None = None,
) -> None:
    """Write an LMDB whose frames span several atom counts.

    ``frames_spec`` lists ``(natoms, count)`` pairs. Mixed nloc plus ``mix:N``
    makes per-epoch batch counts shuffle-dependent, which is what exposes the
    single-process logical-cursor wrap bug.
    """
    if frames_spec is None:
        frames_spec = [(6, 8), (9, 5), (12, 3), (4, 6)]
    env = lmdb.open(path, map_size=20 * 1024 * 1024)
    fmt = "012d"
    nframes = sum(count for _, count in frames_spec)
    metadata = {
        "nframes": nframes,
        "frame_idx_fmt": fmt,
        "system_info": {
            "formula": "mixed",
            "natoms": [2, 4],
            "nframes": nframes,
        },
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        idx = 0
        for natoms, count in frames_spec:
            for _ in range(count):
                key = format(idx, fmt).encode()
                txn.put(
                    key,
                    msgpack.packb(_make_frame(natoms, idx), use_bin_type=True),
                )
                idx += 1
    env.close()


def _create_directory_system(root: Path, *, nframes: int = 8) -> Path:
    system = root / "sys0"
    set_dir = system / "set.000"
    set_dir.mkdir(parents=True)
    (system / "type.raw").write_text("0\n1\n0\n1\n0\n1\n")
    (system / "type_map.raw").write_text("O\nH\n")
    natoms = 6
    rng = np.random.RandomState(0)
    np.save(set_dir / "coord.npy", rng.randn(nframes, natoms * 3).astype(np.float64))
    np.save(set_dir / "box.npy", np.tile(np.eye(3).reshape(9), (nframes, 1)))
    np.save(set_dir / "energy.npy", rng.randn(nframes, 1).astype(np.float64))
    return system


class TestLmdbDataProgress(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.lmdb_path = str(Path(self.tmp.name) / "data.lmdb")
        _create_test_lmdb(self.lmdb_path, nframes=12, natoms=6)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _system(self, **kwargs):
        from deepmd.pt_expt.utils.lmdb_dataset import (
            LmdbDataSystem,
        )

        params = {
            "lmdb_path": self.lmdb_path,
            "type_map": ["O", "H"],
            "batch_size": 3,
            "seed": 11,
            "num_workers": 0,
        }
        params.update(kwargs)
        return LmdbDataSystem(**params)

    def test_mid_epoch_restart_matches_uninterrupted_fids(self) -> None:
        reference = self._system()
        try:
            consumed = [reference.get_batch()["fid"] for _ in range(2)]
            progress = reference.state_dict()
            expected = [reference.get_batch()["fid"] for _ in range(3)]
        finally:
            reference.close()

        restored = self._system()
        try:
            restored.load_state_dict(progress)
            actual = [restored.get_batch()["fid"] for _ in range(3)]
        finally:
            restored.close()

        self.assertEqual(progress["epoch"], 0)
        self.assertEqual(progress["batch_index"], 2)
        self.assertEqual(len(consumed), 2)
        self.assertEqual(actual, expected)

    def test_epoch_boundary_preserves_reshuffle(self) -> None:
        reference = self._system()
        try:
            epoch_len = len(reference._sampler)
            for _ in range(epoch_len):
                reference.get_batch()
            progress = reference.state_dict()
            expected = [reference.get_batch()["fid"] for _ in range(2)]
        finally:
            reference.close()

        restored = self._system()
        try:
            restored.load_state_dict(progress)
            actual = [restored.get_batch()["fid"] for _ in range(2)]
        finally:
            restored.close()

        self.assertEqual(progress["epoch"], 1)
        self.assertEqual(progress["batch_index"], 0)
        self.assertEqual(actual, expected)

    def test_prefetch_does_not_advance_logical_cursor(self) -> None:
        reader = LmdbDataReader(self.lmdb_path, ["O", "H"], batch_size=3)
        sampler = LmdbBatchSampler(reader, shuffle=True, seed=5)
        iterator = LmdbBatchIterator(reader, sampler, num_workers=0)
        try:
            batch = next(iterator)
            progress = iterator.state_dict()
            self.assertIsNotNone(iterator._deferred_indices)
            self.assertEqual(progress["batch_index"], 1)
            self.assertEqual(len(batch["fid"]), 3)
        finally:
            iterator.close()
            reader.close()

    def test_ragged_mix_epoch_boundary_cursor_matches_uninterrupted(self) -> None:
        """Resume across a ragged ``mix:N`` wrap must keep the same next batch.

        Prefetch opens the next ``LmdbBatchSampler`` epoch before the logical
        cursor wraps. That sampler advances on ``__iter__``, so a naive
        ``len(sampler)`` at wrap observes epoch+2. Under ragged ``mix:N`` the
        lengths differ across epochs; using the wrong length wraps early and a
        restored run skips the remaining batches of the true epoch.
        """
        mixed_path = str(Path(self.tmp.name) / "mixed.lmdb")
        _create_mixed_nloc_lmdb(mixed_path)
        seed = 0
        budget = "mix:18"

        def _epoch_lengths(count: int = 4) -> list[int]:
            reader = LmdbDataReader(mixed_path, ["O", "H"], batch_size=budget)
            reader.add_data_requirement(
                [
                    DataRequirementItem("force", 3, atomic=True, must=False),
                    DataRequirementItem("energy", 1, atomic=False, must=False),
                ]
            )
            reader.use_ragged_batches(True)
            sampler = LmdbBatchSampler(reader, shuffle=True, seed=seed)
            lengths = []
            for epoch in range(count):
                sampler.set_epoch(epoch)
                lengths.append(len(sampler))
            reader.close()
            return lengths

        lengths = _epoch_lengths()
        self.assertNotEqual(
            lengths[1],
            lengths[2],
            "fixture must vary mix epoch length so the wrap bug is observable",
        )

        def _open_iterator() -> tuple[LmdbDataReader, LmdbBatchIterator]:
            reader = LmdbDataReader(mixed_path, ["O", "H"], batch_size=budget)
            reader.add_data_requirement(
                [
                    DataRequirementItem("force", 3, atomic=True, must=False),
                    DataRequirementItem("energy", 1, atomic=False, must=False),
                ]
            )
            reader.use_ragged_batches(True)
            sampler = LmdbBatchSampler(reader, shuffle=True, seed=seed)
            return reader, LmdbBatchIterator(reader, sampler, num_workers=0)

        reference_reader, reference = _open_iterator()
        try:
            for _ in range(lengths[0]):
                next(reference)
            self.assertEqual(reference.state_dict(), {"epoch": 1, "batch_index": 0})
            self.assertEqual(reference._logical_epoch_length, lengths[1])
            # Mid-epoch-1 checkpoint under the true length, then continue past
            # where a wrong stored length would have wrapped early.
            mid = lengths[1] // 2
            for _ in range(mid):
                next(reference)
            progress = reference.state_dict()
            expected = [next(reference)["fid"] for _ in range(lengths[1] - mid + 1)]
        finally:
            reference.close()
            reference_reader.close()

        self.assertEqual(progress["epoch"], 1)
        self.assertEqual(progress["batch_index"], mid)

        restored_reader, restored = _open_iterator()
        try:
            restored.load_state_dict(progress)
            actual = [next(restored)["fid"] for _ in range(lengths[1] - mid + 1)]
        finally:
            restored.close()
            restored_reader.close()

        self.assertEqual(actual, expected)

    def test_distributed_ranks_restore_disjoint_shards(self) -> None:
        systems = [self._system(rank=rank, world_size=2, seed=19) for rank in range(2)]
        try:
            for system in systems:
                system.get_batch()
            progresses = [system.state_dict() for system in systems]
            expected = [
                [system.get_batch()["fid"] for _ in range(2)] for system in systems
            ]
        finally:
            for system in systems:
                system.close()

        self.assertEqual(progresses[0]["epoch"], progresses[1]["epoch"])
        self.assertEqual(progresses[0]["batch_index"], progresses[1]["batch_index"])

        restored = [self._system(rank=rank, world_size=2, seed=19) for rank in range(2)]
        try:
            for system, progress in zip(restored, progresses, strict=True):
                system.load_state_dict(progress)
            actual = [
                [system.get_batch()["fid"] for _ in range(2)] for system in restored
            ]
        finally:
            for system in restored:
                system.close()

        self.assertEqual(actual, expected)
        self.assertFalse(set(actual[0][0]) & set(actual[1][0]))

    def test_world_size_mismatch_raises(self) -> None:
        system = self._system(world_size=1)
        try:
            progress = system.state_dict()
            progress["world_size"] = 2
            with self.assertRaises(ValueError):
                system.load_state_dict(progress)
        finally:
            system.close()

    def test_multitask_independent_cursors(self) -> None:
        first = self._system(seed=3)
        second = self._system(seed=5)
        try:
            first.get_batch()
            second.get_batch()
            second.get_batch()
            progress = collect_training_data_progress({"water": first, "ice": second})
            expected = {
                "water": first.get_batch()["fid"],
                "ice": second.get_batch()["fid"],
            }
        finally:
            first.close()
            second.close()

        restored_water = self._system(seed=3)
        restored_ice = self._system(seed=5)
        try:
            restore_training_data_progress(
                {"water": restored_water, "ice": restored_ice},
                progress,
            )
            actual = {
                "water": restored_water.get_batch()["fid"],
                "ice": restored_ice.get_batch()["fid"],
            }
        finally:
            restored_water.close()
            restored_ice.close()

        self.assertEqual(actual, expected)
        self.assertEqual(progress["tasks"]["water"]["batch_index"], 1)
        self.assertEqual(progress["tasks"]["ice"]["batch_index"], 2)


class TestDirectoryDataProgress(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.system_path = _create_directory_system(Path(self.tmp.name))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_directory_mid_set_restart_matches_coords(self) -> None:
        dp_random.seed(123)
        reference = DeepmdDataSystem(
            systems=[str(self.system_path)],
            batch_size=2,
            test_size=1,
            type_map=["O", "H"],
            trn_all_set=True,
        )
        consumed = [reference.get_batch()["coord"].copy() for _ in range(2)]
        progress = collect_training_data_progress({"Default": reference})
        expected = [reference.get_batch()["coord"].copy() for _ in range(2)]

        dp_random.seed(999)
        restored = DeepmdDataSystem(
            systems=[str(self.system_path)],
            batch_size=2,
            test_size=1,
            type_map=["O", "H"],
            trn_all_set=True,
        )
        restore_training_data_progress({"Default": restored}, progress)
        actual = [restored.get_batch()["coord"].copy() for _ in range(2)]

        self.assertEqual(len(consumed), 2)
        for left, right in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_missing_progress_keeps_legacy_start(self) -> None:
        dp_random.seed(1)
        data = DeepmdDataSystem(
            systems=[str(self.system_path)],
            batch_size=2,
            test_size=1,
            type_map=["O", "H"],
            trn_all_set=True,
        )
        restore_training_data_progress({"Default": data}, None)
        self.assertEqual(data.data_systems[0].set_count, 0)
        self.assertEqual(data.data_systems[0].iterator, 0)


class TestTaskSelectionRngProgress(unittest.TestCase):
    def test_rng_roundtrip_in_progress_payload(self) -> None:
        dp_random.seed(7)
        _ = dp_random.choice(np.arange(4), p=[0.1, 0.2, 0.3, 0.4])
        progress = collect_training_data_progress({})
        self.assertEqual(progress["version"], DATA_PROGRESS_VERSION)
        expected = [int(dp_random.choice(np.arange(4))) for _ in range(5)]

        dp_random.seed(0)
        restore_training_data_progress({}, progress)
        actual = [int(dp_random.choice(np.arange(4))) for _ in range(5)]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
