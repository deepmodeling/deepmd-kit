# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for LmdbDataReader, LmdbTestData, LmdbBatchSampler, etc.

Pure dpmodel (NumPy/lmdb) tests — no PyTorch dependency.
"""

import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
from concurrent.futures import (
    Future,
)
from itertools import (
    pairwise,
)
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from unittest import (
    mock,
)

import lmdb
import msgpack
import numpy as np

from deepmd.dpmodel.utils import lmdb_data as lmdb_data_module
from deepmd.dpmodel.utils.lmdb_data import (
    DistributedLmdbBatchSampler,
    LmdbBatchIterator,
    LmdbBatchSampler,
    LmdbDataReader,
    LmdbDecodeConfig,
    LmdbTestData,
    LmdbTestDataNlocView,
    _chop_mixed_nloc,
    _expand_indices_by_blocks,
    _merge_lmdb_chunks,
    _remap_atom_types,
    compute_block_targets,
    decode_lmdb_batch,
    decode_lmdb_frame,
    is_lmdb,
    make_neighbor_stat_data,
)
from deepmd.utils import random as dp_random
from deepmd.utils.data import (
    DataRequirementItem,
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


def _create_mix_probe_lmdb(path: str) -> str:
    """Create an LMDB whose frame-level fields collide in shape with the atom axis.

    Frames alternate between 2 and 9 atoms and carry a two-component
    ``fparam`` alongside a nine-component ``virials``. A padding rule that
    guessed the atom axis from a leading dimension would misclassify
    ``fparam`` on the 2-atom frames and ``virials`` on the 9-atom ones, so this
    fixture pins the classification down to the exact ambiguous shapes.
    """
    nlocs = [2, 9, 2, 9, 2, 9]
    rng = np.random.RandomState(7)
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": len(nlocs),
            "frame_idx_fmt": "012d",
            "frame_nlocs": nlocs,
            "system_info": {"natoms": [1, 1], "formula": "probe"},
        }
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        for idx, natoms in enumerate(nlocs):
            frame = _make_frame(natoms=natoms, seed=idx)
            frame["virials"] = {
                "type": "<f8",
                "shape": (9,),
                "data": rng.randn(9).astype(np.float64).tobytes(),
            }
            frame["fparam"] = {
                "type": "<f8",
                "shape": (2,),
                "data": rng.randn(2).astype(np.float64).tobytes(),
            }
            txn.put(
                format(idx, "012d").encode(),
                msgpack.packb(frame, use_bin_type=True),
            )
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


def _create_lmdb_with_virtual_type(path: str) -> str:
    """Create one LMDB frame containing a negative virtual atom type."""
    atype = np.array([0, -1, 1], dtype=np.int64)
    frame = _make_frame(natoms=len(atype))
    frame["atom_types"] = {
        "type": "<i8",
        "shape": atype.shape,
        "data": atype.tobytes(),
    }
    frame["atom_numbs"] = [
        {
            "type": "<i8",
            "shape": (1,),
            "data": np.array([1], dtype=np.int64).tobytes(),
        },
        {
            "type": "<i8",
            "shape": (1,),
            "data": np.array([1], dtype=np.int64).tobytes(),
        },
    ]

    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    with env.begin(write=True) as txn:
        meta = {
            "nframes": 1,
            "frame_idx_fmt": "012d",
            "type_map": ["O", "H"],
            # Virtual atoms occupy coordinate slots but are not real species.
            "system_info": {"natoms": [1, 1]},
            "frame_nlocs": [len(atype)],
        }
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        txn.put(b"000000000000", msgpack.packb(frame, use_bin_type=True))
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

    def test_batch_iterator_advances_epoch_before_prefetch(self):
        """The prefetched successor uses the next epoch's sampler state."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = LmdbBatchSampler(reader, shuffle=True, seed=7)
        iterator = LmdbBatchIterator(reader, sampler, num_workers=2)

        expected_sampler = LmdbBatchSampler(reader, shuffle=True, seed=7)
        expected_sampler.set_epoch(1)
        expected_second = [
            key for indices in expected_sampler for key in reader.original_keys(indices)
        ]

        try:
            first = [key for _ in range(len(sampler)) for key in next(iterator)["fid"]]
            second = [key for _ in range(len(sampler)) for key in next(iterator)["fid"]]
        finally:
            iterator.close()
            reader.close()

        self.assertNotEqual(first, second)
        self.assertEqual(second, expected_second)

    def test_requirements_are_rejected_after_the_first_decode(self):
        """Late requirements are refused so no partition can predate them.

        Batches are grouped by label availability, and both the sampler
        partition and the worker decoders capture the requirements in force
        when they start, so a later registration would silently disagree with
        the batches already produced.
        """
        requirement = [DataRequirementItem("custom", ndof=1, default=0.0)]

        for label, consume in (
            ("frame", lambda reader: reader[0]),
            ("batch", lambda reader: reader.decode_batch([0, 1])),
            ("worker config", lambda reader: reader.worker_decode_config()),
        ):
            with self.subTest(consumed=label):
                reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
                reader.add_data_requirement(requirement)
                consume(reader)
                with self.assertRaisesRegex(RuntimeError, "before reading any frame"):
                    reader.add_data_requirement(requirement)
                reader.close()

    def test_close_preserves_other_reader(self):
        """Closing one shared-path reader leaves the other transaction valid."""
        first = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        second = LmdbDataReader(
            f"{self._lmdb_path}/.",
            self._type_map,
            batch_size=2,
        )
        first.close()
        self.assertTrue(first.closed)
        self.assertEqual(second[0]["coord"].shape, (6, 3))
        second.close()
        self.assertTrue(second.closed)
        with self.assertRaisesRegex(RuntimeError, "closed LMDB reader"):
            _ = second[0]

    def test_batch_dtype_and_field_order_are_chunk_independent(self):
        """Batch promotion and schema matching do not depend on chunking."""
        path = _create_lmdb(
            f"{self._tmpdir.name}/mixed_dtype.lmdb",
            nframes=2,
            natoms=6,
        )
        frame0 = _make_frame(natoms=6, seed=0)
        frame1 = _make_frame(natoms=6, seed=1)
        frame0["custom"] = {
            "type": "float32",
            "shape": [1],
            "data": np.array([1.25], dtype=np.float32).tobytes(),
        }
        frame1["custom"] = {
            "type": "float64",
            "shape": [1],
            "data": np.array([2.5], dtype=np.float64).tobytes(),
        }
        frame1 = dict(reversed(tuple(frame1.items())))
        environment = lmdb.open(path, readonly=False, lock=False)
        with environment.begin(write=True) as transaction:
            transaction.put(
                b"000000000000",
                msgpack.packb(frame0, use_bin_type=True),
            )
            transaction.put(
                b"000000000001",
                msgpack.packb(frame1, use_bin_type=True),
            )
        environment.close()

        config = LmdbDecodeConfig(
            ntypes=2,
            natoms=6,
            type_remap=None,
            data_requirements={},
        )
        environment = lmdb.open(path, readonly=True, lock=False)
        with environment.begin() as transaction:
            serial = decode_lmdb_batch(
                transaction,
                [0, 1],
                "012d",
                config,
            )
            chunked = _merge_lmdb_chunks(
                [
                    decode_lmdb_batch(transaction, [0], "012d", config),
                    decode_lmdb_batch(transaction, [1], "012d", config),
                ]
            )
        environment.close()

        self.assertEqual(serial["custom"].dtype, np.float64)
        np.testing.assert_array_equal(serial["custom"], chunked["custom"])

    def test_parallel_batch_consistency_guards(self):
        """Parallel batch decoding validates schemas, shapes, and availability."""
        config = LmdbDecodeConfig(
            ntypes=2,
            natoms=6,
            type_remap=None,
            data_requirements={},
        )
        transaction = mock.Mock()
        transaction.get.return_value = b"frame"
        first_frame = {
            "coord": np.zeros((6, 3)),
            "find_energy": np.float32(1.0),
            "fid": 0,
        }
        frame_cases = (
            (
                "fields",
                {**first_frame, "energy": np.zeros(1), "fid": 1},
                "inconsistent fields",
            ),
            (
                "atom count",
                {**first_frame, "coord": np.zeros((7, 3)), "fid": 1},
                "the batch layout gives frame 1 6 atoms, but it holds 7",
            ),
            (
                "shape",
                {**first_frame, "coord": np.zeros((6, 4)), "fid": 1},
                "changes shape within one batch",
            ),
        )
        for guard, second_frame, expected_error in frame_cases:
            with (
                self.subTest(guard=f"frame {guard}"),
                mock.patch.object(
                    lmdb_data_module,
                    "decode_lmdb_frame",
                    side_effect=(first_frame, second_frame),
                ),
                self.assertRaisesRegex(ValueError, expected_error),
            ):
                decode_lmdb_batch(transaction, [0, 1], "012d", config)

        first_chunk = {
            "find_energy": np.float32(1.0),
            "coord": np.zeros((1, 6, 3)),
            "fid": [0],
            "sid": np.array([0], dtype=np.int64),
        }
        chunk_cases = (
            (
                "fields",
                {key: value for key, value in first_chunk.items() if key != "coord"},
                "inconsistent fields",
            ),
            (
                "availability",
                {**first_chunk, "find_energy": np.float32(0.0), "fid": [1]},
                "availability changes across worker chunks",
            ),
        )
        for guard, second_chunk, expected_error in chunk_cases:
            with (
                self.subTest(guard=f"chunk {guard}"),
                self.assertRaisesRegex(ValueError, expected_error),
            ):
                _merge_lmdb_chunks([first_chunk, second_chunk])

    def test_batch_rejects_mixed_label_availability(self):
        """A scalar find flag cannot represent mixed availability in one batch."""
        path = _create_lmdb(
            f"{self._tmpdir.name}/mixed_availability.lmdb",
            nframes=2,
            natoms=6,
        )
        frame = _make_frame(natoms=6, seed=0)
        frame["custom"] = {
            "type": "float64",
            "shape": [1],
            "data": np.array([1.0], dtype=np.float64).tobytes(),
        }
        environment = lmdb.open(path, readonly=False, lock=False)
        with environment.begin(write=True) as transaction:
            transaction.put(
                b"000000000000",
                msgpack.packb(frame, use_bin_type=True),
            )
        environment.close()

        requirement = DataRequirementItem("custom", ndof=1, default=0.0)
        config = LmdbDecodeConfig(
            ntypes=2,
            natoms=6,
            type_remap=None,
            data_requirements={"custom": requirement},
        )
        environment = lmdb.open(path, readonly=True, lock=False)
        with environment.begin() as transaction:
            with self.assertRaisesRegex(
                ValueError,
                "availability changes within one batch",
            ):
                decode_lmdb_batch(
                    transaction,
                    [0, 1],
                    "012d",
                    config,
                )
        environment.close()

    def test_is_lmdb(self):
        self.assertTrue(is_lmdb(self._lmdb_path))
        self.assertTrue(is_lmdb("something.lmdb"))
        self.assertFalse(is_lmdb("/some/npy/system"))

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

    def test_min_pair_dist_requirement_computed(self):
        path = _create_grid_lmdb(f"{self._tmpdir.name}/grid_min_pair.lmdb", nframes=1)
        reader = LmdbDataReader(path, ["TYPE"], batch_size=1)
        reader.add_data_requirement(
            [
                DataRequirementItem(
                    "min_pair_dist",
                    ndof=1,
                    atomic=False,
                    must=False,
                    high_prec=False,
                )
            ]
        )

        frame = reader[0]

        self.assertEqual(frame["find_min_pair_dist"], np.float32(1.0))
        np.testing.assert_allclose(frame["min_pair_dist"], np.array([1.0]))

    def test_min_pair_dist_requirement_defaults_without_atype(self):
        raw_frame = _make_frame(natoms=6, seed=0)
        raw_frame.pop("atom_types")
        requirement = DataRequirementItem(
            "min_pair_dist",
            ndof=1,
            default=0.25,
        )
        config = LmdbDecodeConfig(
            ntypes=2,
            natoms=6,
            type_remap=None,
            data_requirements={"min_pair_dist": requirement},
        )

        frame = decode_lmdb_frame(
            msgpack.packb(raw_frame, use_bin_type=True),
            0,
            config,
            copy_arrays=True,
        )

        self.assertEqual(frame["find_min_pair_dist"], np.float32(0.0))
        np.testing.assert_allclose(frame["min_pair_dist"], np.array([0.25]))


# ============================================================
# Mixed nloc tests
# ============================================================


class TestMixedNloc(unittest.TestCase):
    """Tests for mixed-nloc datasets and LmdbBatchSampler."""

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
        sampler = LmdbBatchSampler(reader, shuffle=False, seed=42)
        for batch_indices in sampler:
            nlocs = [reader.frame_nlocs[i] for i in batch_indices]
            self.assertTrue(all(n == nlocs[0] for n in nlocs))

    def test_sampler_covers_all_frames(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = LmdbBatchSampler(reader, shuffle=False, seed=42)
        all_indices = [i for batch in sampler for i in batch]
        self.assertEqual(sorted(all_indices), list(range(10)))

    def test_sampler_auto_batch_size_per_nloc(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="auto")
        bs_6 = reader.get_batch_size_for_nloc(6)
        bs_12 = reader.get_batch_size_for_nloc(12)
        self.assertGreaterEqual(bs_6, bs_12)

    def test_sampler_shuffle_deterministic(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        s1 = LmdbBatchSampler(reader, shuffle=True, seed=123)
        s2 = LmdbBatchSampler(reader, shuffle=True, seed=123)
        # A seed reproduces the whole sequence of passes, not just the first.
        self.assertEqual(list(s1), list(s2))
        self.assertEqual(list(s1), list(s2))

    def test_sampler_reshuffles_between_passes(self):
        """Every pass draws a fresh shuffle while still covering each frame once.

        Training re-iterates the sampler whenever it exhausts it, so a pass
        that repeated the previous one would show the model the same batches
        for the whole run.
        """
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = LmdbBatchSampler(reader, shuffle=True, seed=5)
        first = [list(batch) for batch in sampler]
        second = [list(batch) for batch in sampler]
        self.assertNotEqual(first, second)
        self.assertEqual(
            sorted(index for batch in first for index in batch),
            sorted(index for batch in second for index in batch),
        )

    def test_sampler_len(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=2)
        sampler = LmdbBatchSampler(reader, shuffle=False)
        self.assertEqual(len(sampler), len(list(sampler)))

    def test_mix_batches_span_several_nlocs(self):
        """``mix:N`` puts frames of different atom counts in one batch.

        Frames are sorted by atom count before batches are filled, so a batch
        spans several counts exactly where a budget boundary falls inside the
        sorted run. The budget here is chosen to land on such a boundary.
        """
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:18")
        self.assertTrue(reader.mixed_nloc)
        batches = list(LmdbBatchSampler(reader, shuffle=False))
        spanned = max(len({reader.frame_nlocs[i] for i in batch}) for batch in batches)
        self.assertGreater(spanned, 1)

    def test_mix_merges_small_neighbouring_atom_count_groups(self):
        """Nearby atom counts share a batch instead of one tiny batch each.

        This is what ``mix:N`` buys over ``max:N``: an atom-count group too
        small to fill a batch on its own no longer forces an under-filled
        step. The gain needs the counts to be close, since padding costs
        ``nframes * max_nloc``; a lone outlier size still batches alone.
        """
        path = f"{self._tmpdir.name}/graded.lmdb"
        _create_mixed_sid_nloc_lmdb(
            path,
            system_specs=[(2, 6), (2, 7), (2, 8)],
            type_map=self._type_map,
        )
        costs = {}
        for spec in ("max:72", "mix:72"):
            reader = LmdbDataReader(path, self._type_map, batch_size=spec)
            costs[spec] = [
                len(b) * reader.batch_pad_nloc(b)
                for b in LmdbBatchSampler(reader, shuffle=False)
            ]
        # One under-filled batch per atom-count group, against a single full one.
        self.assertEqual(len(costs["max:72"]), 3)
        self.assertEqual(costs["mix:72"], [6 * 8])

    def test_mix_respects_the_atom_budget(self):
        """A batch's padded cost stays within the budget above one frame."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:36")
        for batch in LmdbBatchSampler(reader, shuffle=True, seed=3):
            cost = len(batch) * reader.batch_pad_nloc(batch)
            self.assertTrue(len(batch) == 1 or cost <= 36, (len(batch), cost))

    def test_mix_covers_every_frame_once(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:36")
        indices = [
            i for batch in LmdbBatchSampler(reader, shuffle=True, seed=5) for i in batch
        ]
        self.assertEqual(sorted(indices), list(range(len(reader))))

    def test_mix_len_matches_iteration(self):
        """The batch count depends on the shuffle, so ``len`` must reuse it."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:36")
        sampler = LmdbBatchSampler(reader, shuffle=True, seed=9)
        self.assertEqual(len(sampler), len(list(sampler)))

    def test_mix_shuffle_deterministic(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:36")
        s1 = LmdbBatchSampler(reader, shuffle=True, seed=17)
        s2 = LmdbBatchSampler(reader, shuffle=True, seed=17)
        self.assertEqual(list(s1), list(s2))

    def test_mix_pads_the_atom_axis_only(self):
        """Padding widens per-atom fields and leaves frame-level ones alone."""
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:36")
        reader.add_data_requirement(
            [
                DataRequirementItem("energy", 1, atomic=False, must=False),
                DataRequirementItem("force", 3, atomic=True, must=False),
            ]
        )
        # Frames 0 and 8 hold 6 and 12 atoms respectively.
        batch = reader.decode_batch([0, 8])
        self.assertEqual(batch["coord"].shape, (2, 12, 3))
        self.assertEqual(batch["atype"].shape, (2, 12))
        self.assertEqual(batch["force"].shape, (2, 12, 3))
        self.assertEqual(batch["box"].shape, (2, 9))
        self.assertEqual(batch["energy"].shape, (2, 1))

        # The short frame keeps its six atoms; the tail is inert.
        np.testing.assert_array_equal(batch["atype"][0, 6:], -1)
        self.assertTrue(np.all(batch["atype"][0, :6] >= 0))
        np.testing.assert_array_equal(batch["coord"][0, 6:], 0.0)
        np.testing.assert_array_equal(batch["force"][0, 6:], 0.0)
        np.testing.assert_array_equal(batch["atype"][1], reader[8]["atype"])

        # Atom counts stay the real per-frame values, not the padded width.
        self.assertEqual(batch["natoms"][0, 0], 6)
        self.assertEqual(batch["natoms"][1, 0], 12)

    def test_mix_pads_repeated_per_atom_fields(self):
        """A ``repeat != 1`` requirement is stored flat but still padded by atom.

        ``atom_pref`` spends ``repeat`` leading entries per atom instead of
        one, so its padded width is ``pad_nloc * repeat``.
        """
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:36")
        reader.add_data_requirement(
            [DataRequirementItem("atom_pref", 1, atomic=True, must=False, repeat=3)]
        )
        # Frames 0 and 8 hold 6 and 12 atoms respectively.
        batch = reader.decode_batch([0, 8])
        self.assertEqual(batch["atom_pref"].shape, (2, 12 * 3))
        np.testing.assert_array_equal(batch["atom_pref"][0, 6 * 3 :], 0.0)

    def test_mix_chunked_decode_matches_in_process_decode(self):
        """Splitting a padded batch across workers reproduces it exactly.

        Each chunk pads and lays out its fields independently, so a mixed-nloc
        batch is the case where the chunks can disagree: they start on frames
        of different atom counts.
        """
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:36")
        reader.add_data_requirement(
            [
                DataRequirementItem("force", 3, atomic=True, must=False),
                DataRequirementItem("atom_pref", 1, atomic=True, must=False, repeat=3),
            ]
        )
        # Frames 0 and 8 hold 6 and 12 atoms; one per chunk puts the wide
        # frame second, so a per-chunk layout would size the chunks differently.
        indices = [0, 8]
        layout = reader.batch_layout(indices)
        chunks = [
            lmdb_data_module.decode_lmdb_batch(
                reader._transaction(),
                [key],
                reader.frame_format,
                reader._decode_config,
                layout.chunk(position, position + 1),
            )
            for position, key in enumerate(reader.original_keys(indices))
        ]
        merged = lmdb_data_module._merge_lmdb_chunks(chunks)
        expected = reader.decode_batch(indices)
        self.assertEqual(sorted(merged), sorted(expected))
        for field, value in expected.items():
            np.testing.assert_array_equal(merged[field], value, err_msg=field)

    def test_mix_keeps_frame_fields_of_ambiguous_shape(self):
        """``fparam`` and ``virial`` are never padded, whatever the atom count."""
        path = _create_mix_probe_lmdb(f"{self._tmpdir.name}/probe.lmdb")
        reader = LmdbDataReader(path, self._type_map, batch_size="mix:36")
        reader.add_data_requirement(
            [
                DataRequirementItem("virial", 9, atomic=False, must=False),
                DataRequirementItem("fparam", 2, atomic=False, must=False),
                DataRequirementItem("force", 3, atomic=True, must=False),
            ]
        )
        # A 2-atom frame makes fparam (2,) ambiguous; a 9-atom one does the
        # same for virial (9,). Batching them together exercises both.
        batch = reader.decode_batch([0, 1])
        self.assertEqual(batch["fparam"].shape, (2, 2))
        self.assertEqual(batch["virial"].shape, (2, 9))
        self.assertEqual(batch["coord"].shape, (2, 9, 3))
        np.testing.assert_array_equal(batch["fparam"][0], reader[0]["fparam"])
        np.testing.assert_array_equal(batch["virial"][0], reader[0]["virial"])

    def test_mix_tracks_the_atom_proportional_weighting(self):
        """Packing to a budget must not distort how much each frame counts.

        A batch is one optimizer step, and its frame-level terms (energy,
        virial) average over the frames present, so a frame's weight in them
        is ``1 / k_b``. An atom budget asks for ``k_b ~ budget / nloc``, that
        is a weight proportional to atom count, matching what the pooled
        per-atom terms give those frames unconditionally.
        Same-nloc batching misses that ideal wherever an atom-count group is
        too small to fill a batch: its few frames form a short batch and each
        of them is over-weighted many times over. Filling batches across atom
        counts removes that failure mode, and the residual error is the
        padding, so both goals improve together.
        """
        path = f"{self._tmpdir.name}/skewed.lmdb"
        # A few dominant atom counts plus a tail of groups too small to fill a
        # batch, down to one holding a single frame. This is the shape of a
        # real merged dataset in miniature, and the tail is where same-nloc
        # batching goes wrong.
        _create_mixed_sid_nloc_lmdb(
            path,
            system_specs=[
                (120, 6),
                (80, 8),
                (60, 10),
                (3, 12),
                (2, 14),
                (2, 16),
                (1, 20),
            ],
            type_map=self._type_map,
        )
        budget = 120

        def weighting_error(spec):
            """Log-spread of the frame weight against the ideal, and its worst case."""
            reader = LmdbDataReader(path, self._type_map, batch_size=spec)
            nlocs = np.asarray(reader.frame_nlocs, dtype=np.float64)
            batches = LmdbBatchSampler(reader, shuffle=True, seed=0).batches()
            weight = np.empty(len(nlocs))
            for batch in batches:
                weight[batch] = 1.0 / len(batch)
            slots = sum(len(b) * reader.batch_pad_nloc(b) for b in batches)
            # Weights matter only up to a global scale, which is the learning
            # rate, so normalize both sides to mean 1 before comparing.
            ratio = (weight / weight.mean()) / (nlocs / nlocs.mean())
            return np.std(np.log(ratio)), ratio.max(), nlocs.sum() / slots

        same_spread, same_worst, same_efficiency = weighting_error(f"max:{budget}")
        mix_spread, mix_worst, mix_efficiency = weighting_error(f"mix:{budget}")

        self.assertEqual(same_efficiency, 1.0)
        self.assertGreater(mix_efficiency, 0.95)
        # Both the typical and the worst-case deviation shrink by a wide
        # margin; the thresholds leave room for the packing to change.
        self.assertLess(mix_spread, 0.5 * same_spread)
        self.assertLess(mix_worst, 0.5 * same_worst)

    def test_mix_cuts_a_batch_only_when_the_next_frame_does_not_fit(self):
        """No batch is closed early, which is what keeps batches full.

        Sorting a sliding window instead of the whole group would break this:
        each window boundary closes a batch regardless of how full it is, and
        an under-filled batch over-weights every frame it holds.
        """
        budget = 36
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:36")
        # Chop the group directly, which keeps the batches in the atom-count
        # order the greedy produced them in.
        batches = _chop_mixed_nloc(reader, list(range(len(reader))))
        for batch, following in pairwise(batches):
            next_nloc = min(reader.frame_nlocs[index] for index in following)
            self.assertGreater((len(batch) + 1) * next_nloc, budget)

    def _ragged_reader(self, batch_size="mix:36"):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size=batch_size)
        reader.add_data_requirement(
            [
                DataRequirementItem("force", 3, atomic=True, must=False),
                DataRequirementItem("energy", 1, atomic=False, must=False),
            ]
        )
        reader.use_ragged_batches(True)
        return reader

    def test_ragged_layout_concatenates_the_frames(self):
        """The ragged layout carries the same rows on a flat, unpadded axis."""
        rectangular = LmdbDataReader(
            self._lmdb_path, self._type_map, batch_size="mix:36"
        )
        rectangular.add_data_requirement(
            [
                DataRequirementItem("force", 3, atomic=True, must=False),
                DataRequirementItem("energy", 1, atomic=False, must=False),
            ]
        )
        ragged = self._ragged_reader()

        # Frames 0 and 8 hold 6 and 12 atoms respectively.
        indices = [0, 8]
        padded = rectangular.decode_batch(indices)
        flat = ragged.decode_batch(indices)

        self.assertEqual(padded["coord"].shape, (2, 12, 3))
        self.assertEqual(flat["coord"].shape, (18, 3))
        self.assertEqual(flat["atype"].shape, (18,))
        self.assertEqual(flat["force"].shape, (18, 3))
        np.testing.assert_array_equal(flat["n_node"], [6, 12])
        self.assertTrue((flat["atype"] >= 0).all(), "a ragged batch pads nothing")
        # Frame-level fields keep their frame axis under either layout.
        self.assertEqual(flat["energy"].shape, padded["energy"].shape)
        self.assertEqual(flat["box"].shape, padded["box"].shape)

        offset = 0
        for row, count in enumerate(flat["n_node"].tolist()):
            for field in ("coord", "atype", "force"):
                np.testing.assert_array_equal(
                    padded[field][row, :count],
                    flat[field][offset : offset + count],
                    err_msg=field,
                )
            offset += count

    def test_ragged_chunked_decode_matches_in_process_decode(self):
        """Chunks concatenate on the flat axis, so a split decode is the same."""
        reader = self._ragged_reader()
        indices = [0, 8]
        layout = reader.batch_layout(indices)
        chunks = [
            lmdb_data_module.decode_lmdb_batch(
                reader._transaction(),
                [key],
                reader.frame_format,
                reader._decode_config,
                layout.chunk(position, position + 1),
            )
            for position, key in enumerate(reader.original_keys(indices))
        ]
        merged = lmdb_data_module._merge_lmdb_chunks(chunks)
        expected = reader.decode_batch(indices)
        self.assertEqual(sorted(merged), sorted(expected))
        for field, value in expected.items():
            np.testing.assert_array_equal(merged[field], value, err_msg=field)

    def test_ragged_packing_fills_on_real_atoms_and_keeps_the_caller_order(self):
        """Without padding the budget counts real atoms, and sorting is dropped.

        Sorting exists only to keep a rectangular batch's padded width close to
        its frames. A ragged batch has no width, so sorting would merely make
        each optimizer step homogeneous in system size.
        """
        budget = 36
        ragged = self._ragged_reader(batch_size=f"mix:{budget}")
        nlocs = np.asarray(ragged.frame_nlocs)
        # The fixture stores its frames in ascending atom count, so hand them
        # over reversed: only then does sorting leave a visible trace.
        group = list(reversed(range(len(ragged))))
        batches = _chop_mixed_nloc(ragged, group)

        for batch in batches:
            if len(batch) > 1:
                self.assertLessEqual(int(nlocs[batch].sum()), budget)
        self.assertEqual(sorted(index for b in batches for index in b), sorted(group))
        # The frames arrive in the order they were handed over, not sorted.
        self.assertEqual([index for batch in batches for index in batch], group)

        rectangular = LmdbDataReader(
            self._lmdb_path, self._type_map, batch_size=f"mix:{budget}"
        )
        padded = _chop_mixed_nloc(rectangular, group)
        self.assertEqual(
            [int(nlocs[index]) for batch in padded for index in batch],
            sorted(nlocs[group].tolist()),
            "the rectangular layout must still sort by atom count",
        )

    def test_mix_uses_the_fewest_batches_possible(self):
        """The packing attains the minimum batch count, not merely a good one.

        The minimum is computed by dynamic programming over the sorted atom
        counts. Its correctness rests on being free to take every batch
        contiguous in that order: the batch holding the widest frame may be
        given the widest frames outright, since its capacity ``budget // nloc``
        does not depend on which frames fill it. The recurrence then closes
        each batch at its widest frame,

            steps[i] = 1 + steps[max(0, i - budget // nloc[i - 1])],

        whereas the implementation opens each batch at its narrowest. The two
        must agree.
        """
        for budget in (18, 36, 72):
            with self.subTest(budget=budget):
                reader = LmdbDataReader(
                    self._lmdb_path, self._type_map, batch_size=f"mix:{budget}"
                )
                nlocs = sorted(reader.frame_nlocs)
                steps = [0] * (len(nlocs) + 1)
                for i in range(1, len(nlocs) + 1):
                    capacity = max(1, budget // nlocs[i - 1])
                    steps[i] = 1 + steps[max(0, i - capacity)]
                batches = _chop_mixed_nloc(reader, list(range(len(reader))))
                self.assertEqual(len(batches), steps[len(nlocs)])

    def test_mix_distributed_partition_is_disjoint_and_complete(self):
        reader = LmdbDataReader(self._lmdb_path, self._type_map, batch_size="mix:24")
        ranks = [
            DistributedLmdbBatchSampler(
                reader, rank=rank, world_size=2, shuffle=True, seed=4
            )
            for rank in (0, 1)
        ]
        batches = [list(sampler) for sampler in ranks]
        for sampler, own in zip(ranks, batches, strict=True):
            self.assertEqual(len(sampler), len(own))
        indices = [i for own in batches for batch in own for i in batch]
        self.assertEqual(sorted(indices), list(range(len(reader))))

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

    def test_test_data_nloc_view(self):
        """LmdbTestDataNlocView delegates attributes and fixes nloc."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        td.add("energy", 1, atomic=False, must=False, high_prec=True)
        view = LmdbTestDataNlocView(td, 9)

        self.assertEqual(view.pbc, td.pbc)
        self.assertIs(view.nloc_groups, td.nloc_groups)

        expected = td.get_test(nloc=9)
        actual = view.get_test()
        self.assertEqual(actual["coord"].shape, (4, 9 * 3))
        self.assertEqual(actual["type"].shape, (4, 9))
        self.assertEqual(actual.keys(), expected.keys())
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if isinstance(expected_value, np.ndarray):
                np.testing.assert_array_equal(actual_value, expected_value)
            else:
                self.assertEqual(actual_value, expected_value)

    def test_test_data_nloc_view_iterates_selected_frames(self):
        """A label-availability view streams only its selected frame indices."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        selected = td.nloc_groups[9][::2]
        view = LmdbTestDataNlocView(td, 9, frame_indices=selected)

        chunks = list(view.iter_test(chunk_atoms=9))

        self.assertEqual(len(chunks), len(selected))
        expected = td.get_test_by_indices(selected)
        np.testing.assert_array_equal(
            np.concatenate([chunk["coord"] for chunk in chunks]),
            expected["coord"],
        )

    def test_test_data_shuffle_uses_global_seed(self):
        """The CLI random seed makes LMDB test-frame shuffling reproducible."""
        self.addCleanup(dp_random.seed, None)
        dp_random.seed(123)
        first = LmdbTestData(
            self._lmdb_path,
            type_map=self._type_map,
            shuffle_test=True,
        )
        dp_random.seed(123)
        second = LmdbTestData(
            self._lmdb_path,
            type_map=self._type_map,
            shuffle_test=True,
        )

        self.assertEqual(first.nloc_groups, second.nloc_groups)

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

    def test_virtual_type_preserved_during_remap(self):
        """Both LMDB consumers must retain virtual sentinels when remapping."""
        path = _create_lmdb_with_virtual_type(f"{self._tmpdir.name}/virtual_type.lmdb")

        reader = LmdbDataReader(path, ["H", "O"])
        frame = reader[0]
        np.testing.assert_array_equal(frame["atype"], [1, -1, 0])
        np.testing.assert_array_equal(frame["natoms"], [3, 3, 1, 1])

        test_data = LmdbTestData(path, type_map=["H", "O"], shuffle_test=False)
        np.testing.assert_array_equal(test_data.get_test()["type"], [[1, -1, 0]])

    def test_positive_out_of_range_type_still_raises(self):
        """A malformed real type must not be mistaken for a virtual sentinel."""
        with self.assertRaises(IndexError):
            _remap_atom_types(np.array([0, 2]), np.array([1, 0]))


# ============================================================
# auto_prob / frame_system_ids tests
# ============================================================


class TestAutoProb(unittest.TestCase):
    """Test auto_prob support: frame_system_ids, compute_block_targets,
    _expand_indices_by_blocks, and LmdbBatchSampler with block_targets.
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

    def test_compute_block_targets_logs_dropped_block(self):
        """Emptied blocks trigger an INFO log.

        The silent re-normalisation of ``auto_prob_style`` (remaining
        weights rescaled to sum to 1.0) must be visible to operators
        alongside the ``filter:N`` drop line.
        """
        with self.assertLogs("deepmd.dpmodel.utils.lmdb_data", level="INFO") as cm:
            result = compute_block_targets(
                "prob_sys_size;0:1:0.8;1:2:0.2",
                nsystems=2,
                system_nframes=[0, 500],
            )
        self.assertTrue(any("empty blocks" in msg for msg in cm.output))
        self.assertEqual(result, [])

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
        sampler = LmdbBatchSampler(reader, shuffle=True, block_targets=block_targets)
        all_indices = [i for batch in sampler for i in batch]
        self.assertGreater(len(all_indices), 600)
        self.assertEqual(len(set(all_indices)), 600)

    def test_sampler_allocates_block_target_across_find_signatures(self):
        """Independent signature groups must not each round the block target."""

        class TwoSignatureReader:
            """Minimal reader exposing two one-frame availability groups."""

            mixed_nloc = False

            def __init__(self):
                self.nloc_groups = {6: [0, 1]}
                self.frame_system_ids = [0, 0]
                self.frame_nlocs = [6, 6]

            @staticmethod
            def group_indices_by_find_signature(indices):
                self.assertEqual(indices, [0, 1])
                return {(0.0,): [0], (1.0,): [1]}

            @staticmethod
            def get_batch_size_for_nloc(nloc):
                self.assertEqual(nloc, 6)
                return 1

        sampler = LmdbBatchSampler(
            TwoSignatureReader(),
            shuffle=False,
            block_targets=[([0], 3)],
        )
        batches = list(sampler)
        counts = [sum(index in batch for batch in batches) for index in (0, 1)]

        self.assertEqual(len(sampler), len(batches))
        self.assertEqual(sum(map(len, batches)), 3)
        self.assertEqual(sorted(counts), [1, 2])

    def test_sampler_without_block_targets(self):
        reader = LmdbDataReader(self._lmdb_path, ["O", "H"])
        sampler = LmdbBatchSampler(reader, shuffle=False)
        all_indices = [i for batch in sampler for i in batch]
        self.assertEqual(sorted(all_indices), list(range(600)))


# ============================================================
# batch_size = "max:N" / "filter:N" tests
# ============================================================


def _create_mixed_sid_nloc_lmdb(
    path: str,
    system_specs: list[tuple[int, int]],
    type_map: list[str] | None = None,
) -> str:
    """Build an LMDB whose systems have *different* nloc per sid.

    Existing helpers either fix nloc globally or fix sid boundaries; the
    filter:N behaviour we want to exercise depends on nloc varying across
    systems so this helper glues both axes together in one tiny LMDB.

    Parameters
    ----------
    path
        Output LMDB directory.
    system_specs
        ``[(nframes, natoms), ...]`` for each system (sid = list index).
    type_map
        Optional element list stored in metadata.
    """
    total = sum(nf for nf, _ in system_specs)
    frame_system_ids: list[int] = []
    frame_nlocs: list[int] = []
    for sid, (nf, natoms) in enumerate(system_specs):
        frame_system_ids.extend([sid] * nf)
        frame_nlocs.extend([natoms] * nf)

    env = lmdb.open(path, map_size=100 * 1024 * 1024)
    with env.begin(write=True) as txn:
        first_natoms = system_specs[0][1]
        n0 = max(1, first_natoms // 3)
        n1 = first_natoms - n0
        meta = {
            "nframes": total,
            "frame_idx_fmt": "012d",
            "system_info": {"natoms": [n0, n1]},
            "frame_system_ids": frame_system_ids,
            "frame_nlocs": frame_nlocs,
        }
        if type_map is not None:
            meta["type_map"] = type_map
        txn.put(b"__metadata__", msgpack.packb(meta, use_bin_type=True))
        idx = 0
        for _sid, (nf, natoms) in enumerate(system_specs):
            for _ in range(nf):
                frame = _make_frame(natoms=natoms, seed=idx % 100)
                txn.put(
                    format(idx, "012d").encode(),
                    msgpack.packb(frame, use_bin_type=True),
                )
                idx += 1
    env.close()
    return path


class TestMaxFilterBatchSize(unittest.TestCase):
    """Tests for ``batch_size='max:N'`` and ``batch_size='filter:N'``."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._uniform_path = _create_lmdb(
            f"{cls._tmpdir.name}/uniform.lmdb", nframes=10, natoms=6
        )
        cls._mixed_path = _create_mixed_nloc_lmdb(f"{cls._tmpdir.name}/mixed.lmdb")
        cls._type_map = ["O", "H"]

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_max_batch_size_single_nloc(self):
        """``max:N`` uses floor division and clamps to 1."""
        reader = LmdbDataReader(
            self._uniform_path, self._type_map, batch_size="max:500"
        )
        # floor(500 / 6) = 83.
        self.assertEqual(reader.get_batch_size_for_nloc(6), 83)
        self.assertEqual(reader._max_rule, 500)
        self.assertIsNone(reader._auto_rule)
        self.assertIsNone(reader._filter_rule)

        reader_small = LmdbDataReader(
            self._uniform_path, self._type_map, batch_size="max:5"
        )
        # floor(5 / 6) == 0 → clamped to 1 so every nloc still yields a batch.
        self.assertEqual(reader_small.get_batch_size_for_nloc(6), 1)

    def test_auto_vs_max_ceiling_vs_floor(self):
        """``auto:N`` rounds up; ``max:N`` rounds down for the same budget."""
        auto_reader = LmdbDataReader(
            self._uniform_path, self._type_map, batch_size="auto:1024"
        )
        max_reader = LmdbDataReader(
            self._uniform_path, self._type_map, batch_size="max:1000"
        )
        # nloc=148: ceil(1024/148)=7, floor(1000/148)=6
        self.assertEqual(auto_reader.get_batch_size_for_nloc(148), 7)
        self.assertEqual(max_reader.get_batch_size_for_nloc(148), 6)
        # nloc=2: ceil(1024/2)=512, floor(1000/2)=500
        self.assertEqual(auto_reader.get_batch_size_for_nloc(2), 512)
        self.assertEqual(max_reader.get_batch_size_for_nloc(2), 500)

    def test_filter_drops_large_nloc_groups(self):
        """``filter:N`` removes whole nloc groups above the threshold."""
        # _create_mixed_nloc_lmdb produces nloc groups {6:4, 9:4, 12:2}.
        r10 = LmdbDataReader(self._mixed_path, self._type_map, batch_size="filter:10")
        self.assertEqual(set(r10.nloc_groups.keys()), {6, 9})
        self.assertEqual(len(r10), 8)
        self.assertEqual(r10._max_rule, 10)
        self.assertEqual(r10._filter_rule, 10)

        r6 = LmdbDataReader(self._mixed_path, self._type_map, batch_size="filter:6")
        self.assertEqual(set(r6.nloc_groups.keys()), {6})
        self.assertEqual(len(r6), 4)

        r100 = LmdbDataReader(self._mixed_path, self._type_map, batch_size="filter:100")
        self.assertEqual(set(r100.nloc_groups.keys()), {6, 9, 12})
        self.assertEqual(len(r100), 10)

    def test_filter_preserves_system_id_numbering(self):
        """filter:N keeps original sid numbering and zeroes dropped systems."""
        path = f"{self._tmpdir.name}/mixed_sids.lmdb"
        # sid 0..2 at natoms=6; sid=3 at natoms=20 (fully dropped by filter:10).
        _create_mixed_sid_nloc_lmdb(
            path,
            system_specs=[(100, 6), (200, 6), (300, 6), (20, 20)],
            type_map=self._type_map,
        )
        reader = LmdbDataReader(path, self._type_map, batch_size="filter:10")
        # sid=3 is fully filtered but the numbering must survive so that
        # auto_prob block slicing keeps its user-facing semantics.
        self.assertEqual(reader.nsystems, 4)
        self.assertEqual(reader.system_nframes, [100, 200, 300, 0])
        self.assertEqual(reader.system_groups.get(3, []), [])
        self.assertEqual(len(reader), 600)

        block_targets = compute_block_targets(
            "prob_sys_size;0:3:0.5;3:4:0.5",
            nsystems=reader.nsystems,
            system_nframes=reader.system_nframes,
        )
        # Empty block (3:4) drops out, remaining block is already balanced
        # after re-normalisation → no expansion needed.
        self.assertEqual(block_targets, [])

    def test_filter_dataset_index_is_contiguous_and_live(self):
        """After filter:N, every i in range(len(reader)) is a live retrievable frame.

        Regression for the earlier indexing bug where ``len(reader)`` shrank
        to the retained count but ``__getitem__`` still indexed the original
        LMDB key space. Under filter:10 the mixed-nloc LMDB drops the two
        12-atom frames at original keys 8 & 9; we check here that:

        * every dataset index ``0..len(reader)-1`` decodes without raising
          and never returns a filtered-out frame, and
        * ``fid`` reports the stable original LMDB key, not the dataset
          index (so downstream logs survive the remap), and
        * out-of-range indices still raise IndexError.
        """
        reader = LmdbDataReader(
            self._mixed_path, self._type_map, batch_size="filter:10"
        )
        self.assertEqual(len(reader), 8)
        self.assertEqual(len(reader._retained_keys), 8)
        self.assertEqual(reader._retained_keys, [0, 1, 2, 3, 4, 5, 6, 7])

        seen_fids = []
        for i in range(len(reader)):
            frame = reader[i]
            self.assertLessEqual(frame["atype"].shape[0], 10)
            self.assertEqual(
                frame["fid"],
                reader._retained_keys[i],
                msg=f"fid should be the original LMDB key, not dataset index {i}",
            )
            seen_fids.append(frame["fid"])
        # Dropped original keys (8, 9) must never appear as fids.
        self.assertNotIn(8, seen_fids)
        self.assertNotIn(9, seen_fids)

        with self.assertRaises(IndexError):
            reader[len(reader)]
        with self.assertRaises(IndexError):
            reader[-1]

    def test_sampler_with_filter(self):
        """LmdbBatchSampler only emits retained, same-nloc frames."""
        reader = LmdbDataReader(
            self._mixed_path, self._type_map, batch_size="filter:10"
        )
        sampler = LmdbBatchSampler(reader, shuffle=False, seed=0)
        all_batches = list(sampler)
        all_indices = [idx for batch in all_batches for idx in batch]

        # (a) every frame in every batch has nloc <= 10
        for batch in all_batches:
            for idx in batch:
                self.assertLessEqual(reader.frame_nlocs[idx], 10)
        # (b) unique frame index count equals retained frames
        self.assertEqual(len(set(all_indices)), len(reader))
        self.assertEqual(len(reader), 8)
        # (c) each batch is same-nloc
        for batch in all_batches:
            nlocs = {reader.frame_nlocs[idx] for idx in batch}
            self.assertEqual(len(nlocs), 1)
        # The 12-atom frames were at original LMDB keys 8, 9; they must
        # never be reachable via any emitted dataset index.
        reached_original_keys = {reader._retained_keys[idx] for idx in all_indices}
        for original_key in (8, 9):
            self.assertNotIn(original_key, reached_original_keys)

    def test_invalid_batch_size_strings_rejected(self):
        """``<rule>:N`` specs with missing / non-positive N fail at init.

        Before this hardening, ``filter:0`` silently dropped every frame
        and ``max:`` raised a cryptic ``invalid literal for int()``.
        One case per failure mode is enough to pin the behaviour.
        """
        for spec in ("filter:", "filter:0", "max:-1"):
            with self.assertRaises(ValueError) as ctx:
                LmdbDataReader(self._uniform_path, self._type_map, batch_size=spec)
            self.assertIn("positive", str(ctx.exception))

    def test_auto_prob_with_filter_still_works(self):
        """compute_block_targets + sampler survive a fully-dropped block."""
        path = f"{self._tmpdir.name}/auto_prob_filter.lmdb"
        # filter:10 drops sid=2 (natoms=20), and sid=0 is under-represented
        # relative to sid=1 so at least one block still needs expansion.
        _create_mixed_sid_nloc_lmdb(
            path,
            system_specs=[(50, 6), (500, 6), (30, 20)],
            type_map=self._type_map,
        )
        reader = LmdbDataReader(path, self._type_map, batch_size="filter:10")
        self.assertEqual(reader.nsystems, 3)
        self.assertEqual(reader.system_nframes, [50, 500, 0])
        self.assertEqual(len(reader), 550)

        block_targets = compute_block_targets(
            "prob_sys_size;0:1:0.5;1:3:0.5",
            nsystems=reader.nsystems,
            system_nframes=reader.system_nframes,
        )
        # sid=2 in block 1:3 is empty but block 1:3 overall still has 500
        # frames, so compute_block_targets should produce finite targets.
        self.assertTrue(
            all(np.isfinite(target) for _sys_ids, target in block_targets),
            block_targets,
        )
        # Block 0 under-represented relative to weight → expansion needed.
        self.assertGreater(len(block_targets), 0)

        sampler = LmdbBatchSampler(
            reader, shuffle=False, seed=0, block_targets=block_targets
        )
        all_batches = list(sampler)
        all_indices = [idx for batch in all_batches for idx in batch]
        # Every index must be a retained frame — no dropped sid=2 / nloc=20.
        for idx in all_indices:
            self.assertLessEqual(reader.frame_nlocs[idx], 10)
            self.assertNotEqual(reader.frame_system_ids[idx], 2)
        # Every batch is same-nloc
        for batch in all_batches:
            nlocs = {reader.frame_nlocs[idx] for idx in batch}
            self.assertEqual(len(nlocs), 1)
        # Expansion produces more indices than the retained dataset size.
        self.assertGreater(len(all_indices), len(reader))


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

    def test_testdata_add_data_requirement_matches_manual_add(self):
        """DataRequirementItem forwarding matches manual requirement registration."""
        from deepmd.utils.data import (
            DataRequirementItem,
        )

        requirements = [
            DataRequirementItem(
                "drdq",
                ndof=6,
                atomic=True,
                must=False,
                high_prec=False,
                repeat=2,
                default=1.25,
                dtype=np.float64,
            ),
            DataRequirementItem(
                "aux",
                ndof=2,
                atomic=False,
                must=False,
                high_prec=False,
                repeat=3,
                default=-2.0,
                dtype=np.float32,
            ),
        ]
        manual = LmdbTestData(
            self._lmdb_path,
            type_map=self._type_map,
            shuffle_test=False,
        )
        forwarded = LmdbTestData(
            self._lmdb_path,
            type_map=self._type_map,
            shuffle_test=False,
        )
        for item in requirements:
            manual.add(
                item["key"],
                ndof=item["ndof"],
                atomic=item["atomic"],
                must=item["must"],
                high_prec=item["high_prec"],
                repeat=item["repeat"],
                default=item["default"],
                dtype=item["dtype"],
            )
        forwarded.add_data_requirement(requirements)

        manual_result = manual.get_test()
        forwarded_result = forwarded.get_test()
        for item in requirements:
            key = item["key"]
            self.assertEqual(forwarded_result[f"find_{key}"], 0.0)
            self.assertEqual(forwarded_result[key].shape, manual_result[key].shape)
            self.assertEqual(forwarded_result[key].dtype, manual_result[key].dtype)
            np.testing.assert_array_equal(forwarded_result[key], manual_result[key])

    def test_testdata_missing_key_not_found(self):
        """Keys absent from LMDB frames get find_*=0.0 in get_test()."""
        tmpdir = tempfile.TemporaryDirectory()
        path = _create_lmdb(f"{tmpdir.name}/plain.lmdb", nframes=3, natoms=6)
        td = LmdbTestData(path, type_map=["O", "H"], shuffle_test=False)
        result = td.get_test()
        # atom_pref is not in the plain LMDB
        self.assertEqual(result.get("find_atom_pref", 0.0), 0.0)
        tmpdir.cleanup()

    def test_testdata_required_key_must_exist(self):
        """A required LMDB test label cannot be replaced by a default value."""
        td = LmdbTestData(self._lmdb_path, type_map=self._type_map, shuffle_test=False)
        td.add("missing_label", 1, atomic=False, must=True)

        with self.assertRaisesRegex(RuntimeError, "missing_label"):
            td.get_test()

    def test_testdata_normalizes_atomic_label_prefix(self):
        """LMDB atomic_* labels use the same in-memory atom_* keys as NPY data."""
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path = _create_lmdb_with_extra_keys(
            f"{tmpdir.name}/atomic-label.lmdb",
            nframes=2,
            natoms=self._natoms,
            extra_keys={
                "atomic_dipole": (lambda n: (n, 3), np.float64),
            },
        )
        td = LmdbTestData(path, type_map=self._type_map, shuffle_test=False)
        td.add("atom_dipole", 3, atomic=True, must=True)

        result = td.get_test()

        self.assertEqual(result["find_atom_dipole"], 1.0)
        self.assertEqual(
            result["atom_dipole"].shape,
            (2, self._natoms * 3),
        )


class _StalledPool:
    """A pool whose decoder exited, leaving its submissions unfinished.

    This is what a decoder killed mid-result looks like from the parent: the
    futures never resolve and the pool never reports itself broken.
    """

    def __init__(self) -> None:
        self._processes = {1: SimpleNamespace(exitcode=-1)}
        self.submissions = 0

    def submit(self, *args: object, **kwargs: object) -> Future:
        self.submissions += 1
        return Future()


class TestDecoderPoolFailure(unittest.TestCase):
    """A dead decoder must not strand the run waiting for it."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self._path = _create_lmdb(
            f"{self._tmpdir.name}/pool.lmdb", nframes=12, natoms=6
        )
        self._reader = LmdbDataReader(self._path, ["O", "H"], batch_size=4)
        self.addCleanup(self._reader.close)
        # Keep the liveness check from pacing the test.
        patcher = mock.patch.object(
            lmdb_data_module, "_DECODER_LIVENESS_INTERVAL", 0.01
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self._entries: dict[int, object] = {}

    def _iterator(self, pool: object) -> LmdbBatchIterator:
        """Return an iterator over two four-frame batches served by ``pool``.

        Every iterator built on one stand-in pool receives the same entry, as
        the iterators of a rank share the pool registered for their worker
        count.
        """
        entry = self._entries.setdefault(
            id(pool), lmdb_data_module._LmdbPoolEntry(executor=pool, users=0)
        )
        entry.users += 1
        patcher = mock.patch.object(
            lmdb_data_module, "_acquire_lmdb_executor", return_value=entry
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        # The stand-in pool was never registered, so releasing it is a no-op.
        return LmdbBatchIterator(
            self._reader, [[0, 1, 2, 3], [4, 5, 6, 7]], num_workers=2
        )

    def _isolated_iterator(self) -> LmdbBatchIterator:
        """Return an iterator over a real pool that no other test shares.

        The decoder pool is process-wide, so a test that kills or signals its
        decoders is given one of its own rather than leaving the damage behind
        for its neighbours.
        """
        registry: dict = {}
        patcher = mock.patch.object(lmdb_data_module, "_LMDB_POOLS", registry)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(
            lambda: [
                entry.executor.shutdown(wait=False, cancel_futures=True)
                for entry in registry.values()
            ]
        )
        iterator = LmdbBatchIterator(
            self._reader, [[0, 1, 2, 3], [4, 5, 6, 7]], num_workers=2
        )
        self.addCleanup(iterator.close)
        return iterator

    def _assert_same_batch(self, batch: dict, expected: dict) -> None:
        self.assertEqual(sorted(batch), sorted(expected))
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(batch[key], value)

    def test_a_stalled_decoder_falls_back_and_is_not_retried(self) -> None:
        pool = _StalledPool()
        iterator = self._iterator(pool)

        with self.assertLogs(lmdb_data_module.log, level="WARNING") as captured:
            first = next(iterator)
        submissions = pool.submissions
        second = next(iterator)

        # Each batch is the one the pool was asked for, decoded here instead,
        # and the pool is not offered any more work.
        self._assert_same_batch(first, self._reader.decode_batch([0, 1, 2, 3]))
        self._assert_same_batch(second, self._reader.decode_batch([4, 5, 6, 7]))
        self.assertIn("decoder process exited", "\n".join(captured.output))
        self.assertEqual(pool.submissions, submissions)

    def test_close_detects_a_stalled_prefetch(self) -> None:
        """Closing an in-flight prefetch marks a lost decoder pool unhealthy."""
        pool = _StalledPool()
        iterator = self._iterator(pool)
        entry = self._entries[id(pool)]
        iterator._pool = entry
        running = Future()
        self.assertTrue(running.set_running_or_notify_cancel())
        iterator._pending = lmdb_data_module._PendingBatch([0, 1], [running])

        with self.assertLogs(lmdb_data_module.log, level="WARNING"):
            iterator.close()

        self.assertFalse(entry.healthy)

    def test_a_second_iterator_does_not_retry_a_lost_pool(self) -> None:
        """Pool health is shared, so the loss is discovered once for all."""
        pool = _StalledPool()
        first = self._iterator(pool)
        next(first)
        submissions = pool.submissions

        second = self._iterator(pool)
        batch = next(second)

        self._assert_same_batch(batch, self._reader.decode_batch([0, 1, 2, 3]))
        self.assertEqual(pool.submissions, submissions)

    @unittest.skipUnless(
        os.name == "posix" and sys.implementation.name == "cpython",
        "requires CPython's POSIX process-pool pipe",
    )
    def test_a_run_that_lost_its_pool_still_exits(self) -> None:
        """Losing a decoder must not leave the interpreter unable to exit.

        A pool reading the partial result of a decoder killed mid-write is
        joined by the interpreter on the way out, so a run could complete its
        training and then hang forever instead of terminating.
        """
        # Spawned decoders re-import the main module, so the scenario has to
        # live in a file rather than be passed on the command line.
        script = Path(self._tmpdir.name) / "lose_the_pool.py"
        script.write_text(
            textwrap.dedent("""
            import os
            import struct

            from deepmd.dpmodel.utils.lmdb_data import (
                _acquire_lmdb_executor,
                _release_lmdb_executor,
            )


            def idle():
                return os.getpid()


            if __name__ == "__main__":
                entry = _acquire_lmdb_executor(2)
                entry.executor.submit(idle).result()
                # A frame header promising more bytes than ever arrive, which
                # is what a decoder killed mid-write leaves behind.
                os.write(
                    entry.executor._result_queue._writer.fileno(),
                    struct.pack("!i", 4096) + b"partial",
                )
                entry.healthy = False
                _release_lmdb_executor(2)
            """)
        )
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])

    @unittest.skipUnless(
        hasattr(signal, "SIGKILL"),
        "SIGKILL is not available on this platform",
    )
    def test_killing_a_real_decoder_does_not_stop_the_run(self) -> None:
        """The pool reports itself broken, and the batch still arrives.

        A decoder that dies cleanly fails every future the pool holds, which
        is the other way the loss of a decoder reaches the iterator. Only a
        signal it cannot ignore gets it there.
        """
        iterator = self._isolated_iterator()
        next(iterator)
        processes = list(iterator._pool.executor._processes.values())
        self.assertTrue(processes)
        process = processes[0]
        self.assertTrue(process.is_alive())
        os.kill(process.pid, signal.SIGKILL)
        process.join(timeout=5)

        batch = next(iterator)

        self._assert_same_batch(batch, self._reader.decode_batch([4, 5, 6, 7]))
        self.assertFalse(iterator._pool.healthy)


if __name__ == "__main__":
    unittest.main()
