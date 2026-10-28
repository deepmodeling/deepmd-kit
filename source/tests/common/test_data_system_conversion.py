# SPDX-License-Identifier: LGPL-3.0-or-later
import json
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
    make_neighbor_stat_data,
)
from deepmd.entrypoints.test import test as run_model_test
from deepmd.utils import (
    data_conversion,
    data_system,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.data_system import (
    LmdbDataSystem,
    get_data,
    process_systems,
    validate_backend_data_config,
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


def _write_repeated_lmdb(path: str, nframes: int) -> None:
    """Write a compact same-nloc LMDB suitable for bounded-I/O assertions."""
    env = lmdb.open(path, map_size=64 * 1024 * 1024)
    frame = {
        "atom_names": ["H"],
        "atom_numbs": [1],
        "atom_types": _encode_array(np.array([0], dtype=np.int64)),
        "cells": _encode_array(np.eye(3, dtype=np.float64) * 8.0),
        "coords": _encode_array(np.zeros((1, 3), dtype=np.float64)),
        "energies": _encode_array(np.array([0.0], dtype=np.float64)),
        "forces": _encode_array(np.zeros((1, 3), dtype=np.float64)),
    }
    packed_frame = msgpack.packb(frame, use_bin_type=True)
    metadata = {
        "nframes": nframes,
        "frame_idx_fmt": "012d",
        "frame_nlocs": [1] * nframes,
        "type_map": ["H"],
        "system_info": {
            "formula": "H",
            "natoms": [1],
            "nframes": nframes,
        },
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for index in range(nframes):
            txn.put(f"{index:012d}".encode(), packed_frame)
    env.close()


def _write_mixed_nloc_lmdb(path: str) -> None:
    """Write two frames that exercise legacy mix:N padding."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    frames = []
    for nloc in (1, 2):
        frames.append(
            {
                "atom_names": ["H"],
                "atom_numbs": [nloc],
                "atom_types": _encode_array(np.zeros(nloc, dtype=np.int64)),
                "cells": _encode_array(np.eye(3, dtype=np.float64) * 8.0),
                "coords": _encode_array(np.zeros((nloc, 3), dtype=np.float64)),
                "energies": _encode_array(np.array([0.0], dtype=np.float64)),
                "forces": _encode_array(np.zeros((nloc, 3), dtype=np.float64)),
            }
        )
    metadata = {
        "nframes": 2,
        "frame_idx_fmt": "012d",
        "frame_nlocs": [1, 2],
        "type_map": ["H"],
        "system_info": {
            "formula": "H",
            "natoms": [1],
            "nframes": 2,
        },
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for index, frame in enumerate(frames):
            txn.put(f"{index:012d}".encode(), msgpack.packb(frame, use_bin_type=True))
    env.close()


def _write_mixed_pbc_lmdb(path: str) -> None:
    """Write equal-nloc periodic and non-periodic frames."""
    env = lmdb.open(path, map_size=10 * 1024 * 1024)
    frames = []
    for cell in (np.eye(3, dtype=np.float64) * 8.0, np.zeros((3, 3))):
        frames.append(
            {
                "atom_names": ["H"],
                "atom_numbs": [1],
                "atom_types": _encode_array(np.array([0], dtype=np.int64)),
                "cells": _encode_array(cell),
                "coords": _encode_array(np.zeros((1, 3), dtype=np.float64)),
                "energies": _encode_array(np.array([0.0], dtype=np.float64)),
                "forces": _encode_array(np.zeros((1, 3), dtype=np.float64)),
            }
        )
    metadata = {
        "nframes": 2,
        "frame_idx_fmt": "012d",
        "frame_nlocs": [1, 1],
        "type_map": ["H"],
        "system_info": {"formula": "H", "natoms": [1], "nframes": 2},
    }
    with env.begin(write=True) as txn:
        txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
        for index, frame in enumerate(frames):
            txn.put(f"{index:012d}".encode(), msgpack.packb(frame, use_bin_type=True))
    env.close()


class _FakeMultiSystems:
    write_count = 0
    load_calls: ClassVar[list[tuple[str, str]]] = []
    to_calls: ClassVar[list[tuple[str, str, dict]]] = []

    def __init__(self, *systems) -> None:
        self.systems = list(systems)
        self.loaded = False

    def load_systems_from_file(self, file_name: str, fmt: str):
        self.load_calls.append((file_name, fmt))
        self.loaded = True
        return self

    def __len__(self) -> int:
        return 1 if self.loaded or self.systems else 0

    def to(self, fmt: str, file_name: str, **kwargs) -> None:
        type(self).write_count += 1
        self.to_calls.append((fmt, file_name, kwargs))
        if fmt == "deepmd/hdf5":
            _write_minimal_deepmd_hdf5(file_name)
        elif fmt == "deepmd/lmdb":
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
        self.addCleanup(self.tmpdir.cleanup)
        self.root = Path(self.tmpdir.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, self.old_cwd)
        self.source = self.root / "data.extxyz"
        self.source.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")
        _FakeMultiSystems.write_count = 0
        _FakeMultiSystems.load_calls = []
        _FakeMultiSystems.to_calls = []
        data_conversion._DPDATA_CONVERSION_CACHE.clear()
        self.fake_dpdata = types.SimpleNamespace(
            MultiSystems=_FakeMultiSystems,
            LabeledSystem=_FakeLabeledSystem,
        )

    def tearDown(self) -> None:
        data_conversion._DPDATA_CONVERSION_CACHE.clear()

    def test_process_systems_defaults_to_deepmd_lmdb_and_reuses_cache(self) -> None:
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
        self.assertEqual(
            _FakeMultiSystems.to_calls,
            [("deepmd/lmdb", systems[0], {"overwrite": True})],
        )

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

    def test_process_systems_revalidates_in_memory_cache(self) -> None:
        """A source rewrite in one process must refresh its converted output."""
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            systems = process_systems(str(self.source), fmt="extxyz")
            output_mtime = Path(systems[0]).stat().st_mtime
            refreshed_mtime = output_mtime + 1.0
            os.utime(self.source, (refreshed_mtime, refreshed_mtime))
            systems_again = process_systems(str(self.source), fmt="extxyz")

        self.assertEqual(systems, systems_again)
        self.assertEqual(_FakeMultiSystems.write_count, 2)
        self.assertTrue(
            all(
                call == ("deepmd/lmdb", systems[0], {"overwrite": True})
                for call in _FakeMultiSystems.to_calls
            )
        )

    def test_lmdb_alias_uses_canonical_dpdata_writer(self) -> None:
        """The legacy alias must share the canonical dpdata cache entry."""
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            systems = process_systems(str(self.source), fmt="extxyz", out_fmt="lmdb")

        self.assertEqual(
            _FakeMultiSystems.to_calls,
            [("deepmd/lmdb", systems[0], {"overwrite": True})],
        )

    def test_real_dpdata_lmdb_writer_is_deepmd_compatible(self) -> None:
        """Verify dpdata 1.1 writes the schema consumed by DeePMD's LMDB reader."""
        extxyz = (
            "1\n"
            "Properties=species:S:1:pos:R:3:forces:R:3 energy=0.0 "
            'Lattice="8 0 0 0 8 0 0 0 8"\n'
            "H 0 0 0 0 0 0\n"
        )
        self.source.write_text(extxyz)

        systems = process_systems(str(self.source), fmt="extxyz")
        # A changed source exercises dpdata's own transactional overwrite path;
        # DeePMD-kit must not remove or rename the LMDB directory around it.
        original_stat = self.source.stat()
        self.source.write_text(extxyz.replace("energy=0.0", "energy=42.0"))
        os.utime(self.source, (original_stat.st_atime, original_stat.st_mtime - 86400))
        data_conversion._DPDATA_CONVERSION_CACHE.clear()
        systems_again = process_systems(str(self.source), fmt="extxyz")

        self.assertEqual(len(systems), 1)
        self.assertEqual(systems_again, systems)
        self.assertTrue(is_lmdb(systems[0]))
        data = LmdbDataSystem(systems[0], ["H"], batch_size=1)
        self.addCleanup(data.close)
        data.add_data_requirements(
            [DataRequirementItem("energy", 1, atomic=False, must=True)]
        )
        batch = data.get_batch()
        self.assertEqual(float(batch["energy"].item()), 42.0)
        self.assertEqual(batch["coord"].shape, (1, 3))
        self.assertEqual(batch["type"].shape, (1, 1))

    def test_cache_cleanup_unlinks_directory_symlink(self) -> None:
        """Cleanup must not recurse through a symlink outside the cache."""
        target = self.root / "outside"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_text("keep")
        link = self.root / ".deepmd_dpdata_cache" / "stale.tmp"
        link.parent.mkdir()
        link.symlink_to(target, target_is_directory=True)

        data_conversion._remove_path(link)

        self.assertFalse(link.exists())
        self.assertTrue(target.is_dir())
        self.assertEqual(sentinel.read_text(), "keep")

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
            self.addCleanup(data.close)

        self.assertEqual(data.get_nsystems(), 1)
        self.assertIsInstance(data, LmdbDataSystem)
        self.assertTrue(data.mixed_type)
        self.assertEqual(_FakeMultiSystems.load_calls, [(str(self.source), "extxyz")])
        batch = data.get_batch()
        self.assertIn("type", batch)
        self.assertIn("natoms_vec", batch)
        self.assertEqual(batch["coord"].shape, (1, 3))
        self.assertIsNot(data.data_systems[0], data)
        self.assertEqual(data.type_map, ["H"])
        self.assertEqual(data.data_systems[0].get_test()["coord"].shape, (1, 3))
        stat_set = data._load_set(data.dirs[0])
        self.assertEqual(stat_set["coord"].shape, (1, 3))
        self.assertEqual(stat_set["type"].shape, (1, 1))
        data.close()
        data.close()

    def test_multiple_lmdb_paths_use_native_group_composition(self) -> None:
        """Converted support preserves upstream's multiple-source data API."""
        lmdb_a = self.root / "a.lmdb"
        lmdb_b = self.root / "b.lmdb"
        _write_minimal_lmdb(str(lmdb_a))
        _write_minimal_lmdb(str(lmdb_b))
        data = get_data(
            {"systems": [str(lmdb_a), str(lmdb_b)], "batch_size": 1},
            0.0,
            ["H"],
            None,
        )
        self.addCleanup(data_system.close_data_systems, data)
        self.assertEqual(len(data.data_systems), 2)
        self.assertEqual(data.get_batch()["coord"].shape, (1, 3))

    def test_lmdb_stack_frames_rejects_empty_batch(self) -> None:
        lmdb_path = self.root / "empty-batch.lmdb"
        _write_minimal_lmdb(str(lmdb_path))
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size=1)
        self.addCleanup(data.close)

        with self.assertRaisesRegex(ValueError, "empty LMDB frame batch"):
            data._stack_frames([])

    def test_requirements_can_be_registered_after_adapter_construction(self) -> None:
        """PBC probing must not freeze the model's label contract."""
        lmdb_path = self.root / "requirements.lmdb"
        _write_minimal_lmdb(str(lmdb_path))
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size=[1])
        self.addCleanup(data.close)

        data.add_data_requirements(
            [DataRequirementItem("energy", 1, atomic=False, must=True)]
        )
        batch = data.get_batch()

        self.assertEqual(batch["energy"].shape, (1, 1))
        self.assertEqual(float(batch["find_energy"]), 1.0)

    def test_legacy_collation_ands_find_flags(self) -> None:
        lmdb_path = self.root / "find-flags.lmdb"
        _write_minimal_lmdb(str(lmdb_path))
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size=2)
        self.addCleanup(data.close)
        data.add_data_requirements(
            [DataRequirementItem("energy", 1, atomic=False, must=False)]
        )
        first = data._reader.peek_frame(0)
        second = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in first.items()
        }
        second["find_energy"] = np.float32(0.0)

        batch = data._stack_frames([first, second])

        self.assertEqual(float(batch["find_energy"]), 0.0)

    def test_neighbor_stat_reads_are_sampled_and_chunked(self) -> None:
        """A large same-nloc group must never be decoded as one Python list."""
        lmdb_path = self.root / "bounded.lmdb"
        _write_repeated_lmdb(str(lmdb_path), 2101)
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size=1000)
        self.addCleanup(data.close)

        total_sampled = sum(len(indices) for indices in data._nloc_set_indices.values())
        self.assertEqual(total_sampled, 2000)
        self.assertLessEqual(
            max(len(indices) for indices in data._nloc_set_indices.values()), 128
        )
        first = data._load_set(data.dirs[0])
        self.assertLessEqual(first["coord"].shape[0], 128)
        self.assertEqual(data.get_stat_nsystems(), 1)
        self.assertEqual(data.get_stat_numb_batches(0), 17)
        self.assertEqual(data.get_stat_batch(0)["coord"].shape, (128, 3))

    def test_mix_batch_pads_different_atom_counts(self) -> None:
        lmdb_path = self.root / "mixed.lmdb"
        _write_mixed_nloc_lmdb(str(lmdb_path))
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size="mix:4", seed=0)
        self.addCleanup(data.close)

        batch = data.get_batch()

        self.assertEqual(batch["coord"].shape, (2, 6))
        self.assertEqual(batch["type"].shape, (2, 2))
        self.assertIn(-1, batch["type"])

    def test_periodic_and_nonperiodic_frames_are_separate_views(self) -> None:
        lmdb_path = self.root / "mixed-pbc.lmdb"
        _write_mixed_pbc_lmdb(str(lmdb_path))
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size=2, seed=0)
        self.addCleanup(data.close)

        self.assertEqual(len(data.data_systems), 2)
        self.assertEqual({view.pbc for view in data.data_systems}, {False, True})
        batches = [data.get_batch(), data.get_batch()]
        self.assertEqual({float(batch["find_box"]) for batch in batches}, {0.0, 1.0})

    def test_multiple_conversion_inputs_use_grouped_readers(self) -> None:
        """Each input has its own cache and composes through the native API."""
        second = self.root / "second.extxyz"
        second.write_text(self.source.read_text())
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            data = get_data(
                {
                    "systems": [str(self.source), str(second)],
                    "format": "extxyz",
                    "batch_size": 1,
                    "auto_prob": "prob_uniform",
                },
                0,
                ["H"],
                None,
            )
        self.addCleanup(data.close)
        self.assertEqual(_FakeMultiSystems.write_count, 2)
        self.assertEqual(len(data.children), 2)
        self.assertEqual(data.get_batch()["coord"].shape, (1, 3))
        # A grouped checkpoint must restore the next batch, not its prefetch.
        state = data.state_dict()
        expected = data.get_batch()
        data.load_state_dict(state)
        np.testing.assert_array_equal(data.get_batch()["coord"], expected["coord"])

    def test_group_construction_failure_closes_completed_readers(self) -> None:
        """Factories retain ownership until every group is constructed."""
        paths = [self.root / "valid.lmdb", self.root / "later.lmdb"]
        for path in paths:
            _write_minimal_lmdb(str(path))
        acquired = []

        def factory(path, batch_size, auto_prob):
            if acquired:
                raise ValueError("later source failed")
            reader = LmdbDataSystem(path, ["H"], batch_size, auto_prob)
            acquired.append(reader)
            return reader

        with self.assertRaisesRegex(ValueError, "later source failed"):
            get_data(
                {"systems": list(map(str, paths)), "batch_size": 1},
                0,
                ["H"],
                None,
                lmdb_factory=factory,
            )
        self.assertTrue(acquired[0]._closed)

    def test_rare_nonperiodic_frame_outside_probe_is_partitioned(self) -> None:
        """A single missed box must never disable PBC on its periodic neighbors."""
        path = self.root / "rare-pbc.lmdb"
        _write_repeated_lmdb(str(path), 300)
        probe = {300 * i // 256 for i in range(256)}
        rare = next(i for i in range(300) if i not in probe)
        with lmdb.open(str(path)) as env, env.begin(write=True) as txn:
            key = f"{rare:012d}".encode()
            frame = msgpack.unpackb(txn.get(key), raw=False)
            frame["cells"] = _encode_array(np.zeros((3, 3)))
            txn.put(key, msgpack.packb(frame, use_bin_type=True))
        data = LmdbDataSystem(str(path), ["H"], batch_size=32, seed=0)
        self.addCleanup(data.close)
        self.assertEqual({view.pbc for view in data.data_systems}, {False, True})
        seen = {False: 0, True: 0}
        for _ in range(data.get_batch_pass_length()):
            batch = data.get_batch()
            periodic = bool(batch["find_box"])
            cells = batch["box"].reshape(-1, 9)
            self.assertTrue(np.all(np.any(cells != 0, axis=1) == periodic))
            self.assertEqual(len(batch["default_mesh"]), 7 if periodic else 1)
            seen[periodic] += len(cells)
        self.assertEqual(seen, {False: 1, True: 299})
        # Full-validation views use their own availability index.
        groups = data._test_data.availability_groups(np.arange(300))
        self.assertEqual(sorted(map(len, groups)), [1, 299])
        neighbor_data = make_neighbor_stat_data(str(path), ["H"])
        self.assertEqual(
            {view.pbc for view in neighbor_data.data_systems}, {False, True}
        )
        for view in neighbor_data.data_systems:
            box = view._load_set(view.dirs[0])["box"]
            self.assertEqual(box is not None, view.pbc)

    def test_legacy_composition_counts_obey_mixed_type_invariant(self) -> None:
        """Independent per-type maxima must not overrun the padded atom axis."""
        path = self.root / "compositions.lmdb"
        _write_repeated_lmdb(str(path), 2)
        with lmdb.open(str(path)) as env, env.begin(write=True) as txn:
            frame = msgpack.unpackb(txn.get(b"000000000001"), raw=False)
            metadata = msgpack.unpackb(txn.get(b"__metadata__"), raw=False)
            metadata["type_map"] = ["H", "O"]
            txn.put(b"__metadata__", msgpack.packb(metadata, use_bin_type=True))
            frame["atom_names"] = ["H", "O"]
            frame["atom_numbs"] = [0, 1]
            frame["atom_types"] = _encode_array(np.array([1], dtype=np.int64))
            txn.put(b"000000000001", msgpack.packb(frame, use_bin_type=True))
        data = LmdbDataSystem(str(path), ["H", "O"], batch_size=2, seed=0)
        self.addCleanup(data.close)
        batch = data.get_batch()
        np.testing.assert_array_equal(batch["natoms_vec"], [1, 1, 1, 0])
        self.assertEqual(
            set(map(tuple, batch["real_natoms_vec"])), {(1, 1, 1, 0), (1, 1, 0, 1)}
        )

    def test_lmdb_sys_probs_fail_before_conversion(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support explicit sys_probs"):
            get_data(
                {
                    "systems": str(self.source),
                    "format": "extxyz",
                    "batch_size": 1,
                    "sys_probs": [1.0],
                },
                0.0,
                ["H"],
                None,
            )

        self.assertEqual(_FakeMultiSystems.write_count, 0)

    def test_paddle_rejects_default_lmdb_before_conversion(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "Paddle backend"):
            validate_backend_data_config(
                {"systems": str(self.source), "format": "extxyz"},
                backend_name="Paddle",
                lmdb_supported=False,
            )

        self.assertEqual(_FakeMultiSystems.write_count, 0)

    def test_waiter_checks_source_only_after_lock_release(self) -> None:
        output = self.root / "wait.lmdb"
        output.mkdir()
        lock_path = output.with_suffix(".lmdb.lock")
        lock_path.write_text(
            json.dumps(
                {
                    "hostname": data_conversion.socket.gethostname(),
                    "pid": os.getpid(),
                    "process_start": data_conversion._process_start_time(os.getpid()),
                }
            )
        )
        waits = 0

        def release_after_long_valid_conversion(_seconds: float) -> None:
            nonlocal waits
            waits += 1
            if waits == 301:
                lock_path.unlink()

        with (
            patch.object(
                data_conversion.time, "sleep", release_after_long_valid_conversion
            ),
            patch.object(
                data_conversion,
                "_is_conversion_current",
                return_value=True,
            ) as is_current,
        ):
            self.assertTrue(
                data_conversion._wait_for_conversion(self.source, output, lock_path)
            )

        self.assertEqual(waits, 301)
        is_current.assert_called_once_with(self.source, output)

    def test_dead_conversion_owner_is_recovered(self) -> None:
        lock_path = self.root / "dead.lock"
        lock_path.write_text(
            json.dumps(
                {
                    "hostname": data_conversion.socket.gethostname(),
                    "pid": 2147483647,
                    "process_start": "missing",
                }
            )
        )

        self.assertTrue(data_conversion._recover_stale_conversion_lock(lock_path))
        self.assertFalse(lock_path.exists())

    def test_directory_publication_rolls_back_on_failure(self) -> None:
        output = self.root / "published"
        output.mkdir()
        (output / "old.txt").write_text("old")
        staged = self.root / "staged"
        staged.mkdir()
        (staged / "new.txt").write_text("new")
        real_replace = os.replace
        calls = 0

        def fail_publication(source: str | Path, target: str | Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("publish failed")
            real_replace(source, target)

        with patch.object(data_conversion.os, "replace", fail_publication):
            with self.assertRaisesRegex(OSError, "publish failed"):
                data_conversion._publish_conversion_output(staged, output)

        self.assertEqual((output / "old.txt").read_text(), "old")
        self.assertFalse((output / "new.txt").exists())

    def test_unlabeled_input_uses_dpdata_unlabeled_loader(self) -> None:
        class UnlabeledMultiSystems(_FakeMultiSystems):
            calls: ClassVar[list[bool]] = []

            def load_systems_from_file(
                self, file_name: str, fmt: str, *, labeled: bool = True
            ):
                type(self).calls.append(labeled)
                if labeled:
                    raise ValueError("missing labels")
                self.loaded = True
                return self

        UnlabeledMultiSystems.calls = []
        fake_dpdata = types.SimpleNamespace(
            MultiSystems=UnlabeledMultiSystems,
            LabeledSystem=_FakeLabeledSystem,
        )

        with patch.dict(sys.modules, {"dpdata": fake_dpdata}):
            systems = process_systems(str(self.source), fmt="extxyz")

        self.assertEqual(UnlabeledMultiSystems.calls, [True, False])
        self.assertTrue(is_lmdb(systems[0]))

    def test_cache_path_changes_with_dpdata_version(self) -> None:
        with patch.object(
            data_conversion.importlib.metadata,
            "version",
            side_effect=["1.1.0", "1.2.0"],
        ):
            first = data_conversion._conversion_cache_path(
                self.source, "extxyz", "deepmd/lmdb"
            )
            second = data_conversion._conversion_cache_path(
                self.source, "extxyz", "deepmd/lmdb"
            )

        self.assertNotEqual(first, second)

    def test_restored_sources_invalidate_memory_and_disk_caches(self) -> None:
        """Restored mtimes must invalidate file and directory conversions."""
        for directory in (False, True):
            for fresh_process in (False, True):
                for preserve_mtime in (False, True):
                    with self.subTest(
                        directory=directory,
                        fresh_process=fresh_process,
                        preserve_mtime=preserve_mtime,
                    ):
                        source = self.source
                        if directory:
                            source = self.root / "source-dir"
                            child = source / "nested"
                            child.mkdir(parents=True, exist_ok=True)
                            source_file = child / "frame.extxyz"
                        else:
                            source_file = source
                        source_file.write_text("original")
                        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
                            systems = process_systems(str(source), fmt="extxyz")
                            before = _FakeMultiSystems.write_count
                            stat = source_file.stat()
                            source_file.write_text("restored")  # Same byte count.
                            mtime = stat.st_mtime_ns - (
                                0 if preserve_mtime else 86400 * 10**9
                            )
                            os.utime(source_file, ns=(stat.st_atime_ns, mtime))
                            if fresh_process:
                                data_conversion._DPDATA_CONVERSION_CACHE.clear()
                            self.assertEqual(
                                process_systems(str(source), fmt="extxyz"), systems
                            )
                            self.assertEqual(_FakeMultiSystems.write_count, before + 1)
                            process_systems(str(source), fmt="extxyz")
                            self.assertEqual(_FakeMultiSystems.write_count, before + 1)

    def test_source_signature_tracks_tree_membership_and_excludes_cache(self) -> None:
        """Publishing within the source directory cannot invalidate itself."""
        output = self.root / ".deepmd_dpdata_cache" / "cached.lmdb"
        first = data_conversion._source_signature(self.root, output)
        output.mkdir(parents=True)
        (output / "data.mdb").write_text("cache data")
        self.assertEqual(first, data_conversion._source_signature(self.root, output))
        child = self.root / "new.extxyz"
        child.write_text("new input")
        self.assertNotEqual(first, data_conversion._source_signature(self.root, output))
        child.unlink()
        self.assertEqual(first, data_conversion._source_signature(self.root, output))

    def test_directory_conversion_tracks_dangling_symlinks(self) -> None:
        """Unrelated broken links do not block conversion or hide later changes."""
        source = self.root / "source-tree"
        source.mkdir()
        (source / "frame.extxyz").write_text(self.source.read_text())
        link = source / "optional.extxyz"
        link.symlink_to(self.root / "missing.extxyz")
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            systems = process_systems(str(source), fmt="extxyz")
            self.assertEqual(process_systems(str(source), fmt="extxyz"), systems)
            self.assertEqual(_FakeMultiSystems.write_count, 1)

            # Changing a broken link's destination is a source-tree change.
            link.unlink()
            target = self.root / "target.extxyz"
            link.symlink_to(target)
            process_systems(str(source), fmt="extxyz")
            self.assertEqual(_FakeMultiSystems.write_count, 2)

            # Resolving a formerly broken link must also invalidate the cache.
            target.write_text(self.source.read_text())
            process_systems(str(source), fmt="extxyz")
            self.assertEqual(_FakeMultiSystems.write_count, 3)

    def test_valid_symlink_target_edits_invalidate_conversion(self) -> None:
        """Follow valid links even when target size and mtime are unchanged."""
        source = self.root / "source-tree"
        source.mkdir()
        (source / "frame.extxyz").symlink_to(self.source)
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            process_systems(str(source), fmt="extxyz")
            before = self.source.stat()
            self.source.write_text(self.source.read_text().replace("H 0", "H 1"))
            os.utime(self.source, ns=(before.st_atime_ns, before.st_mtime_ns))
            process_systems(str(source), fmt="extxyz")
            self.assertEqual(_FakeMultiSystems.write_count, 2)

    def test_directory_symlinks_track_edits_retargets_and_cycles(self) -> None:
        """ABACUS-style linked output trees participate in cache freshness."""
        source = self.root / "run"
        source.mkdir()
        targets = [self.root / "out-a", self.root / "out-b"]
        for target in targets:
            target.mkdir()
            (target / "running_scf.log").write_text("energy 0")
        link = source / "OUT.ABACUS"
        link.symlink_to(targets[0], target_is_directory=True)
        # Follow a back edge without recursing forever.
        (targets[0] / "parent").symlink_to(source, target_is_directory=True)
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            process_systems(str(source), fmt="abacus/scf")
            process_systems(str(source), fmt="abacus/scf")
            self.assertEqual(_FakeMultiSystems.write_count, 1)
            child = targets[0] / "running_scf.log"
            before = child.stat()
            child.write_text("energy 42")
            os.utime(child, ns=(before.st_atime_ns, before.st_mtime_ns))
            process_systems(str(source), fmt="abacus/scf")
            self.assertEqual(_FakeMultiSystems.write_count, 2)
            link.unlink()
            link.symlink_to(targets[1], target_is_directory=True)
            process_systems(str(source), fmt="abacus/scf")
            self.assertEqual(_FakeMultiSystems.write_count, 3)

    def test_source_signature_rejects_disappearing_regular_file(self) -> None:
        """Only dangling links may be fingerprinted after followed stat fails."""
        output = self.root / data_conversion._DPDATA_CACHE_DIR / "output.lmdb"
        original_stat = Path.stat

        def stat_after_removal(path: Path, **kwargs: object) -> os.stat_result:
            if path == self.source:
                raise FileNotFoundError("source disappeared")
            return original_stat(path, **kwargs)

        with patch.object(Path, "stat", stat_after_removal):
            with self.assertRaisesRegex(FileNotFoundError, "source disappeared"):
                data_conversion._source_signature(self.root, output)

    def test_source_change_during_conversion_does_not_publish_manifest(self) -> None:
        """A concurrent edit cannot mark a possibly mixed snapshot fresh."""

        def changing_writer(source: Path, fmt: str, out_fmt: str, output: Path) -> None:
            _write_minimal_lmdb(str(output))
            source.write_text("changed while converting")

        with patch.object(data_conversion, "_write_dpdata_conversion", changing_writer):
            with self.assertRaisesRegex(
                RuntimeError, "input changed during conversion"
            ):
                process_systems(str(self.source), fmt="extxyz")
        output = data_conversion._conversion_cache_path(
            self.source, "extxyz", "deepmd/lmdb"
        )
        self.assertFalse(data_conversion._is_conversion_current(self.source, output))
        self.assertFalse(data_conversion._conversion_manifest_path(output).exists())
        self.assertFalse(output.with_suffix(".lmdb.lock").exists())

    def test_missing_or_invalid_manifest_rebuilds_cache(self) -> None:
        """Legacy and interrupted conversions cannot validate via output mtime."""
        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            systems = process_systems(str(self.source), fmt="extxyz")
            manifest = data_conversion._conversion_manifest_path(Path(systems[0]))
            for invalid in (None, "bad json", "null", "{}"):
                if invalid is None:
                    manifest.unlink()
                else:
                    manifest.write_text(invalid)
                before = _FakeMultiSystems.write_count
                process_systems(str(self.source), fmt="extxyz")
                self.assertEqual(_FakeMultiSystems.write_count, before + 1)

    def test_auto_format_handles_xyz_headers_and_decode_errors(self) -> None:
        """Probe XYZ headers only when readable, otherwise use the suffix."""
        xyz = self.root / "frame.xyz"
        for header, expected in (
            (b"1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n", "extxyz"),
            (b'1\nLattice="8 0 0 0 8 0 0 0 8"\nH 0 0 0\n', "extxyz"),
            (b"1\nplain comment\nH 0 0 0\n", "xyz"),
            (b"1\n\xff\xfe\n", "xyz"),
        ):
            xyz.write_bytes(header)
            self.assertEqual(
                data_conversion._normalize_dpdata_format("auto", xyz), expected
            )
        with patch.object(Path, "open", side_effect=OSError("unreadable")):
            self.assertEqual(
                data_conversion._normalize_dpdata_format("auto", xyz), "xyz"
            )
        xyz.unlink()
        self.assertEqual(data_conversion._normalize_dpdata_format("auto", xyz), "xyz")
        for name, expected in (
            ("frame.traj", "ase/traj"),
            ("frame.extxyz", "extxyz"),
            ("frame", "auto"),
        ):
            self.assertEqual(
                data_conversion._normalize_dpdata_format("auto", Path(name)), expected
            )
        self.assertEqual(
            data_conversion._normalize_dpdata_format("ase", xyz), "ase/structure"
        )

    def test_lmdb_modifier_fails_before_conversion(self) -> None:
        """Never silently drop modifiers on direct or converted LMDB data."""
        lmdb_path = self.root / "direct.lmdb"
        _write_minimal_lmdb(str(lmdb_path))
        for params in (
            {"systems": str(self.source), "format": "extxyz", "batch_size": 1},
            {"systems": str(lmdb_path), "batch_size": 1},
        ):
            with self.assertRaisesRegex(ValueError, "does not support data modifiers"):
                get_data(params, 0.0, ["H"], object())
        self.assertEqual(_FakeMultiSystems.write_count, 0)

    def test_lmdb_batch_order_respects_shared_training_seed(self) -> None:
        """Shared training seeds preserve reproducible, rank-specific shuffling."""
        lmdb_path = self.root / "seed.lmdb"
        _write_repeated_lmdb(str(lmdb_path), 32)

        def order(seed: list[int]) -> list[list[int]]:
            data_system.dp_random.seed(seed)
            with get_data(
                {"systems": str(lmdb_path), "batch_size": 2}, 0.0, ["H"], None
            ) as data:
                return [list(batch) for batch in data._sampler]

        self.assertEqual(order([0, 42]), order([0, 42]))
        self.assertNotEqual(order([0, 42]), order([0, 43]))
        self.assertNotEqual(order([0, 42]), order([1, 42]))

    def test_lock_owner_liveness_branches(self) -> None:
        """Only same-host process evidence can establish owner liveness."""
        hostname = data_conversion.socket.gethostname()
        self.assertIsNone(
            data_conversion._lock_owner_is_alive({"hostname": "other-host"})
        )
        for payload in ({}, {"pid": None}, {"pid": "invalid"}, {"pid": 0}, {"pid": -1}):
            self.assertIsNone(
                data_conversion._lock_owner_is_alive({"hostname": hostname, **payload})
            )
        payload = {"hostname": hostname, "pid": os.getpid(), "process_start": "123"}
        with patch.object(data_conversion, "_process_start_time", return_value="123"):
            self.assertTrue(data_conversion._lock_owner_is_alive(payload))
        with patch.object(data_conversion, "_process_start_time", return_value="456"):
            self.assertFalse(data_conversion._lock_owner_is_alive(payload))
        for error, expected in (
            (None, True),
            (PermissionError(), True),
            (ProcessLookupError(), False),
        ):
            with (
                patch.object(data_conversion, "_process_start_time", return_value=None),
                patch.object(data_conversion.os, "kill", side_effect=error),
            ):
                self.assertEqual(
                    data_conversion._lock_owner_is_alive(payload), expected
                )

    def test_stale_lock_recovery_preserves_live_owners_and_successors(self) -> None:
        """An old heartbeat alone cannot evict a known live local owner."""
        path = self.root / "owner.lock"
        for alive, expired, expected in (
            (True, True, False),
            (None, False, False),
            (None, True, True),
            (False, False, True),
        ):
            path.write_text("{}")
            stamp = data_conversion.time.time() - (
                data_conversion._CONVERSION_LOCK_STALE_SECONDS + 1 if expired else 0
            )
            os.utime(path, (stamp, stamp))
            with patch.object(
                data_conversion, "_lock_owner_is_alive", return_value=alive
            ):
                self.assertEqual(
                    data_conversion._recover_stale_conversion_lock(path), expected
                )
            self.assertEqual(path.exists(), not expected)
        path.write_text("{}")
        with (
            patch.object(data_conversion, "_lock_owner_is_alive", return_value=False),
            patch.object(data_conversion, "_same_lock_file", return_value=False),
        ):
            self.assertFalse(data_conversion._recover_stale_conversion_lock(path))
        self.assertTrue(path.exists())

    def test_remote_lock_is_never_evicted_on_stale_heartbeat(self) -> None:
        """A remote heartbeat cached by NFS is not proof that its writer died."""
        path = self.root / "remote.lock"
        path.write_text(json.dumps({"hostname": "other-host", "pid": 123}))
        thirty_seconds_ago = data_conversion.time.time() - 31
        os.utime(path, (thirty_seconds_ago, thirty_seconds_ago))
        self.assertFalse(data_conversion._recover_stale_conversion_lock(path))
        expired = (
            data_conversion.time.time()
            - data_conversion._CONVERSION_LOCK_STALE_SECONDS
            - 1
        )
        os.utime(path, (expired, expired))
        with self.assertRaisesRegex(RuntimeError, "Cannot verify the remote owner"):
            data_conversion._recover_stale_conversion_lock(path)
        self.assertTrue(path.exists())

    def test_lock_initialization_failure_closes_file_and_removes_lock(self) -> None:
        """Metadata, payload or heartbeat failures cannot strand a fresh lock."""
        for target in ("os.fstat", "json.dump", "threading.Thread.start"):
            path = self.root / "init.lock"
            with patch(
                f"deepmd.utils.data_conversion.{target}",
                side_effect=OSError("init failed"),
            ):
                with path.open("x") as lock_file:
                    with self.assertRaisesRegex(OSError, "init failed"):
                        data_conversion._ConversionLock(path, lock_file)
                self.assertTrue(lock_file.closed)
                self.assertFalse(path.exists())

    def test_dp_test_forwards_conversion_format_from_training_config(self) -> None:
        config_path = self.root / "input.json"
        config_path.write_text("{}")
        config = {
            "training": {
                "training_data": {
                    "systems": "data.extxyz",
                    "format": "extxyz",
                    "out_format": "deepmd/lmdb",
                    "rglob_patterns": ["*.extxyz"],
                }
            }
        }
        with (
            patch("deepmd.entrypoints.test.j_loader", return_value=config),
            patch(
                "deepmd.entrypoints.test.update_deepmd_input",
                side_effect=lambda value: value,
            ),
            patch(
                "deepmd.entrypoints.test.process_systems",
                side_effect=RuntimeError("stop after routing"),
            ) as process,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after routing"):
                run_model_test(
                    model="unused.pb",
                    system=None,
                    datafile=None,
                    train_json=str(config_path),
                    numb_test=1,
                    rand_seed=None,
                    shuffle_test=False,
                    detail_file="detail",
                    atomic=False,
                )

        process.assert_called_once_with(
            "data.extxyz",
            patterns=["*.extxyz"],
            root_dir=config_path.parent,
            fmt="extxyz",
            out_fmt="deepmd/lmdb",
        )


if __name__ == "__main__":
    unittest.main()
