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
)
from deepmd.entrypoints.test import test as run_model_test
from deepmd.utils import (
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
        self.root = Path(self.tmpdir.name)
        self.old_cwd = Path.cwd()
        os.chdir(self.root)
        self.source = self.root / "data.extxyz"
        self.source.write_text("1\nProperties=species:S:1:pos:R:3\nH 0 0 0\n")
        _FakeMultiSystems.write_count = 0
        _FakeMultiSystems.load_calls = []
        _FakeMultiSystems.to_calls = []
        data_system._DPDATA_CONVERSION_CACHE.clear()
        data_system._DPDATA_SOURCE_MTIME_CACHE.clear()
        self.fake_dpdata = types.SimpleNamespace(
            MultiSystems=_FakeMultiSystems,
            LabeledSystem=_FakeLabeledSystem,
        )

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)
        self.tmpdir.cleanup()
        data_system._DPDATA_CONVERSION_CACHE.clear()
        data_system._DPDATA_SOURCE_MTIME_CACHE.clear()

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
        self.source.write_text(extxyz.replace("energy=0.0", "energy=1.0"))
        systems_again = process_systems(str(self.source), fmt="extxyz")

        self.assertEqual(len(systems), 1)
        self.assertEqual(systems_again, systems)
        self.assertTrue(is_lmdb(systems[0]))
        data = LmdbDataSystem(systems[0], ["H"], batch_size=1)
        batch = data.get_batch()
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

        data_system._remove_path(link)

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

    def test_requirements_can_be_registered_after_adapter_construction(self) -> None:
        """PBC probing must not freeze the model's label contract."""
        lmdb_path = self.root / "requirements.lmdb"
        _write_minimal_lmdb(str(lmdb_path))
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size=[1])

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

        batch = data.get_batch()

        self.assertEqual(batch["coord"].shape, (2, 6))
        self.assertEqual(batch["type"].shape, (2, 2))
        self.assertIn(-1, batch["type"])

    def test_periodic_and_nonperiodic_frames_are_separate_views(self) -> None:
        lmdb_path = self.root / "mixed-pbc.lmdb"
        _write_mixed_pbc_lmdb(str(lmdb_path))
        data = LmdbDataSystem(str(lmdb_path), ["H"], batch_size=2, seed=0)

        self.assertEqual(len(data.data_systems), 2)
        self.assertEqual({view.pbc for view in data.data_systems}, {False, True})
        batches = [data.get_batch(), data.get_batch()]
        self.assertEqual({float(batch["find_box"]) for batch in batches}, {0.0, 1.0})

    def test_multiple_conversion_inputs_fail_before_dpdata_io(self) -> None:
        second = self.root / "second.extxyz"
        second.write_text(self.source.read_text())

        with patch.dict(sys.modules, {"dpdata": self.fake_dpdata}):
            with self.assertRaisesRegex(ValueError, "exactly one resolved input"):
                process_systems(
                    [str(self.source), str(second)],
                    fmt="extxyz",
                )

        self.assertEqual(_FakeMultiSystems.load_calls, [])
        self.assertEqual(_FakeMultiSystems.write_count, 0)

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
                    "hostname": data_system.socket.gethostname(),
                    "pid": os.getpid(),
                    "process_start": data_system._process_start_time(os.getpid()),
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
                data_system.time, "sleep", release_after_long_valid_conversion
            ),
            patch.object(
                data_system, "_is_conversion_current", return_value=True
            ) as is_current,
        ):
            self.assertTrue(
                data_system._wait_for_conversion(self.source, output, lock_path)
            )

        self.assertEqual(waits, 301)
        is_current.assert_called_once_with(self.source, output, force_source_scan=True)

    def test_dead_conversion_owner_is_recovered(self) -> None:
        lock_path = self.root / "dead.lock"
        lock_path.write_text(
            json.dumps(
                {
                    "hostname": data_system.socket.gethostname(),
                    "pid": 2147483647,
                    "process_start": "missing",
                }
            )
        )

        self.assertTrue(data_system._recover_stale_conversion_lock(lock_path))
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

        with patch.object(data_system.os, "replace", fail_publication):
            with self.assertRaisesRegex(OSError, "publish failed"):
                data_system._publish_conversion_output(staged, output)

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
            data_system.importlib.metadata, "version", side_effect=["1.1.0", "1.2.0"]
        ):
            first = data_system._conversion_cache_path(
                self.source, "extxyz", "deepmd/lmdb"
            )
            second = data_system._conversion_cache_path(
                self.source, "extxyz", "deepmd/lmdb"
            )

        self.assertNotEqual(first, second)

    def test_directory_freshness_scan_is_reused_within_one_routing_pass(self) -> None:
        source_dir = self.root / "source-dir"
        source_dir.mkdir()
        (source_dir / "frame.xyz").write_text("frame")
        output = self.root / ".deepmd_dpdata_cache" / "cached.lmdb"
        output.parent.mkdir()

        first = data_system._source_mtime(source_dir, output)
        with patch.object(
            Path, "rglob", side_effect=AssertionError("unexpected rescan")
        ):
            second = data_system._source_mtime(source_dir, output)

        self.assertEqual(first, second)

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
            str(self.source.resolve()),
            patterns=["*.extxyz"],
            fmt="extxyz",
            out_fmt="deepmd/lmdb",
        )


if __name__ == "__main__":
    unittest.main()
