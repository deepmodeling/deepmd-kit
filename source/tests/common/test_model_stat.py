# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for backend-agnostic statistics sampling helpers."""

import unittest

import numpy as np

from deepmd.utils.model_stat import (
    make_stat_input,
)


class _FakeTypePath:
    """Fake path object that returns in-memory atom types."""

    def __init__(self, real_types: np.ndarray) -> None:
        self.real_types = real_types

    def load_numpy(self) -> np.ndarray:
        """Return the stored atom-type array."""
        return self.real_types


class _FakeSetDir:
    """Fake set directory exposing ``real_atom_types.npy``."""

    def __init__(self, real_types: np.ndarray) -> None:
        self.real_types = real_types

    def __truediv__(self, name: str) -> _FakeTypePath:
        """Return a fake path for the atom-type file."""
        assert name == "real_atom_types.npy"
        return _FakeTypePath(self.real_types)


class _FakeMixedDataSystem:
    """Minimal mixed-type data system for stat sampling tests."""

    mixed_type = True
    enforce_type_map = False
    natoms = 2
    dirs: list[_FakeSetDir]
    prefix_sum: list[int]

    def __init__(self) -> None:
        self.dirs = [_FakeSetDir(np.array([[0, -1], [1, -1]], dtype=np.int32))]
        self.prefix_sum = [2]

    def get_ntypes(self) -> int:
        """Return the number of real atom types."""
        return 2

    def get_single_frame(self, index: int, num_worker: int = 1) -> dict:
        """Return the representative frame containing the missing type."""
        assert index == 1
        return {
            "coord": np.zeros((6,), dtype=np.float32),
            "type": np.array([1, -1], dtype=np.int32),
            "atype": np.array([1, -1], dtype=np.int32),
            "box": np.eye(3, dtype=np.float32).reshape(-1),
            "real_natoms_vec": np.array([2, 2, 0, 1], dtype=np.int32),
            "find_energy": np.float32(1.0),
            "energy": np.array([1.0], dtype=np.float64),
            "fid": index,
        }


class _FakeMixedData:
    """Minimal multi-system data wrapper for ``make_stat_input``."""

    mixed_systems = False
    natoms_vec: list[np.ndarray]
    default_mesh: list[np.ndarray]

    def __init__(self) -> None:
        self.data_systems = [_FakeMixedDataSystem()]
        self.natoms_vec = [np.array([2, 2, 1, 0], dtype=np.int32)]
        self.default_mesh = [np.array([], dtype=np.int32)]

    def get_nsystems(self) -> int:
        """Return the number of systems."""
        return 1

    def get_batch(self, sys_idx: int | None = None) -> dict:
        """Return the initially sampled batch that misses one type."""
        assert sys_idx == 0
        return {
            "coord": np.zeros((1, 6), dtype=np.float32),
            "type": np.array([[0, -1]], dtype=np.int32),
            "box": np.eye(3, dtype=np.float32).reshape(1, 9),
            "real_natoms_vec": np.array([[2, 2, 1, 0]], dtype=np.int32),
            "natoms_vec": np.array([2, 2, 1, 0], dtype=np.int32),
            "default_mesh": np.array([], dtype=np.int32),
            "find_energy": np.float32(1.0),
            "energy": np.array([[0.0]], dtype=np.float64),
        }


class _FakeFixedDataSystem:
    """One-frame fixed-composition system used by mixed-batch tests."""

    mixed_type = False

    def __init__(self, atom_type: int) -> None:
        self.atom_type = atom_type

    def get_single_frame(self, index: int, num_worker: int = 1) -> dict:
        """Return the only frame in this fixed-composition system."""
        assert index == 0
        return {
            "coord": np.full((6,), self.atom_type, dtype=np.float32),
            "type": np.full((2,), self.atom_type, dtype=np.int32),
            "atype": np.full((2,), self.atom_type, dtype=np.int32),
            "box": np.eye(3, dtype=np.float32).reshape(-1),
            "find_energy": np.float32(1.0),
            "energy": np.array([float(self.atom_type)], dtype=np.float64),
            "fid": index,
        }


class _FakeMixedSystemsData:
    """Mixed-batch wrapper whose random sample always misses the rare system."""

    mixed_systems = True

    def __init__(self) -> None:
        self.data_systems = [_FakeFixedDataSystem(0), _FakeFixedDataSystem(1)]
        self.natoms_vec = [
            np.array([2, 2, 2, 0], dtype=np.int32),
            np.array([2, 2, 0, 2], dtype=np.int32),
        ]
        self.default_mesh = [
            np.array([], dtype=np.int32),
            np.array([], dtype=np.int32),
        ]

    def get_nsystems(self) -> int:
        """Return the number of underlying systems."""
        return 2

    def get_ntypes(self) -> int:
        """Return the global number of atom types."""
        return 2

    def get_batch(self, sys_idx: int | None = None) -> dict:
        """Mimic a low-probability rare system by always sampling type zero."""
        raw_batch = self.data_systems[0].get_single_frame(0)
        raw_batch = {
            key: value.reshape((1, *value.shape))
            if isinstance(value, np.ndarray) and value.ndim >= 1
            else value
            for key, value in raw_batch.items()
            if key not in {"atype", "fid"}
        }
        raw_batch["natoms_vec"] = self.natoms_vec[0]
        raw_batch["default_mesh"] = self.default_mesh[0]
        return self._merge_batch_data([raw_batch])

    def _merge_batch_data(self, batch_data: list[dict]) -> dict:
        """Merge same-sized frames using the production mixed-batch schema."""
        return {
            "natoms_vec": np.array([2, 2, 2, 0], dtype=np.int32),
            "real_natoms_vec": np.vstack([batch["natoms_vec"] for batch in batch_data]),
            "type": np.concatenate([batch["type"] for batch in batch_data]),
            "default_mesh": np.array([], dtype=np.int32),
            "coord": np.concatenate([batch["coord"] for batch in batch_data]),
            "box": np.concatenate([batch["box"] for batch in batch_data]),
            "find_energy": batch_data[0]["find_energy"],
            "energy": np.concatenate([batch["energy"] for batch in batch_data]),
        }


class TestModelStatSamplingCoverage(unittest.TestCase):
    """Mixed-type make_stat_input should cover types beyond initial batches."""

    def test_make_stat_input_appends_missing_mixed_type_frame(self) -> None:
        """Append a representative frame when the first batch misses a type."""
        data = _FakeMixedData()
        ordinary_batch = data.get_batch(sys_idx=0)
        self.assertIn("type", ordinary_batch)
        self.assertNotIn("atype", ordinary_batch)
        self.assertNotIn("fid", ordinary_batch)

        raw_frame = data.data_systems[0].get_single_frame(1)
        self.assertIn("type", raw_frame)
        self.assertIn("atype", raw_frame)
        self.assertIn("fid", raw_frame)

        sampled = make_stat_input(data, nbatches=1)

        self.assertEqual(len(sampled), 1)
        counts = sampled[0]["real_natoms_vec"][:, 2:].sum(axis=0)
        self.assertTrue(np.all(counts > 0))
        self.assertEqual(sampled[0]["atype"].shape, (2, 2))
        self.assertEqual(sampled[0]["energy"].shape[0], 2)
        self.assertNotIn("fid", sampled[0])

    def test_make_stat_input_covers_rare_type_in_mixed_systems(self) -> None:
        """Cover a low-probability system in ``batch_size: mixed`` sampling."""
        sampled = make_stat_input(_FakeMixedSystemsData(), nbatches=1)

        self.assertEqual(len(sampled), 2)
        for system in sampled:
            counts = system["real_natoms_vec"][:, 2:].sum(axis=0)
            self.assertTrue(np.all(counts > 0))
            self.assertEqual(system["atype"].shape, (2, 2))
            self.assertEqual(system["energy"].shape, (2, 1))
            self.assertNotIn("fid", system)


if __name__ == "__main__":
    unittest.main()
