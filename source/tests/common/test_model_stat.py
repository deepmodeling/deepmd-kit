# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for backend-agnostic statistics sampling helpers."""

import unittest

import numpy as np

from deepmd.utils.model_stat import (
    make_stat_input,
)


class _FakeTypePath:
    def __init__(self, real_types: np.ndarray) -> None:
        self.real_types = real_types

    def load_numpy(self) -> np.ndarray:
        return self.real_types


class _FakeSetDir:
    def __init__(self, real_types: np.ndarray) -> None:
        self.real_types = real_types

    def __truediv__(self, name: str) -> _FakeTypePath:
        assert name == "real_atom_types.npy"
        return _FakeTypePath(self.real_types)


class _FakeMixedDataSystem:
    mixed_type = True
    enforce_type_map = False
    natoms = 2
    dirs: list[_FakeSetDir]
    prefix_sum: list[int]

    def __init__(self) -> None:
        self.dirs = [_FakeSetDir(np.array([[0, -1], [1, -1]], dtype=np.int32))]
        self.prefix_sum = [2]

    def get_ntypes(self) -> int:
        return 2

    def get_single_frame(self, index: int, num_worker: int = 1) -> dict:
        assert index == 1
        return {
            "coord": np.zeros((6,), dtype=np.float32),
            "type": np.array([1, -1], dtype=np.int32),
            "atype": np.array([1, -1], dtype=np.int32),
            "box": np.eye(3, dtype=np.float32).reshape(-1),
            "real_natoms_vec": np.array([2, 2, 0, 1], dtype=np.int32),
            "find_energy": np.float32(1.0),
            "energy": np.array([1.0], dtype=np.float64),
        }


class _FakeMixedData:
    mixed_systems = False
    natoms_vec: list[np.ndarray]
    default_mesh: list[np.ndarray]

    def __init__(self) -> None:
        self.data_systems = [_FakeMixedDataSystem()]
        self.natoms_vec = [np.array([2, 2, 1, 0], dtype=np.int32)]
        self.default_mesh = [np.array([], dtype=np.int32)]

    def get_nsystems(self) -> int:
        return 1

    def get_batch(self, sys_idx: int | None = None) -> dict:
        assert sys_idx == 0
        return {
            "coord": np.zeros((1, 6), dtype=np.float32),
            "type": np.array([[0, -1]], dtype=np.int32),
            "atype": np.array([[0, -1]], dtype=np.int32),
            "box": np.eye(3, dtype=np.float32).reshape(1, 9),
            "real_natoms_vec": np.array([[2, 2, 1, 0]], dtype=np.int32),
            "natoms_vec": np.array([2, 2, 1, 0], dtype=np.int32),
            "default_mesh": np.array([], dtype=np.int32),
            "find_energy": np.float32(1.0),
            "energy": np.array([[0.0]], dtype=np.float64),
        }


class TestModelStatSamplingCoverage(unittest.TestCase):
    """Mixed-type make_stat_input should cover types beyond initial batches."""

    def test_make_stat_input_appends_missing_mixed_type_frame(self) -> None:
        sampled = make_stat_input(_FakeMixedData(), nbatches=1)

        self.assertEqual(len(sampled), 1)
        counts = sampled[0]["real_natoms_vec"][:, 2:].sum(axis=0)
        self.assertTrue(np.all(counts > 0))
        self.assertEqual(sampled[0]["energy"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
