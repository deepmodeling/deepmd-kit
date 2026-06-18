# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for type_map validation and local→global atom-type remapping."""

import sys
from unittest.mock import (
    MagicMock,
)

import numpy as np
import pytest

_mock_torch = MagicMock()
_mock_torch.Tensor = type("Tensor", (), {})
sys.modules.setdefault("torch", _mock_torch)

from dpa_adapt.data.errors import DPADataError
from dpa_adapt.data.loader import load_data
from dpa_adapt.finetuner import (
    DPAFineTuner,
    _read_data_type_map,
)

PERIODIC_PREFIX_9 = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F"]


def _make_system(tmp_path, name, type_indices, type_map):
    """Create a minimal deepmd/npy system and load it via dpdata."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    n_atoms = len(type_indices)
    (root / "type.raw").write_text("\n".join(str(i) for i in type_indices) + "\n")
    (root / "type_map.raw").write_text("\n".join(type_map) + "\n")
    sd = root / "set.000"
    sd.mkdir(exist_ok=True)
    np.save(sd / "coord.npy", np.zeros((1, n_atoms * 3)))
    np.save(sd / "box.npy", np.eye(3).reshape(1, 9))
    return load_data(str(root))[0]


# ---------------------------------------------------------------------------
# _validate_type_map
# ---------------------------------------------------------------------------


class TestValidateTypeMapSubset:
    def test_non_prefix_subset_accepted(self, tmp_path):
        sys = _make_system(tmp_path, "qm9", [0, 1, 2], ["H", "C", "N"])
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        ft._validate_type_map([], [sys])
        ft._validate_type_map(["H", "C", "N", "O", "F"], [sys])

    def test_empty_checkpoint_skips(self, tmp_path):
        sys = _make_system(tmp_path, "sys", [0], ["Xx"])
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = []
        ft._validate_type_map(["Xx"], [sys])

    def test_no_type_map_raw_skips(self, tmp_path):
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n")
        # No type_map.raw → no atom_names
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((1, 3)))
        np.save(sd / "box.npy", np.eye(3).reshape(1, 9))
        sys = load_data(str(root))[0]
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        ft._validate_type_map([], [sys])


class TestValidateTypeMapUnsupported:
    def test_unsupported_in_user_type_map(self, tmp_path):
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        with pytest.raises(DPADataError) as ei:
            ft._validate_type_map(["H", "C", "Xx"], [])
        msg = str(ei.value)
        assert "not supported" in msg
        assert "Xx" in msg
        assert "prefix" not in msg.lower()

    def test_unsupported_in_data_type_map(self, tmp_path):
        sys = _make_system(tmp_path, "sys", [0, 1], ["H", "Xx"])
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        with pytest.raises(DPADataError) as ei:
            ft._validate_type_map([], [sys])
        msg = str(ei.value)
        assert "not supported" in msg
        assert "Xx" in msg
        assert "prefix" not in msg.lower()


# ---------------------------------------------------------------------------
# _remap_atom_types
# ---------------------------------------------------------------------------


class TestRemapAtomTypes:
    def test_remap_via_atom_names(self, tmp_path):
        sys = _make_system(tmp_path, "qm9", [0, 1, 2, 3, 4], ["H", "C", "N", "O", "F"])
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        atom_types = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        out = ft._remap_atom_types(atom_types, sys)
        np.testing.assert_array_equal(out, [0, 5, 6, 7, 8])

    def test_remap_with_arbitrary_order(self, tmp_path):
        sys = _make_system(tmp_path, "sys", [0, 1, 0], ["O", "H"])
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        out = ft._remap_atom_types(np.array([0, 1, 0]), sys)
        np.testing.assert_array_equal(out, [7, 0, 7])

    def test_fallback_to_user_type_map(self, tmp_path):
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n1\n")
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((1, 6)))
        np.save(sd / "box.npy", np.eye(3).reshape(1, 9))
        sys = load_data(str(root))[0]
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        ft.type_map = ["C", "F"]
        out = ft._remap_atom_types(np.array([0, 1]), sys)
        np.testing.assert_array_equal(out, [5, 8])

    def test_no_type_map_in_range_passes_through(self, tmp_path):
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n1\n")
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((1, 6)))
        np.save(sd / "box.npy", np.eye(3).reshape(1, 9))
        sys = load_data(str(root))[0]
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        out = ft._remap_atom_types(np.array([0, 1]), sys)
        np.testing.assert_array_equal(out, [0, 1])

    def test_no_type_map_out_of_range_raises(self, tmp_path):
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n42\n")
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((1, 6)))
        np.save(sd / "box.npy", np.eye(3).reshape(1, 9))
        sys = load_data(str(root))[0]
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        with pytest.raises(DPADataError, match="out of range"):
            ft._remap_atom_types(np.array([0, 42]), sys)

    def test_unsupported_element_in_data_type_map_raises(self, tmp_path):
        sys = _make_system(tmp_path, "sys", [0], ["Xx"])
        ft = DPAFineTuner(pretrained="fake.pt")
        ft._checkpoint_type_map = list(PERIODIC_PREFIX_9)
        with pytest.raises(DPADataError) as ei:
            ft._remap_atom_types(np.array([0]), sys)
        assert "not supported" in str(ei.value)
        assert "Xx" in str(ei.value)


# ---------------------------------------------------------------------------
# _read_data_type_map
# ---------------------------------------------------------------------------


class TestReadDataTypeMap:
    def test_reads_elements(self, tmp_path):
        sys = _make_system(tmp_path, "sys", [0, 1, 2], ["H", "C", "N"])
        assert _read_data_type_map(sys) == ["H", "C", "N"]

    def test_returns_empty_when_missing(self, tmp_path):
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n")
        # No type_map.raw
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((1, 3)))
        np.save(sd / "box.npy", np.eye(3).reshape(1, 9))
        sys = load_data(str(root))[0]
        assert _read_data_type_map(sys) == []

    def test_strips_blank_lines(self, tmp_path):
        sys = _make_system(tmp_path, "sys", [0, 1], ["H", "C"])
        assert _read_data_type_map(sys) == ["H", "C"]
