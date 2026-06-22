# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for data loading, dpdata integration, and attach_labels."""

import numpy as np
import pytest

from dpa_adapt.data.convert import (
    _key_from_head,
    attach_labels,
)
from dpa_adapt.data.errors import (
    DPADataError,
)
from dpa_adapt.data.loader import (
    load_data,
)
from dpa_adapt.finetuner import (
    _load_labels,
    _load_npy_system,
)


def _make_system(tmp_path, name="sys", set_indices=(0,), n_atoms=2, n_frames=3):
    """Create a minimal deepmd/npy system dir and load it via dpdata."""
    root = tmp_path / name
    root.mkdir()
    (root / "type.raw").write_text("\n".join(str(i % 2) for i in range(n_atoms)) + "\n")
    (root / "type_map.raw").write_text("H\nO\n")
    for idx in set_indices:
        sd = root / f"set.{idx:03d}"
        sd.mkdir()
        np.save(sd / "coord.npy", np.random.rand(n_frames, n_atoms * 3))
        np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (n_frames, 1)))
        np.save(sd / "energy.npy", np.random.rand(n_frames))
    return load_data(str(root))[0]


# ---------------------------------------------------------------------------
# set.* sort ordering
# ---------------------------------------------------------------------------


class TestSetDirSorting:
    """dpdata preserves set.* numeric ordering during loading."""

    def test_sorted_order_in_load_labels(self, tmp_path):
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n")
        (root / "type_map.raw").write_text("H\n")

        markers = {0: 0.0, 1: 1.0, 10: 10.0, 100: 100.0}
        for idx, val in markers.items():
            sd = root / f"set.{idx:03d}"
            sd.mkdir()
            np.save(sd / "coord.npy", np.zeros((1, 3)))
            np.save(sd / "box.npy", np.eye(3).reshape(1, 9))
            np.save(sd / "energy.npy", np.array([val]))

        system = load_data(str(root))[0]
        labels = _load_labels([system], "energy")
        assert list(labels) == [0.0, 1.0, 10.0, 100.0], (
            f"Expected [0, 1, 10, 100], got {list(labels)}"
        )

    def test_sorted_order_in_load_npy_system(self, tmp_path):
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n")
        (root / "type_map.raw").write_text("H\n")

        for idx in [0, 1, 10, 100]:
            sd = root / f"set.{idx:03d}"
            sd.mkdir()
            np.save(sd / "coord.npy", np.full((1, 3), float(idx)))
            np.save(sd / "box.npy", np.eye(3).reshape(1, 9))

        system = load_data(str(root))[0]
        coords, _, _ = _load_npy_system(system)
        frame_values = coords[:, 0].tolist()  # first atom, first coord axis
        assert frame_values == [0.0, 1.0, 10.0, 100.0], (
            f"Expected [0, 1, 10, 100], got {frame_values}"
        )


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_valid_system_returns_dpdata_system(self, tmp_path):
        system = _make_system(tmp_path)
        result = load_data(system)
        assert len(result) == 1
        assert result[0] is system  # passthrough, no copy

    def test_path_loads_dpdata_system(self, tmp_path):
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n1\n")
        (root / "type_map.raw").write_text("H\nO\n")
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((2, 6)))
        np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (2, 1)))

        result = load_data(str(root))
        assert len(result) == 1
        import dpdata

        assert isinstance(result[0], dpdata.System)

    def test_list_of_systems(self, tmp_path):
        s1 = _make_system(tmp_path, "a")
        s2 = _make_system(tmp_path, "b")
        result = load_data([s1, s2])
        assert len(result) == 2

    def test_mixed_list_paths_and_objects(self, tmp_path):
        s1 = _make_system(tmp_path, "a")
        root = tmp_path / "b"
        root.mkdir()
        (root / "type.raw").write_text("0\n")
        (root / "type_map.raw").write_text("H\n")
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((2, 3)))
        np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (2, 1)))

        result = load_data([s1, str(root)])
        assert len(result) == 2

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(DPADataError, match="does not exist"):
            load_data(str(tmp_path / "ghost"))

    def test_passthrough_no_copy(self, tmp_path):
        s = _make_system(tmp_path)
        result = load_data(s)
        assert result[0] is s


class TestGlob:
    def test_mixed_files_and_dirs_fails_fast(self, tmp_path):
        """Glob with deepmd/npy fmt must reject non-directory matches."""
        # Create a valid deepmd/npy directory
        _make_system(tmp_path, "sys")
        # Create a non-directory file
        (tmp_path / "file.xyz").write_text("dummy")

        with pytest.raises(DPADataError, match="non-directory paths"):
            load_data(str(tmp_path / "*"))

    def test_explicit_fmt_bypasses_precheck(self, tmp_path):
        """With an explicit non-deepmd/npy fmt the pre-check is skipped."""
        (tmp_path / "file.xyz").write_text("6\n\nH 0 0 0\nO 1 1 1\n")

        with pytest.raises(DPADataError, match="Failed to load"):
            # Not deepmd/npy → skips the directory pre-check, tries dpdata
            load_data(str(tmp_path / "file.xyz"), fmt="xyz")


# ---------------------------------------------------------------------------
# attach_labels — _key_from_head
# ---------------------------------------------------------------------------


class TestKeyFromHead:
    def test_string_head(self):
        assert _key_from_head("energy") == "energy"
        assert _key_from_head("bandgap") == "bandgap"

    def test_dict_with_property_name(self):
        assert (
            _key_from_head(
                {"type": "property", "property_name": "bandgap", "task_dim": 1}
            )
            == "bandgap"
        )
        assert _key_from_head({"property_name": "humo"}) == "humo"

    def test_dict_known_types(self):
        assert _key_from_head({"type": "dos", "numb_dos": 250}) == "dos"
        assert _key_from_head({"type": "dipole"}) == "dipole"
        assert _key_from_head({"type": "polar"}) == "polar"

    def test_dict_unknown_type_raises_with_supported_list(self):
        with pytest.raises(ValueError, match="Unknown dict head type 'forces'"):
            _key_from_head({"type": "forces"})
        with pytest.raises(ValueError, match="dos.*dipole|dipole.*dos"):
            _key_from_head({"type": "unknown_xyz"})

    def test_dict_property_type_without_property_name_raises(self):
        with pytest.raises(ValueError, match="property_name"):
            _key_from_head({"type": "property", "task_dim": 1})

    def test_dict_missing_both_keys_raises(self):
        with pytest.raises(ValueError, match="property_name.*type"):
            _key_from_head({"task_dim": 1})

    def test_non_str_non_dict_raises(self):
        with pytest.raises(TypeError, match="str or dict"):
            _key_from_head(42)


def _make_system_path(tmp_path, name="sys", set_indices=(0,), n_atoms=2, n_frames=3):
    """Create a minimal deepmd/npy system directory on disk (no dpdata loading).

    Returns the **Path** to the system root.
    """
    root = tmp_path / name
    root.mkdir()
    (root / "type.raw").write_text("\n".join(str(i % 2) for i in range(n_atoms)) + "\n")
    (root / "type_map.raw").write_text("H\nO\n")
    for idx in set_indices:
        sd = root / f"set.{idx:03d}"
        sd.mkdir()
        np.save(sd / "coord.npy", np.random.rand(n_frames, n_atoms * 3))
        np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (n_frames, 1)))
        np.save(sd / "energy.npy", np.random.rand(n_frames))
    return root


class TestAttachLabels:
    """Path-based attach_labels: single and multi-system."""

    # ── single-system ────────────────────────────────────────────────────

    def test_string_head_writes_npy(self, tmp_path):
        sys_path = _make_system_path(tmp_path, name="sys", n_frames=3)
        attach_labels(sys_path, head="bandgap", values=np.array([1.0, 2.0, 3.0]))
        written = np.load(sys_path / "set.000" / "bandgap.npy")
        np.testing.assert_array_equal(written, [1.0, 2.0, 3.0])

    def test_dict_head_property_name(self, tmp_path):
        sys_path = _make_system_path(tmp_path, name="sys", n_frames=3)
        values = np.array([[1.0], [2.0], [3.0]])
        attach_labels(
            sys_path,
            head={"type": "property", "property_name": "gap", "task_dim": 1},
            values=values,
        )
        written = np.load(sys_path / "set.000" / "gap.npy")
        np.testing.assert_array_equal(written, values)

    def test_2d_values_written_correctly(self, tmp_path):
        sys_path = _make_system_path(tmp_path, name="sys", n_frames=3)
        values = np.arange(3 * 250, dtype=float).reshape(3, 250)
        attach_labels(sys_path, head={"type": "dos", "numb_dos": 250}, values=values)
        written = np.load(sys_path / "set.000" / "dos.npy")
        assert written.shape == (3, 250)
        np.testing.assert_array_equal(written, values)

    def test_frame_count_mismatch_raises(self, tmp_path):
        sys_path = _make_system_path(tmp_path, name="sys", n_frames=3)
        with pytest.raises(ValueError, match="frames"):
            attach_labels(sys_path, head="energy", values=np.array([1.0, 2.0]))

    def test_same_key_overwrites(self, tmp_path):
        sys_path = _make_system_path(tmp_path, name="sys", n_frames=3)
        attach_labels(sys_path, head="energy", values=np.array([1.0, 2.0, 3.0]))
        attach_labels(sys_path, head="energy", values=np.array([9.0, 8.0, 7.0]))
        written = np.load(sys_path / "set.000" / "energy.npy")
        np.testing.assert_array_equal(written, [9.0, 8.0, 7.0])

    def test_different_keys_are_additive(self, tmp_path):
        sys_path = _make_system_path(tmp_path, name="sys", n_frames=3)
        attach_labels(sys_path, head="energy", values=np.array([1.0, 2.0, 3.0]))
        attach_labels(sys_path, head="bandgap", values=np.array([4.0, 5.0, 6.0]))
        e_written = np.load(sys_path / "set.000" / "energy.npy")
        b_written = np.load(sys_path / "set.000" / "bandgap.npy")
        np.testing.assert_array_equal(e_written, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(b_written, [4.0, 5.0, 6.0])

    def test_multi_set_not_implemented(self, tmp_path):
        sys_path = _make_system_path(
            tmp_path, name="sys", set_indices=(0, 1), n_frames=3
        )
        with pytest.raises(NotImplementedError, match="Multiple set"):
            attach_labels(sys_path, head="energy", values=np.array([1.0, 2.0, 3.0]))

    def test_no_set_dir_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "type.raw").write_text("0\n")
        with pytest.raises(ValueError, match="No set"):
            attach_labels(empty, head="energy", values=np.array([1.0]))

    def test_path_is_file_raises(self, tmp_path):
        f = tmp_path / "not_a_dir"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="not a directory"):
            attach_labels(f, head="energy", values=np.array([1.0]))

    def test_coord_npy_missing_raises(self, tmp_path):
        sys_path = _make_system_path(tmp_path, name="sys", n_frames=3)
        (sys_path / "set.000" / "coord.npy").unlink()
        with pytest.raises(ValueError, match="coord.npy not found"):
            attach_labels(sys_path, head="energy", values=np.array([1.0, 2.0, 3.0]))

    # ── multi-system ─────────────────────────────────────────────────────

    def test_multi_system_all_written(self, tmp_path):
        parent = tmp_path / "multi"
        parent.mkdir()
        for i in range(3):
            _make_system_path(parent, name=f"sys_{i:04d}", n_frames=2)
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        attach_labels(parent, head="bandgap", values=values)
        for i in range(3):
            written = np.load(parent / f"sys_{i:04d}" / "set.000" / "bandgap.npy")
            np.testing.assert_array_equal(written, values[i])

    def test_multi_system_values_mismatch_raises(self, tmp_path):
        parent = tmp_path / "multi"
        parent.mkdir()
        _make_system_path(parent, name="sys_0000", n_frames=2)
        _make_system_path(parent, name="sys_0001", n_frames=2)
        with pytest.raises(ValueError, match="entries along the first axis"):
            attach_labels(
                parent,
                head="bandgap",
                values=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            )

    def test_multi_system_no_subdirs_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No set.* directories or system"):
            attach_labels(empty, head="energy", values=np.array([1.0]))

    def test_multi_system_hidden_dirs_ignored(self, tmp_path):
        parent = tmp_path / "multi"
        parent.mkdir()
        _make_system_path(parent, name="sys_0000", n_frames=2)
        (parent / ".hidden").mkdir()
        values = np.array([[1.0, 2.0]])
        attach_labels(parent, head="bandgap", values=values)
        written = np.load(parent / "sys_0000" / "set.000" / "bandgap.npy")
        np.testing.assert_array_equal(written, [1.0, 2.0])


# ---------------------------------------------------------------------------
# _load_labels — custom label key fallback
# ---------------------------------------------------------------------------


class TestLoadLabelsCustomKey:
    """_load_labels falls back to set.*/key.npy when key not in dpdata's store."""

    def test_custom_label_key_loaded_from_npy(self, tmp_path):
        """target_key="property" loads set.000/property.npy directly."""
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n1\n")
        (root / "type_map.raw").write_text("H\nO\n")
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((3, 6)))
        np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (3, 1)))
        # Custom label — NOT loaded by dpdata into system.data
        np.save(sd / "property.npy", np.array([10.0, 20.0, 30.0]))

        [system] = load_data(str(root))
        assert "property" not in system.data

        labels = _load_labels([system], "property")
        np.testing.assert_array_equal(labels, [10.0, 20.0, 30.0])

    def test_custom_key_not_found_raises_clear_error(self, tmp_path):
        """When neither dpdata nor set.*/key.npy has the key, error lists both."""
        root = tmp_path / "sys"
        root.mkdir()
        (root / "type.raw").write_text("0\n")
        (root / "type_map.raw").write_text("H\n")
        sd = root / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((2, 3)))
        np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (2, 1)))

        [system] = load_data(str(root))

        with pytest.raises(DPADataError, match="nonexistent"):
            _load_labels([system], "nonexistent")
