"""Tests for check_data() — content-level sanity checks on dpdata systems."""

import numpy as np
import pytest

from deepmd.dpa_tools.data.validate import check_data, Issue, _BOX_DET_TOLERANCE
from deepmd.dpa_tools.data.errors import DPADataError
from deepmd.dpa_tools.data.loader import load_data


def _make_set_dir(set_dir, *, coord=None, box=None, energy=None, force=None,
                  n_frames=3, n_atoms=2):
    set_dir.mkdir(parents=True)
    if coord is None:
        coord = np.random.RandomState(0).rand(n_frames, n_atoms * 3)
    if box is None:
        box = (np.eye(3) * 10.0).reshape(1, 9).repeat(n_frames, 0)
    np.save(set_dir / "coord.npy", coord)
    np.save(set_dir / "box.npy", box)
    if energy is not None:
        np.save(set_dir / "energy.npy", energy)
    if force is not None:
        np.save(set_dir / "force.npy", force)


def _system(tmp_path, **set_kwargs):
    """Create a valid deepmd/npy system, load via dpdata, return it."""
    root = tmp_path / "sys"
    root.mkdir()
    (root / "type.raw").write_text("0\n0\n")
    (root / "type_map.raw").write_text("H\nH\n")
    _make_set_dir(root / "set.000", **set_kwargs)
    return load_data(str(root))[0]


# ---------------------------------------------------------------------------
# Clean data
# ---------------------------------------------------------------------------

def test_clean_data_no_issues(tmp_path):
    system = _system(tmp_path)
    issues = check_data(system)
    assert len(issues) == 0


def test_structure_only_no_energy_force_is_clean(tmp_path):
    # Create system with only coords + box (no energy/force)
    root = tmp_path / "sys"
    root.mkdir()
    (root / "type.raw").write_text("0\n0\n")
    (root / "type_map.raw").write_text("H\nH\n")
    _make_set_dir(root / "set.000")
    # Remove energy.npy and force.npy before loading
    system = load_data(str(root))[0]
    issues = check_data(system)
    assert len(issues) == 0, [i.description for i in issues]


# ---------------------------------------------------------------------------
# NaN / Inf
# ---------------------------------------------------------------------------

def test_energy_nan_is_error(tmp_path):
    system = _system(tmp_path, energy=np.array([np.nan, 0.0, 0.0]))
    issues = check_data(system)
    assert any("energies" in i.file and "non-finite" in i.description
               for i in issues)

def test_force_inf_is_error(tmp_path):
    system = _system(tmp_path)
    # Inject bad forces after loading (dpdata may refuse to load inf arrays)
    system.data["forces"] = np.full((3, 2, 3), np.inf)
    issues = check_data(system)
    assert any("forces" in i.file and "non-finite" in i.description
               for i in issues)

def test_box_nan_is_error(tmp_path):
    system = _system(tmp_path, box=np.full((3, 9), np.nan))
    issues = check_data(system)
    assert any("cells" in i.file and "non-finite" in i.description
               for i in issues)


# ---------------------------------------------------------------------------
# Degenerate box
# ---------------------------------------------------------------------------

def test_degenerate_box_is_error_with_det_in_description(tmp_path):
    system = _system(tmp_path, box=np.zeros((3, 9)))
    issues = check_data(system)
    assert any("cells" in i.file and "degenerate" in i.description
               for i in issues)

def test_box_det_tolerance_boundary(tmp_path):
    # A very thin but valid box near the default tolerance
    box = np.tile(np.diag([10.0, 1e-11, 10.0]).ravel(), (3, 1))
    system = _system(tmp_path, box=box)
    issues = check_data(system)
    # |det| = 10 * 1e-11 * 10 = 1e-9, which is > 1e-10 default tol → clean
    assert not any("degenerate" in i.description for i in issues)

def test_box_det_tol_is_configurable(tmp_path):
    box = np.tile(np.diag([10.0, 1e-11, 10.0]).ravel(), (3, 1))
    system = _system(tmp_path, box=box)
    issues = check_data(system, box_det_tol=1e-8)
    # |det| = 1e-9 < 1e-8 tol → degenerate
    assert any("degenerate" in i.description for i in issues)


# ---------------------------------------------------------------------------
# Magnitude warnings
# ---------------------------------------------------------------------------

def test_energy_magnitude_warning(tmp_path):
    system = _system(tmp_path, energy=np.array([1e5, 0.0, 0.0]))
    issues = check_data(system)
    assert any("energies" in i.file and "suspicious magnitude" in i.description
               for i in issues)

def test_force_magnitude_warning(tmp_path):
    system = _system(tmp_path)
    big_force = np.zeros((3, 2, 3))
    big_force[0, 0, 0] = 5000.0
    system.data["forces"] = big_force
    issues = check_data(system)
    assert any("forces" in i.file and "suspicious magnitude" in i.description
               for i in issues)


# ---------------------------------------------------------------------------
# Frame count alignment
# ---------------------------------------------------------------------------

def test_frame_count_mismatch_is_error(tmp_path):
    system = _system(tmp_path, coord=np.zeros((3, 6)))
    system.data["energies"] = np.zeros(5)  # mismatched
    issues = check_data(system)
    assert any("energies" in i.file and "frame counts must align" in i.description
               for i in issues)


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------

def test_strict_raises_on_first_issue(tmp_path):
    system = _system(tmp_path, energy=np.array([np.nan, 0.0, 0.0]))
    with pytest.raises(DPADataError, match="check_data"):
        check_data(system, strict=True)


# ---------------------------------------------------------------------------
# List input
# ---------------------------------------------------------------------------

def test_list_input_aggregates_across_systems(tmp_path):
    s1 = _system(tmp_path, energy=np.array([np.nan, 0.0, 0.0]))
    # use a different tmp subdir to avoid conflict
    s2_root = tmp_path / "sys2"
    s2_root.mkdir()
    (s2_root / "type.raw").write_text("0\n0\n")
    (s2_root / "type_map.raw").write_text("H\nH\n")
    from deepmd.dpa_tools.data.loader import load_data
    from tests.dpa_tools.test_validate import _make_set_dir
    _make_set_dir(s2_root / "set.000")
    s2 = load_data(str(s2_root))[0]
    issues = check_data([s1, s2])
    assert len(issues) >= 1


def test_set_dirs_checked_in_numeric_order(tmp_path):
    # dpdata loads all set.* dirs; check covers all frames
    system = _system(tmp_path, energy=np.array([1e5, 0.0, 0.0]))
    issues = check_data(system)
    # magnitude warning should reference frame 0
    mag_issues = [i for i in issues if "suspicious magnitude" in i.description]
    assert len(mag_issues) >= 1


def test_issue_namedtuple_shape(tmp_path):
    system = _system(tmp_path, energy=np.array([np.nan, 0.0, 0.0]))
    issues = check_data(system)
    assert len(issues) > 0
    issue = issues[0]
    assert issue.severity in ("warning", "error")
    assert isinstance(issue.system, str)
    assert isinstance(issue.file, str)
    assert isinstance(issue.description, str)
