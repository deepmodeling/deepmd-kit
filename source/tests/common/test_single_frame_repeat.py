# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest

from deepmd.utils.data import (
    DeepmdData,
)


@pytest.mark.parametrize("present", [False, True])
@pytest.mark.parametrize("nframes", [1, 2])
@pytest.mark.parametrize("backend", ["torch", "paddle"])
def test_single_frame_repeat_matches_batch(tmp_path, present, nframes, backend):
    """Single-frame and batch loaders repeat atomic labels identically."""
    np.savetxt(tmp_path / "type.raw", [1, 0], fmt="%d")
    set_path = tmp_path / "set.000"
    set_path.mkdir()
    np.save(set_path / "coord.npy", np.zeros((nframes, 6)))
    np.save(set_path / "box.npy", np.tile(np.eye(3).reshape(1, 9), (nframes, 1)))
    if present:
        np.save(set_path / "atom_pref.npy", np.tile([[2.0, 3.0]], (nframes, 1)))
    data = DeepmdData(str(tmp_path))
    data.add("atom_pref", 1, atomic=True, repeat=3, default=1.0)
    expected = data._load_set(set_path)["atom_pref"][0]
    frame = getattr(data, f"get_item_{backend}")(0)
    np.testing.assert_array_equal(frame["atom_pref"].reshape(-1), expected)
    assert frame["find_atom_pref"] == float(present)
