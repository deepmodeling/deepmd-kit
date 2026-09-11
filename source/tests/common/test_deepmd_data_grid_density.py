# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for the frame-major grid/density branches in DeepmdData.

These keys are loaded verbatim (no reshaping to natoms) and must carry a
leading frame dimension matching the set's frame count, so that shuffling
never pairs a structure with another frame's grid or density label.
"""

import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.utils.data import (
    DeepmdData,
)

NATOMS = 3
NGRID = 5
NFRAMES = 4


class TestDeepmdDataGridDensity(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.set_dir = self.root / "set.000"
        self.set_dir.mkdir()
        rng = np.random.default_rng(42)

        np.savetxt(self.root / "type.raw", np.zeros(NATOMS, dtype=np.int32), fmt="%d")
        np.save(
            self.set_dir / "coord.npy",
            rng.random((NFRAMES, NATOMS * 3), dtype=np.float32),
        )
        np.save(
            self.set_dir / "box.npy",
            np.eye(3, dtype=np.float32).reshape(1, 9).repeat(NFRAMES, axis=0),
        )
        np.save(
            self.set_dir / "type.npy",
            np.zeros((NFRAMES, NATOMS), dtype=np.int32),
        )
        self.set_data("grid", rng.random((NFRAMES, NGRID, 3), dtype=np.float32))
        self.set_data("density", rng.random((NFRAMES, NGRID, 1), dtype=np.float32))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def set_data(self, key: str, arr: np.ndarray | None) -> None:
        path = self.set_dir / f"{key}.npy"
        if arr is None:
            path.unlink(missing_ok=True)
        else:
            np.save(path, arr)

    def build_data(self, must: bool = True) -> DeepmdData:
        data = DeepmdData(str(self.root))
        data.add("coord", 3, atomic=True, must=True)
        data.add("box", 9, atomic=False, must=True)
        data.add("type", 1, atomic=True, must=True)
        data.add("grid", 3, atomic=True, must=must, high_prec=True)
        data.add("density", 1, atomic=True, must=must, high_prec=True)
        return data

    def test_load_set_frame_major(self) -> None:
        data = self.build_data()
        loaded = data._load_set(self.set_dir)
        self.assertEqual(loaded["find_grid"], 1.0)
        self.assertEqual(loaded["find_density"], 1.0)
        self.assertEqual(loaded["grid"].shape, (NFRAMES, NGRID, 3))
        self.assertEqual(loaded["density"].shape, (NFRAMES, NGRID, 1))

    def test_single_frame(self) -> None:
        data = self.build_data()
        frame = data.get_single_frame(0, 0)
        self.assertEqual(frame["find_grid"], 1.0)
        self.assertEqual(frame["find_density"], 1.0)
        self.assertEqual(frame["grid"].shape, (NGRID, 3))
        self.assertEqual(frame["density"].shape, (NGRID, 1))

    def test_frame_count_mismatch_load_set(self) -> None:
        self.set_data("grid", np.zeros((NFRAMES - 1, NGRID, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "frame count"):
            self.build_data()._load_set(self.set_dir)

    def test_frame_count_mismatch_single_frame(self) -> None:
        self.set_data("density", np.zeros((NFRAMES - 1, NGRID, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "frame count"):
            self.build_data().get_single_frame(0, 0)

    def test_missing_frame_dimension_load_set(self) -> None:
        # stored without a leading frame dimension: rejected rather than
        # silently misaligned with the frames
        self.set_data("grid", np.zeros((NGRID, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, r"frame count|leading frame"):
            self.build_data()._load_set(self.set_dir)

    def test_missing_frame_dimension_single_frame(self) -> None:
        self.set_data("density", np.zeros((NGRID, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, r"frame count|leading frame"):
            self.build_data().get_single_frame(0, 0)

    def test_missing_file_optional(self) -> None:
        # optional labels absent: the loader must not fail and must signal
        # find=0 so that downstream code can skip the residual
        self.set_data("density", None)
        data = self.build_data(must=False)
        loaded = data._load_set(self.set_dir)
        self.assertEqual(loaded["find_density"], 0.0)
        frame = data.get_single_frame(0, 0)
        self.assertEqual(frame["find_density"], 0.0)


if __name__ == "__main__":
    unittest.main()
