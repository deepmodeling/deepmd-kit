# SPDX-License-Identifier: LGPL-3.0-or-later
"""Preserve periodic-boundary semantics across batch normalization."""

import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel.utils.batch import (
    normalize_batch,
    split_batch,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


class TestNormalizeBatchPBC(unittest.TestCase):
    """Translate the legacy mesh encoding before dropping its metadata."""

    @staticmethod
    def _batch(default_mesh_size: int, box: np.ndarray) -> dict:
        return {
            "coord": np.zeros((1, 2, 3)),
            "type": np.zeros((1, 2), dtype=np.int32),
            "box": box,
            "default_mesh": np.zeros(default_mesh_size, dtype=np.int32),
        }

    def test_nonperiodic_box_is_none(self) -> None:
        for mesh_size in (0, 1):
            with self.subTest(mesh_size=mesh_size):
                box = np.eye(3).reshape(1, 9)
                batch = self._batch(mesh_size, box)

                normalized = normalize_batch(batch)

                self.assertIsNone(normalized["box"])
                self.assertNotIn("default_mesh", normalized)
                self.assertIs(batch["box"], box)

    def test_periodic_box_is_preserved(self) -> None:
        for mesh_size in (6, 7):
            with self.subTest(mesh_size=mesh_size):
                box = np.eye(3).reshape(1, 9)

                normalized = normalize_batch(self._batch(mesh_size, box))

                self.assertIs(normalized["box"], box)

    def test_missing_default_mesh_preserves_box(self) -> None:
        box = np.eye(3).reshape(1, 9)
        batch = self._batch(0, box)
        del batch["default_mesh"]

        normalized = normalize_batch(batch)

        self.assertIs(normalized["box"], box)

    def test_nonperiodic_mesh_without_box_does_not_add_box(self) -> None:
        batch = self._batch(0, np.eye(3).reshape(1, 9))
        del batch["box"]

        normalized = normalize_batch(batch)

        self.assertNotIn("box", normalized)


class TestDeepmdDataSystemNopbc(unittest.TestCase):
    """A valid nopbc NPY system must reach the model without a box."""

    def test_zero_box_placeholder_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            system = Path(tmpdir)
            set_dir = system / "set.000"
            set_dir.mkdir()
            (system / "type.raw").write_text("0 0\n")
            (system / "type_map.raw").write_text("H\n")
            (system / "nopbc").touch()
            np.save(
                set_dir / "coord.npy",
                np.array([[0.0, 0.0, 0.0, 0.74, 0.0, 0.0]]),
            )

            data_system = DeepmdDataSystem(
                [str(system)],
                batch_size=1,
                test_size=1,
                rcut=6.0,
                type_map=["H"],
                trn_all_set=True,
                shuffle_test=False,
            )
            raw = data_system.get_batch()
            inputs, _ = split_batch(normalize_batch(raw))

            self.assertFalse(data_system.data_systems[0].pbc)
            self.assertEqual(raw["default_mesh"].size, 0)
            self.assertTrue(np.allclose(raw["box"], 0.0))
            self.assertIsNone(inputs["box"])


if __name__ == "__main__":
    unittest.main()
