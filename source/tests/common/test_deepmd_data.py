# SPDX-License-Identifier: LGPL-3.0-or-later
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.utils.data import (
    DeepmdData,
)


class TestDeepmdDataTypeMap(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.set_dir = self.root / "set.000"
        self.set_dir.mkdir()

        # minimal required dataset
        atom_types = np.array([0, 1, 0, 1], dtype=np.int32)
        np.savetxt(self.root / "type.raw", atom_types, fmt="%d")
        np.savetxt(
            self.root / "type_map.raw",
            np.array(["O", "H", "Si"], dtype=object),
            fmt="%s",
        )

        coord = np.zeros((1, atom_types.size * 3), dtype=np.float32)
        box = np.eye(3, dtype=np.float32).reshape(1, 9)
        np.save(self.set_dir / "coord.npy", coord)
        np.save(self.set_dir / "box.npy", box)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_remap_with_unused_types(self) -> None:
        data = DeepmdData(str(self.root), type_map=["H", "O", "Si"])

        expected_atom_types = np.array([1, 0, 1, 0], dtype=np.int32)
        np.testing.assert_array_equal(data.atom_type, expected_atom_types)
        self.assertEqual(data.type_map, ["H", "O", "Si"])

        loaded = data._load_set(self.set_dir)
        expected_sorted = expected_atom_types[data.idx_map]
        np.testing.assert_array_equal(loaded["type"], np.tile(expected_sorted, (1, 1)))
