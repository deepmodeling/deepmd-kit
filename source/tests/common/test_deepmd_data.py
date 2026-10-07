# SPDX-License-Identifier: LGPL-3.0-or-later
import subprocess
import sys
import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel.utils.dist_check import (
    compute_min_pair_margin_single,
    pair_half_thresholds,
)
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

    def test_covalent_margin_without_dataset_type_map(self) -> None:
        """Model type indices retain their element names without a dataset map."""
        (self.root / "type_map.raw").unlink()
        coord = np.array(
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
        )
        box = 6.0 * np.eye(3)
        np.save(self.set_dir / "coord.npy", coord.reshape(1, -1))
        np.save(self.set_dir / "box.npy", box.reshape(1, 9))
        type_map = ["O", "H", "Si"]
        data = DeepmdData(str(self.root), type_map=type_map)
        data.add("pair_margin", 1, default=0.5, length_scale="covalent")
        expected = compute_min_pair_margin_single(
            coord,
            box,
            np.array([0, 1, 0, 1]),
            pair_half_thresholds(0.5, "covalent", type_map),
            screened=True,
        )
        for loaded in (data.get_single_frame(0, num_worker=1), data.get_batch(1)):
            self.assertEqual(float(loaded["find_pair_margin"]), 1.0)
            np.testing.assert_allclose(loaded["pair_margin"], expected)
        self.assertEqual(data.get_type_map(), type_map)
        np.testing.assert_array_equal(data.atom_type, [0, 1, 0, 1])

    def test_required_dataset_type_map_cannot_be_replaced(self) -> None:
        """A supplied model map does not make a required dataset map optional."""
        (self.root / "type_map.raw").unlink()
        with self.assertRaisesRegex(AssertionError, "must have type_map.raw"):
            DeepmdData(str(self.root), type_map=["O", "H"], optional_type_map=False)

    def test_data_system_imports_without_preloading_dpmodel(self) -> None:
        """The generic reader imports independently of backend initialization."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from deepmd.utils.data_system import DeepmdDataSystem",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


class _RaisingModifier:
    """A modifier whose ``modify_data`` always fails."""

    use_cache = True

    def modify_data(self, data: dict, data_sys: DeepmdData) -> None:
        raise ValueError("modifier failure")


class TestDeepmdDataModifierError(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        set_dir = self.root / "set.000"
        set_dir.mkdir()
        atom_types = np.array([0, 1], dtype=np.int32)
        np.savetxt(self.root / "type.raw", atom_types, fmt="%d")
        np.save(
            set_dir / "coord.npy",
            np.zeros((3, atom_types.size * 3), dtype=np.float32),
        )
        np.save(
            set_dir / "box.npy",
            np.tile(np.eye(3, dtype=np.float32).reshape(9), (3, 1)),
        )

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_get_single_frame_propagates_modifier_error(self) -> None:
        data = DeepmdData(str(self.root), modifier=_RaisingModifier())
        # a failing modifier must surface its error, not be swallowed
        with self.assertRaises(ValueError):
            data.get_single_frame(0, num_worker=1)
        # and the unmodified frame must not be cached
        self.assertNotIn(0, data._modified_frame_cache)
