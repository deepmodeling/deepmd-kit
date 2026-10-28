# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import (
    Path,
)

import h5py
import numpy as np
import pytest

from deepmd.dpmodel.utils.dist_check import (
    compute_min_pair_margin_single,
    pair_half_thresholds,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)


def test_cached_numpy_set_applies_copy_weights_once(tmp_path: Path) -> None:
    np.savetxt(tmp_path / "type.raw", [0], fmt="%d")
    set_path = tmp_path / "set.000"
    set_path.mkdir()
    np.save(set_path / "coord.npy", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.save(set_path / "box.npy", np.tile(np.eye(3).reshape(1, 9), (2, 1)))
    np.save(set_path / "numb_copy.npy", [2, 1])
    data = DeepmdData(str(tmp_path))

    assert data.get_mean_batch_nframes(8) == 3
    for _ in range(3):
        batch = data.get_batch(8)
        np.testing.assert_array_equal(np.sort(batch["coord"][:, 0]), [0, 0, 1])


@pytest.mark.parametrize("hdf5", [False, True])
def test_uniform_numpy_counts_undersized_sets(tmp_path: Path, hdf5: bool) -> None:
    paths = []
    container = tmp_path / "systems.h5"
    for index, sizes in enumerate(((1, 5), (2, 8))):
        root = tmp_path / f"system_{index}"
        root.mkdir()
        np.savetxt(root / "type.raw", [0], fmt="%d")
        for set_index, size in enumerate(sizes):
            set_path = root / f"set.{set_index:03d}"
            set_path.mkdir()
            np.save(set_path / "coord.npy", np.full((size, 3), index, dtype=np.float64))
            np.save(set_path / "box.npy", np.tile(np.eye(3).reshape(1, 9), (size, 1)))
        paths.append(str(root))
        if hdf5:
            with h5py.File(container, "a") as handle:
                group = handle.create_group(root.name)
                group.create_dataset("type.raw", data=[0])
                for file in root.rglob("*.npy"):
                    group.create_dataset(
                        str(file.relative_to(root)), data=np.load(file)
                    )
            paths[-1] = f"{container}#/{root.name}"
    data = DeepmdDataSystem(paths, [3, 5], 1, auto_prob_style="prob_uniform")
    means = [
        np.mean([data.get_batch(sys_idx=index)["coord"].shape[0] for _ in range(4)])
        for index in range(2)
    ]
    np.testing.assert_allclose(means, [2, 3.5])
    np.testing.assert_allclose(data.sys_probs * means, [14 / 11, 14 / 11])
    data = DeepmdDataSystem(paths, "mixed:4", 1, auto_prob_style="prob_uniform")
    np.testing.assert_allclose(data.sys_probs, [0.5, 0.5])
    assert data.uniform_sampling_mass == 0.5
    assert data.get_batch()["coord"].shape[0] == 4


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
        self.assertIsNone(data.get_type_map())
        self.assertEqual(data.get_ntypes(), 2)
        np.testing.assert_array_equal(data.atom_type, [0, 1, 0, 1])

    def test_required_dataset_type_map_cannot_be_replaced(self) -> None:
        """A supplied model map does not make a required dataset map optional."""
        (self.root / "type_map.raw").unlink()
        with self.assertRaisesRegex(AssertionError, "must have type_map.raw"):
            DeepmdData(str(self.root), type_map=["O", "H"], optional_type_map=False)

    def test_model_map_preserves_virtual_spin_type_counts(self) -> None:
        """Unnamed virtual types remain part of the reader's numeric type domain."""
        fixture_dir = Path(__file__).parent.parent / "tf"
        model = json.loads((fixture_dir / "test_model_spin.json").read_text())["model"]
        system = fixture_dir / "model_spin"
        self.assertFalse((system / "type_map.raw").exists())
        stored_types = np.loadtxt(system / "type.raw", dtype=np.int32)
        counts = np.bincount(stored_types)
        self.assertGreater(len(counts), len(model["type_map"]))
        data = DeepmdDataSystem(
            [str(system)], batch_size=1, test_size=1, type_map=model["type_map"]
        )
        self.assertEqual(data.get_ntypes(), len(counts))
        np.testing.assert_array_equal(
            data.get_batch()["natoms_vec"],
            np.concatenate(([stored_types.size, stored_types.size], counts)),
        )

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
