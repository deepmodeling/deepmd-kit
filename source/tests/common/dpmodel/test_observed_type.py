# SPDX-License-Identifier: LGPL-3.0-or-later
import shutil
import tempfile
import unittest
from unittest.mock import (
    patch,
)

import numpy as np

from deepmd.dpmodel.atomic_model.pairtab_atomic_model import (
    PairTabAtomicModel,
)
from deepmd.dpmodel.utils.stat import (
    _restore_observed_type_from_file,
    _save_observed_type_to_file,
    collect_observed_types,
)
from deepmd.utils.path import (
    DPPath,
)


class TestCollectObservedTypes(unittest.TestCase):
    """Test collect_observed_types with mock sampled data (numpy backend)."""

    def test_single_system(self) -> None:
        sampled = [
            {"atype": np.array([[0, 1, 0, 1]])},
        ]
        type_map = ["O", "H", "Au"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["H", "O"])

    def test_multiple_systems(self) -> None:
        sampled = [
            {"atype": np.array([[0, 0, 0]])},
            {"atype": np.array([[1, 1, 2]])},
        ]
        type_map = ["O", "H", "Au"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["H", "O", "Au"])

    def test_subset_of_types(self) -> None:
        sampled = [
            {"atype": np.array([[2, 2]])},
        ]
        type_map = ["O", "H", "Au"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["Au"])

    def test_multi_frame(self) -> None:
        sampled = [
            {"atype": np.array([[0, 1], [0, 0]])},
        ]
        type_map = ["O", "H"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["H", "O"])

    def test_out_of_range_index_ignored(self) -> None:
        sampled = [
            {"atype": np.array([[0, 5]])},
        ]
        type_map = ["O", "H"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["O"])


class TestObservedTypeStatFile(unittest.TestCase):
    """Test stat file save/load round-trip for observed_type (dpmodel)."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def test_save_and_restore(self) -> None:
        stat_path = DPPath(self.tmpdir, mode="w")
        observed = ["H", "O"]
        _save_observed_type_to_file(stat_path, observed)
        restored = _restore_observed_type_from_file(DPPath(self.tmpdir))
        self.assertEqual(restored, observed)

    def test_restore_missing_file(self) -> None:
        stat_path = DPPath(self.tmpdir, mode="r")
        result = _restore_observed_type_from_file(stat_path)
        self.assertIsNone(result)

    def test_restore_none_path(self) -> None:
        result = _restore_observed_type_from_file(None)
        self.assertIsNone(result)

    def test_save_none_path(self) -> None:
        # Should not raise
        _save_observed_type_to_file(None, ["H", "O"])


class TestPairTabObservedType(unittest.TestCase):
    """Test observed_type collection for dpmodel PairTabAtomicModel."""

    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        # 3 types -> ntypes*(ntypes+1)/2 = 6 energy columns -> 7 total columns
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
                [0.01, 0.8, 1.6, 2.4, 1.2, 2.0, 2.8],
                [0.015, 0.5, 1.0, 1.5, 0.75, 1.25, 1.75],
                [0.02, 0.25, 0.4, 0.75, 0.35, 0.6, 0.9],
            ]
        )
        self.model = PairTabAtomicModel(
            tab_file="dummy_path", rcut=0.02, sel=2, type_map=["H", "O", "Au"]
        )
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def _make_sampled(self, atypes: list[list[list[int]]]) -> list[dict]:
        """Create mock sampled data from atype arrays."""
        return [{"atype": np.array(a)} for a in atypes]

    def test_compute_observed_type_from_data(self) -> None:
        """PairTab should collect observed types from sampled data."""
        sampled = self._make_sampled([[[0, 1, 0, 1]]])  # H and O only
        self.model.compute_or_load_stat(
            lambda: sampled,
            stat_file_path=DPPath(self.tmpdir, mode="w"),
            compute_or_load_out_stat=False,
        )
        self.assertIsNotNone(self.model.observed_type)
        self.assertIn("H", self.model.observed_type)
        self.assertIn("O", self.model.observed_type)
        self.assertNotIn("Au", self.model.observed_type)

    def test_preset_observed_type_takes_priority(self) -> None:
        """Preset observed_type should override data-based computation."""
        sampled = self._make_sampled([[[0, 1]]])  # H and O in data
        preset = ["H", "O", "Au"]
        self.model.compute_or_load_stat(
            lambda: sampled,
            stat_file_path=DPPath(self.tmpdir, mode="w"),
            compute_or_load_out_stat=False,
            preset_observed_type=preset,
        )
        self.assertEqual(self.model.observed_type, preset)


class TestLinearModelObservedType(unittest.TestCase):
    """Test observed_type propagation in dpmodel DPZBLLinearEnergyAtomicModel."""

    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        from deepmd.dpmodel.atomic_model import (
            DPAtomicModel,
        )
        from deepmd.dpmodel.atomic_model.linear_atomic_model import (
            DPZBLLinearEnergyAtomicModel,
        )
        from deepmd.dpmodel.descriptor import (
            DescrptDPA1,
        )
        from deepmd.dpmodel.fitting.invar_fitting import (
            InvarFitting,
        )

        type_map = ["H", "O", "Au"]

        # 3 types -> ntypes*(ntypes+1)/2 = 6 energy columns -> 7 total columns
        mock_loadtxt.return_value = np.array(
            [
                [0.005, 1.0, 2.0, 3.0, 1.5, 2.5, 3.5],
                [0.01, 0.8, 1.6, 2.4, 1.2, 2.0, 2.8],
                [0.015, 0.5, 1.0, 1.5, 0.75, 1.25, 1.75],
                [0.02, 0.25, 0.4, 0.75, 0.35, 0.6, 0.9],
            ]
        )
        zbl_model = PairTabAtomicModel(
            tab_file="dummy_path", rcut=0.02, sel=2, type_map=type_map
        )

        ds = DescrptDPA1(
            rcut_smth=0.3,
            rcut=0.4,
            sel=[3],
            ntypes=len(type_map),
        )
        ft = InvarFitting(
            "energy",
            len(type_map),
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        dp_model = DPAtomicModel(ds, ft, type_map=type_map)

        self.linear_model = DPZBLLinearEnergyAtomicModel(
            dp_model,
            zbl_model,
            sw_rmin=0.1,
            sw_rmax=0.25,
            type_map=type_map,
        )
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)

    def _make_sampled(self, atypes: list[list[list[int]]]) -> list[dict]:
        """Create mock sampled data from atype arrays."""
        return [{"atype": np.array(a)} for a in atypes]

    def test_parent_observed_type_from_data(self) -> None:
        """Parent (linear) model should collect observed types from data."""
        sampled = self._make_sampled([[[0, 1, 0, 1]]])  # H and O only
        # Mock descriptor/fitting input stats to avoid needing coord/nlist
        with (
            patch.object(self.linear_model.models[0].descriptor, "compute_input_stats"),
            patch.object(
                self.linear_model.models[0].fitting_net, "compute_input_stats"
            ),
        ):
            self.linear_model.compute_or_load_stat(
                lambda: sampled,
                stat_file_path=DPPath(self.tmpdir, mode="w"),
                compute_or_load_out_stat=False,
            )
        self.assertIsNotNone(self.linear_model.observed_type)
        self.assertIn("H", self.linear_model.observed_type)
        self.assertIn("O", self.linear_model.observed_type)
        self.assertNotIn("Au", self.linear_model.observed_type)

    def test_submodels_get_propagated_observed_type(self) -> None:
        """Sub-models should receive parent's observed type via propagation."""
        sampled = self._make_sampled([[[0, 1, 0, 1]]])  # H and O only
        # Mock descriptor/fitting input stats to avoid needing coord/nlist
        with (
            patch.object(self.linear_model.models[0].descriptor, "compute_input_stats"),
            patch.object(
                self.linear_model.models[0].fitting_net, "compute_input_stats"
            ),
        ):
            self.linear_model.compute_or_load_stat(
                lambda: sampled,
                stat_file_path=DPPath(self.tmpdir, mode="w"),
                compute_or_load_out_stat=False,
            )
        dp_model = self.linear_model.models[0]
        zbl_model = self.linear_model.models[1]
        # All three should have the same observed type (propagated from parent)
        self.assertEqual(dp_model.observed_type, self.linear_model.observed_type)
        self.assertEqual(zbl_model.observed_type, self.linear_model.observed_type)


if __name__ == "__main__":
    unittest.main()
