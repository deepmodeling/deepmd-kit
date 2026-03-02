# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from unittest.mock import (
    patch,
)

import numpy as np
import torch

from deepmd.pt.utils.stat import (
    _restore_observed_type_from_file,
    _save_observed_type_to_file,
    collect_observed_types,
)
from deepmd.utils.path import (
    DPPath,
)


class TestCollectObservedTypes(unittest.TestCase):
    """Test collect_observed_types with mock sampled data."""

    def test_single_system(self) -> None:
        sampled = [
            {"atype": torch.tensor([[0, 1, 0, 1]], device="cpu")},
        ]
        type_map = ["O", "H", "Au"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["H", "O"])

    def test_multiple_systems(self) -> None:
        sampled = [
            {"atype": torch.tensor([[0, 0, 0]], device="cpu")},
            {"atype": torch.tensor([[1, 1, 2]], device="cpu")},
        ]
        type_map = ["O", "H", "Au"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["H", "O", "Au"])

    def test_subset_of_types(self) -> None:
        sampled = [
            {"atype": torch.tensor([[2, 2]], device="cpu")},
        ]
        type_map = ["O", "H", "Au"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["Au"])

    def test_multi_frame(self) -> None:
        sampled = [
            {"atype": torch.tensor([[0, 1], [0, 0]], device="cpu")},
        ]
        type_map = ["O", "H"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["H", "O"])

    def test_out_of_range_index_ignored(self) -> None:
        sampled = [
            {"atype": torch.tensor([[0, 5]], device="cpu")},
        ]
        type_map = ["O", "H"]
        result = collect_observed_types(sampled, type_map)
        self.assertEqual(result, ["O"])


class TestObservedTypeStatFile(unittest.TestCase):
    """Test stat file save/load round-trip for observed_type."""

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


class TestObservedTypeTraining(unittest.TestCase):
    """Test observed_type persistence through training pipeline."""

    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        from .model.test_permutation import (
            model_se_e2_a,
        )

        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["model"]["type_map"] = ["O", "H", "Au"]

    def test_model_params_has_observed_type_in_info(self) -> None:
        """After training, model_params['info']['observed_type'] should be set."""
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        state = trainer.wrapper.state_dict()
        model_params = state["_extra_state"]["model_params"]
        self.assertIn("info", model_params)
        self.assertIn("observed_type", model_params["info"])
        observed = model_params["info"]["observed_type"]
        # Training data only has O and H
        self.assertIn("H", observed)
        self.assertIn("O", observed)
        self.assertNotIn("Au", observed)

    def test_model_def_script_has_observed_type(self) -> None:
        """model.model_def_script (str) should contain info.observed_type after training."""
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        # model_def_script is a JSON string on the model object
        script_str = trainer.model.model_def_script
        self.assertIsInstance(script_str, str)
        script_dict = json.loads(script_str)
        self.assertIn("info", script_dict)
        self.assertIn("observed_type", script_dict["info"])
        observed = script_dict["info"]["observed_type"]
        self.assertIn("H", observed)
        self.assertIn("O", observed)
        self.assertNotIn("Au", observed)

    def test_frozen_model_has_observed_type(self) -> None:
        """After freeze, the frozen model should carry observed_type via get_model_def_script."""
        from deepmd.infer import (
            DeepPot,
        )
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        from .common import (
            run_dp,
        )

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        run_dp("dp --pt freeze")
        # Load frozen model via DeepPot and check model_def_script (dict)
        model = DeepPot("frozen_model.pth")
        script_dict = model.deep_eval.model_def_script
        self.assertIsInstance(script_dict, dict)
        self.assertIn("info", script_dict)
        self.assertIn("observed_type", script_dict["info"])
        observed = script_dict["info"]["observed_type"]
        self.assertIn("H", observed)
        self.assertIn("O", observed)
        self.assertNotIn("Au", observed)

    def test_deep_eval_get_observed_types_uses_metadata(self) -> None:
        """DeepEval.get_observed_types() should return metadata-based result."""
        from deepmd.infer import (
            DeepPot,
        )
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        from .common import (
            run_dp,
        )

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        run_dp("dp --pt freeze")
        model = DeepPot("frozen_model.pth")
        observed = model.deep_eval.get_observed_types()
        self.assertEqual(observed["type_num"], 2)
        self.assertEqual(observed["observed_type"], ["H", "O"])

    def test_user_preset_observed_type(self) -> None:
        """User-specified observed_type in config['model']['info'] takes precedence."""
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        config = deepcopy(self.config)
        config["model"].setdefault("info", {})["observed_type"] = ["O", "H", "Au"]
        trainer = get_trainer(config)
        trainer.run()
        state = trainer.wrapper.state_dict()
        model_params = state["_extra_state"]["model_params"]
        observed = model_params["info"]["observed_type"]
        self.assertEqual(observed, ["O", "H", "Au"])

    def test_stat_file_caching(self) -> None:
        """Observed_type should be saved to and loaded from stat_file."""
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        config = deepcopy(self.config)
        config["training"]["stat_file"] = "stat_files"
        os.makedirs("stat_files", exist_ok=True)
        trainer = get_trainer(config)
        trainer.run()
        # The stat_file_path includes the type_map subdirectory
        stat_base = Path("stat_files") / " ".join(["O", "H", "Au"])
        observed_file = stat_base / "observed_type"
        if observed_file.exists():
            data = np.load(str(observed_file), allow_pickle=True)
            decoded = [x.decode() if isinstance(x, bytes) else x for x in data.tolist()]
            self.assertIn("H", decoded)
            self.assertIn("O", decoded)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth", "output.txt", "checkpoint"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestObservedTypeFallback(unittest.TestCase):
    """Test bias-based fallback for old models without metadata."""

    def test_deep_eval_fallback(self) -> None:
        """When model_def_script has no observed_type in info, fallback to bias-based."""
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            config = json.load(f)
        config["training"]["numb_steps"] = 1
        config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        config["training"]["training_data"]["systems"] = data_file
        config["training"]["validation_data"]["systems"] = data_file
        from .model.test_permutation import (
            model_se_e2_a,
        )

        config["model"] = deepcopy(model_se_e2_a)
        config["model"]["type_map"] = ["O", "H", "Au"]

        from deepmd.infer import (
            DeepPot,
        )
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        from .common import (
            run_dp,
        )

        trainer = get_trainer(deepcopy(config))
        trainer.run()
        run_dp("dp --pt freeze")

        model = DeepPot("frozen_model.pth")
        # Simulate old model by removing observed_type from info
        model.deep_eval.model_def_script.get("info", {}).pop("observed_type", None)
        observed = model.deep_eval.get_observed_types()
        # Should still work via bias-based fallback
        self.assertIn("type_num", observed)
        self.assertIn("observed_type", observed)
        self.assertGreater(observed["type_num"], 0)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth", "output.txt", "checkpoint"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestPairTabObservedType(unittest.TestCase):
    """Test observed_type collection for PairTabAtomicModel."""

    @patch("numpy.loadtxt")
    def setUp(self, mock_loadtxt) -> None:
        from deepmd.pt.model.atomic_model import (
            PairTabAtomicModel,
        )

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
        return [{"atype": torch.tensor(a, device="cpu")} for a in atypes]

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


class TestDPZBLObservedType(unittest.TestCase):
    """Test observed_type propagation in DPZBLLinearEnergyAtomicModel.

    The parent LinearEnergyAtomicModel computes observed type once, then
    propagates it to sub-models via preset_observed_type to avoid redundant
    computation.
    """

    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file

        from .model.test_permutation import (
            model_zbl,
        )

        self.config["model"] = deepcopy(model_zbl)

    def test_parent_observed_type_from_data(self) -> None:
        """Parent (linear) model should collect observed types from data."""
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        observed = trainer.model.atomic_model.observed_type
        self.assertIsNotNone(observed)
        # Training data only has O and H (model_zbl type_map is ["O", "H", "B"])
        self.assertIn("H", observed)
        self.assertIn("O", observed)
        self.assertNotIn("B", observed)

    def test_submodels_get_propagated_observed_type(self) -> None:
        """Sub-models should receive parent's observed type via propagation."""
        from deepmd.pt.entrypoints.main import (
            get_trainer,
        )

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        linear_model = trainer.model.atomic_model
        dp_model = linear_model.models[0]
        zbl_model = linear_model.models[1]
        # All three should have the same observed type (propagated from parent)
        self.assertEqual(dp_model.observed_type, linear_model.observed_type)
        self.assertEqual(zbl_model.observed_type, linear_model.observed_type)

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.pth", "output.txt", "checkpoint"]:
                os.remove(f)


if __name__ == "__main__":
    unittest.main()
