# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np

from deepmd.pd.entrypoints.main import (
    get_trainer,
)
from deepmd.pd.utils.env import (
    enable_prim,
)
from deepmd.pd.utils.finetune import (
    get_finetune_rules,
)

from .model.test_permutation import (
    model_dpa1,
    model_dpa2,
    model_se_e2_a,
)


class DPTrainTest:
    def test_dp_train(self) -> None:
        # test training from scratch
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        state_dict_trained = trainer.wrapper.model.state_dict()

        # test fine-tuning using same input
        finetune_model = self.config["training"].get("save_ckpt", "model.ckpt") + ".pd"
        self.config["model"], finetune_links = get_finetune_rules(
            finetune_model,
            self.config["model"],
        )
        trainer_finetune = get_trainer(
            deepcopy(self.config),
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # test fine-tuning using empty input
        self.config_empty = deepcopy(self.config)
        if "descriptor" in self.config_empty["model"]:
            self.config_empty["model"]["descriptor"] = {}
        if "fitting_net" in self.config_empty["model"]:
            self.config_empty["model"]["fitting_net"] = {}
        self.config_empty["model"], finetune_links = get_finetune_rules(
            finetune_model,
            self.config_empty["model"],
            change_model_params=True,
        )
        trainer_finetune_empty = get_trainer(
            deepcopy(self.config_empty),
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # test fine-tuning using random fitting
        self.config["model"], finetune_links = get_finetune_rules(
            finetune_model, self.config["model"], model_branch="RANDOM"
        )
        trainer_finetune_random = get_trainer(
            deepcopy(self.config_empty),
            finetune_model=finetune_model,
            finetune_links=finetune_links,
        )

        # check parameters
        state_dict_finetuned = trainer_finetune.wrapper.model.state_dict()
        state_dict_finetuned_empty = trainer_finetune_empty.wrapper.model.state_dict()
        state_dict_finetuned_random = trainer_finetune_random.wrapper.model.state_dict()
        for state_key in state_dict_finetuned:
            if "out_bias" not in state_key and "out_std" not in state_key:
                np.testing.assert_allclose(
                    state_dict_trained[state_key].numpy(),
                    state_dict_finetuned[state_key].numpy(),
                )
                np.testing.assert_allclose(
                    state_dict_trained[state_key].numpy(),
                    state_dict_finetuned_empty[state_key].numpy(),
                )
                if (
                    ("fitting_net" not in state_key)
                    or ("fparam" in state_key)
                    or ("aparam" in state_key)
                ):
                    np.testing.assert_allclose(
                        state_dict_trained[state_key].numpy(),
                        state_dict_finetuned_random[state_key].numpy(),
                    )

        # check running
        trainer_finetune.run()
        trainer_finetune_empty.run()
        trainer_finetune_random.run()

    def test_trainable(self) -> None:
        fix_params = deepcopy(self.config)
        fix_params["model"]["descriptor"]["trainable"] = False
        fix_params["model"]["fitting_net"]["trainable"] = False
        free_descriptor = hasattr(self, "not_all_grad") and self.not_all_grad
        if free_descriptor:
            # can not set requires_grad false for all parameters,
            # because the input coord has no grad, thus the loss if all set to false
            # we only check trainable for fitting net
            fix_params["model"]["descriptor"]["trainable"] = True
            trainer_fix = get_trainer(fix_params)
            model_dict_before_training = deepcopy(
                trainer_fix.model.get_fitting_net().state_dict()
            )
            trainer_fix.run()
            model_dict_after_training = deepcopy(
                trainer_fix.model.get_fitting_net().state_dict()
            )
        else:
            trainer_fix = get_trainer(fix_params)
            model_dict_before_training = deepcopy(trainer_fix.model.state_dict())
            trainer_fix.run()
            model_dict_after_training = deepcopy(trainer_fix.model.state_dict())
        for key in model_dict_before_training:
            np.testing.assert_allclose(
                model_dict_before_training[key].numpy(),
                model_dict_after_training[key].numpy(),
            )

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pd"):
                if os.path.exists(f):
                    os.remove(f)
            if f in ["lcurve.out"]:
                if os.path.exists(f):
                    os.remove(f)
            if f in ["stat_files"]:
                if os.path.exists(f):
                    shutil.rmtree(f)


class TestEnergyModelSeA(unittest.TestCase, DPTrainTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        enable_prim(True)

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestEnergyModelGradientAccumulation(unittest.TestCase, DPTrainTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.config["training"]["acc_freq"] = 4
        enable_prim(True)

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestFparam(unittest.TestCase, DPTrainTest):
    """Test if `fparam` can be loaded correctly."""

    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["model"]["fitting_net"]["numb_fparam"] = 1
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        self.set_path = Path(__file__).parent / "water/data/data_0" / "set.000"
        shutil.copyfile(self.set_path / "energy.npy", self.set_path / "fparam.npy")
        self.config["model"]["data_stat_nbatch"] = 100

    def tearDown(self) -> None:
        (self.set_path / "fparam.npy").unlink(missing_ok=True)
        DPTrainTest.tearDown(self)


class TestEnergyModelDPA1(unittest.TestCase, DPTrainTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_dpa1)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestEnergyModelDPA2(unittest.TestCase, DPTrainTest):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_dpa2)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1

    def tearDown(self) -> None:
        DPTrainTest.tearDown(self)


class TestModelChangeOutBiasFittingStat(unittest.TestCase):
    """Verify model_change_out_bias produces the same fitting stat as the old code path.

    The old code called compute_fitting_input_stat inside change_out_bias (make_model.py).
    The new code calls get_fitting_net().compute_input_stats() separately in
    model_change_out_bias (training.py). This test verifies they produce identical
    out_bias, fparam_avg, and fparam_inv_std.
    """

    def test_fitting_stat_consistency(self) -> None:
        from deepmd.pd.model.model import get_model as get_model_pd
        from deepmd.pd.model.model.ener_model import EnergyModel as EnergyModelPD
        from deepmd.pd.train.training import (
            model_change_out_bias,
        )
        from deepmd.pd.utils.utils import to_numpy_array as paddle_to_numpy
        from deepmd.pd.utils.utils import to_paddle_tensor as numpy_to_paddle
        from deepmd.utils.argcheck import model_args as model_args_fn

        # Build a model with numb_fparam=2 so fitting stat is non-trivial
        model_params = model_args_fn().normalize_value(
            {
                "type_map": ["O", "H"],
                "descriptor": {
                    "type": "se_e2_a",
                    "sel": [20, 20],
                    "rcut_smth": 0.50,
                    "rcut": 6.00,
                    "neuron": [3, 6],
                    "resnet_dt": False,
                    "axis_neuron": 2,
                    "precision": "float64",
                    "type_one_side": True,
                    "seed": 1,
                },
                "fitting_net": {
                    "neuron": [5, 5],
                    "resnet_dt": True,
                    "precision": "float64",
                    "seed": 1,
                    "numb_fparam": 2,
                },
            },
            trim_pattern="_*",
        )

        # Create two identical models via serialize/deserialize
        model_orig = get_model_pd(model_params)
        serialized = model_orig.serialize()
        model_a = EnergyModelPD.deserialize(deepcopy(serialized))
        model_b = EnergyModelPD.deserialize(deepcopy(serialized))

        # Build mock stat data with fparam
        nframes = 4
        natoms = 6
        coords = np.random.default_rng(42).random((nframes, natoms, 3)) * 13.0
        atype = np.array([[0, 0, 1, 1, 1, 1]] * nframes, dtype=np.int32)
        box = np.tile(
            np.eye(3, dtype=np.float64).reshape(1, 3, 3) * 13.0, (nframes, 1, 1)
        )
        natoms_data = np.array([[6, 6, 2, 4]] * nframes, dtype=np.int32)
        energy = np.array([10.0, 20.0, 15.0, 25.0]).reshape(nframes, 1)
        # fparam with varying values so mean != 0 and std != 0
        fparam = np.array(
            [[1.0, 3.0], [5.0, 7.0], [2.0, 8.0], [6.0, 4.0]], dtype=np.float64
        )

        merged = [
            {
                "coord": numpy_to_paddle(coords),
                "atype": numpy_to_paddle(atype),
                "atype_ext": numpy_to_paddle(atype),
                "box": numpy_to_paddle(box),
                "natoms": numpy_to_paddle(natoms_data),
                "energy": numpy_to_paddle(energy),
                "find_energy": np.float32(1.0),
                "fparam": numpy_to_paddle(fparam),
                "find_fparam": np.float32(1.0),
            }
        ]

        # Model A: simulate the OLD code path
        # old change_out_bias called both bias adjustment + compute_fitting_input_stat
        model_a.change_out_bias(merged, bias_adjust_mode="set-by-statistic")
        model_a.atomic_model.compute_fitting_input_stat(merged)

        # Model B: use the NEW code path via model_change_out_bias
        sample_func = lambda: merged  # noqa: E731
        model_change_out_bias(model_b, sample_func, "set-by-statistic")

        # Compare out_bias
        bias_a = paddle_to_numpy(model_a.get_out_bias())
        bias_b = paddle_to_numpy(model_b.get_out_bias())
        np.testing.assert_allclose(bias_a, bias_b, rtol=1e-10, atol=1e-10)

        # Compare fparam_avg and fparam_inv_std
        fit_a = model_a.get_fitting_net()
        fit_b = model_b.get_fitting_net()
        fparam_avg_a = paddle_to_numpy(fit_a.fparam_avg)
        fparam_avg_b = paddle_to_numpy(fit_b.fparam_avg)
        fparam_inv_std_a = paddle_to_numpy(fit_a.fparam_inv_std)
        fparam_inv_std_b = paddle_to_numpy(fit_b.fparam_inv_std)

        np.testing.assert_allclose(fparam_avg_a, fparam_avg_b, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            fparam_inv_std_a, fparam_inv_std_b, rtol=1e-10, atol=1e-10
        )

        # Verify non-trivial: avg should not be zeros, inv_std should not be ones
        assert not np.allclose(fparam_avg_a, 0.0), (
            "fparam_avg is still zero — stat was not computed"
        )
        assert not np.allclose(fparam_inv_std_a, 1.0), (
            "fparam_inv_std is still ones — stat was not computed"
        )


if __name__ == "__main__":
    unittest.main()
