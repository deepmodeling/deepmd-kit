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

import numpy as np
import paddle
from paddle.static import (
    InputSpec,
)

from deepmd.entrypoints.test import test as dp_test
from deepmd.pd.entrypoints.main import (
    get_trainer,
)
from deepmd.pd.utils.utils import (
    to_numpy_array,
)

from .model.test_permutation import (
    model_se_e2_a,
    model_spin,
)


class DPTest:
    @unittest.skip(
        "Paddle do not support testing in frozen models(.json and .pdiparams file), "
        "will be supported in the future."
    )
    def test_dp_test_1_frame(self):
        trainer = get_trainer(deepcopy(self.config))
        device = paddle.get_device()
        paddle.set_device("cpu")
        input_dict, label_dict, _ = trainer.get_data(is_train=False)
        # exit()
        paddle.set_device(device)
        has_spin = getattr(trainer.model, "has_spin", False)

        if callable(has_spin):
            has_spin = has_spin()
        if not has_spin:
            input_dict.pop("spin", None)
        input_dict["do_atomic_virial"] = True
        result = trainer.model(**input_dict)
        paddle.set_flags(
            {
                "FLAGS_save_cf_stack_op": 1,
                "FLAGS_prim_enable_dynamic": 1,
                "FLAGS_enable_pir_api": 1,
            }
        )
        model = paddle.jit.to_static(
            trainer.model,
            full_graph=True,
            input_spec=[
                InputSpec([-1, -1, 3], dtype="float64", name="coord"),
                InputSpec([-1, -1], dtype="int32", name="atype"),
                InputSpec([-1, -1, -1], dtype="int32", name="nlist"),
            ],
        )
        tmp_model = tempfile.NamedTemporaryFile(delete=False, suffix=".pd")
        paddle.jit.save(
            model,
            tmp_model.name,
            skip_prune_program=True,
        )
        dp_test(
            model=tmp_model.name,
            system=self.config["training"]["validation_data"]["systems"][0],
            datafile=None,
            set_prefix="set",
            numb_test=2,
            rand_seed=None,
            shuffle_test=False,
            detail_file=self.detail_file,
            atomic=False,
        )
        os.unlink(tmp_model.name)
        natom = input_dict["atype"].shape[1]
        pred_e = np.loadtxt(self.detail_file + ".e.out", ndmin=2)[0, 1]
        np.testing.assert_almost_equal(
            pred_e,
            to_numpy_array(result["energy"])[0][0],
        )
        pred_e_peratom = np.loadtxt(self.detail_file + ".e_peratom.out", ndmin=2)[0, 1]
        np.testing.assert_almost_equal(pred_e_peratom, pred_e / natom)
        if not has_spin:
            pred_f = np.loadtxt(self.detail_file + ".f.out", ndmin=2)[:, 3:6]
            np.testing.assert_almost_equal(
                pred_f,
                to_numpy_array(result["force"]).reshape(-1, 3),
            )
            pred_v = np.loadtxt(self.detail_file + ".v.out", ndmin=2)[:, 9:18]
            np.testing.assert_almost_equal(
                pred_v,
                to_numpy_array(result["virial"]),
            )
            pred_v_peratom = np.loadtxt(self.detail_file + ".v_peratom.out", ndmin=2)[
                :, 9:18
            ]
            np.testing.assert_almost_equal(pred_v_peratom, pred_v / natom)
        else:
            pred_fr = np.loadtxt(self.detail_file + ".fr.out", ndmin=2)[:, 3:6]
            np.testing.assert_almost_equal(
                pred_fr,
                to_numpy_array(result["force"]).reshape(-1, 3),
            )
            pred_fm = np.loadtxt(self.detail_file + ".fm.out", ndmin=2)[:, 3:6]
            np.testing.assert_almost_equal(
                pred_fm,
                to_numpy_array(
                    result["force_mag"][result["mask_mag"].bool().squeeze(-1)]
                ).reshape(-1, 3),
            )

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pd"):
                os.remove(f)
            if f.startswith(self.detail_file):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


class TestDPTestSeA(DPTest, unittest.TestCase):
    def setUp(self):
        self.detail_file = "test_dp_test_ener_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)


class TestDPTestSeASpin(DPTest, unittest.TestCase):
    def setUp(self):
        self.detail_file = "test_dp_test_ener_spin_detail"
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "NiO/data/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_spin)
        self.config["model"]["type_map"] = ["Ni", "O", "B"]
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)


if __name__ == "__main__":
    unittest.main()
