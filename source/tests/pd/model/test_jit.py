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

import paddle
from paddle.static import (
    InputSpec,
)

from deepmd.pd.entrypoints.main import (
    get_trainer,
)
from deepmd.pd.infer import (
    inference,
)

from .test_permutation import (
    model_se_e2_a,
)


class JITTest:
    def test_jit(self):
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()
        paddle.set_flags(
            {
                "FLAGS_save_cf_stack_op": 1,
                "FLAGS_prim_enable_dynamic": 1,
                "FLAGS_enable_pir_api": 1,
            }
        )
        model = paddle.jit.to_static(
            inference.Tester("./model.pd").model, full_graph=True
        )
        paddle.jit.save(
            model,
            "./frozen_model",
            input_spec=[
                InputSpec([-1, -1, 3], dtype="float64"),
                InputSpec([-1, -1], dtype="int32"),
                InputSpec([-1, -1, -1], dtype="int32"),
            ],
        )

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith("pd"):
                os.remove(f)
            if f in ["lcurve.out", "frozen_model.json", "frozen_model.pdiparams"]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)
            if f in ["checkpoint"]:
                os.remove(f)


class TestEnergyModelSeA(unittest.TestCase, JITTest):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_se_e2_a)
        self.config["training"]["numb_steps"] = 10
        self.config["training"]["save_freq"] = 10

    def tearDown(self):
        JITTest.tearDown(self)


if __name__ == "__main__":
    unittest.main()
