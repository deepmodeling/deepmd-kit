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
import torch

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.infer import (
    inference,
)


class TestDPTest(unittest.TestCase):
    def setUp(self):
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

    def test_dp_test(self):
        trainer = get_trainer(deepcopy(self.config))
        trainer.run()

        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        _, _, more_loss = trainer.wrapper(**input_dict, label=label_dict, cur_lr=1.0)

        tester = inference.Tester("model.pt", input_script=self.input_json)
        try:
            res = tester.run()
        except StopIteration:
            print("Unexpected stop iteration.(test step < total batch)")
            raise StopIteration
        for k, v in res.items():
            if k == "rmse" or "mae" in k or k not in more_loss:
                continue
            np.testing.assert_allclose(
                v, more_loss[k].cpu().detach().numpy(), rtol=1e-04, atol=1e-07
            )

    def tearDown(self):
        for f in os.listdir("."):
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", self.input_json]:
                os.remove(f)
            if f in ["stat_files"]:
                shutil.rmtree(f)


if __name__ == "__main__":
    unittest.main()
