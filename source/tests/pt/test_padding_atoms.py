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
import torch.nn.functional as F

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from .model.test_permutation import model_property

class TestPaddingAtomsPropertySeA(unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "property/input.json")
        with open(input_json) as f:
            self.config = json.load(f)
        data_file = [str(Path(__file__).parent / "property/single")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = data_file
        self.config["model"] = deepcopy(model_property)
        self.config["model"]["type_map"] = [
            self.config["model"]["type_map"][i] for i in [1, 0, 3, 2]
        ]

    def test_dp_test_padding_atoms(self) -> None:
        trainer = get_trainer(deepcopy(self.config))
        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        input_dict.pop("spin", None)
        result = trainer.model(**input_dict)
        padding_atoms_list = [1, 5, 10]
        for padding_atoms in padding_atoms_list:
            input_dict_padding = deepcopy(input_dict)
            input_dict_padding["atype"] = F.pad(
                input_dict_padding["atype"], (0, padding_atoms), value=-1
            )
            input_dict_padding["coord"] = F.pad(
                input_dict_padding["coord"], (0, 0, 0, padding_atoms, 0, 0), value=0
            )
            result_padding = trainer.model(**input_dict_padding)
            np.testing.assert_almost_equal(
                to_numpy_array(result[trainer.model.get_var_name()])[0],
                to_numpy_array(result_padding[trainer.model.get_var_name()])[0],
            )

    def tearDown(self) -> None:
        pass


if __name__ == "__main__":
    unittest.main()
