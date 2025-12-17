# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import unittest
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.modifier.base_modifier import (
    BaseModifier,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)
from deepmd.utils.argcheck import (
    modifier_args_plugin,
)
from deepmd.utils.data import (
    DeepmdData,
)


@modifier_args_plugin.register("random_tester")
def modifier_random_tester() -> list:
    return []


@modifier_args_plugin.register("zero_tester")
def modifier_zero_tester() -> list:
    return []


@BaseModifier.register("random_tester")
class ModifierRandomTester(BaseModifier):
    def __new__(cls) -> BaseModifier:
        return super().__new__(cls)

    def __init__(self) -> None:
        """Construct a basic model for different tasks."""
        super().__init__()
        self.modifier_type = "random_tester"

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {"coord": coord}

    def modify_data(self, data: dict[str, Array | float], data_sys: DeepmdData) -> None:
        """Multiply by a random factor."""
        if (
            "find_energy" not in data
            and "find_force" not in data
            and "find_virial" not in data
        ):
            return

        if "find_energy" in data and data["find_energy"] == 1.0:
            data["energy"] = data["energy"] * np.random.default_rng().random()
        if "find_force" in data and data["find_force"] == 1.0:
            data["force"] = data["force"] * np.random.default_rng().random()
        if "find_virial" in data and data["find_virial"] == 1.0:
            data["virial"] = data["virial"] * np.random.default_rng().random()


@BaseModifier.register("zero_tester")
class ModifierZeroTester(BaseModifier):
    def __new__(cls) -> BaseModifier:
        return super().__new__(cls)

    def __init__(self) -> None:
        """Construct a basic model for different tasks."""
        super().__init__()
        self.modifier_type = "zero_tester"

    def forward(
        self,
        coord: torch.Tensor,
        atype: torch.Tensor,
        box: torch.Tensor | None = None,
        fparam: torch.Tensor | None = None,
        aparam: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return {"coord": coord}

    def modify_data(self, data: dict[str, Array | float], data_sys: DeepmdData) -> None:
        """Multiply by a random factor."""
        if (
            "find_energy" not in data
            and "find_force" not in data
            and "find_virial" not in data
        ):
            return

        if "find_energy" in data and data["find_energy"] == 1.0:
            data["energy"] -= data["energy"]
        if "find_force" in data and data["find_force"] == 1.0:
            data["force"] -= data["force"]
        if "find_virial" in data and data["find_virial"] == 1.0:
            data["virial"] -= data["virial"]


class TestDataModifier(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test fixtures."""
        input_json = str(Path(__file__).parent / "water/se_e2_a.json")
        with open(input_json, encoding="utf-8") as f:
            config = json.load(f)
        config["training"]["numb_steps"] = 10
        config["training"]["save_freq"] = 1
        config["learning_rate"]["start_lr"] = 1.0
        config["training"]["training_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]
        config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]
        self.config = config

    def test_init_modify_data(self):
        """Ensure modify_data applied."""
        tmp_config = self.config.copy()
        # add tester data modifier
        tmp_config["model"]["modifier"] = {"type": "zero_tester"}

        # data modification is finished in __init__
        trainer = get_trainer(tmp_config)

        # training data
        training_data = trainer.get_data(is_train=True)
        # validation data
        validation_data = trainer.get_data(is_train=False)

        for dataset in [training_data, validation_data]:
            for kw in ["energy", "force"]:
                data = to_numpy_array(dataset[1][kw])
                np.testing.assert_allclose(data, np.zeros_like(data))

    def test_full_modify_data(self):
        """Ensure modify_data only applied once."""
        tmp_config = self.config.copy()
        # add tester data modifier
        tmp_config["model"]["modifier"] = {"type": "random_tester"}

        # data modification is finished in __init__
        trainer = get_trainer(tmp_config)

        # training data
        training_data_before = trainer.get_data(is_train=True)
        # validation data
        validation_data_before = trainer.get_data(is_train=False)

        trainer.run()

        # training data
        training_data_after = trainer.get_data(is_train=True)
        # validation data
        validation_data_after = trainer.get_data(is_train=False)

        for kw in ["energy", "force"]:
            np.testing.assert_allclose(
                to_numpy_array(training_data_before[1][kw]),
                to_numpy_array(training_data_after[1][kw]),
            )
            np.testing.assert_allclose(
                to_numpy_array(validation_data_before[1][kw]),
                to_numpy_array(validation_data_after[1][kw]),
            )

    def tearDown(self) -> None:
        for f in os.listdir("."):
            if f.startswith("frozen_model") and f.endswith(".pth"):
                os.remove(f)
            if f.startswith("model") and f.endswith(".pt"):
                os.remove(f)
            if f in ["lcurve.out", "checkpoint"]:
                os.remove(f)
