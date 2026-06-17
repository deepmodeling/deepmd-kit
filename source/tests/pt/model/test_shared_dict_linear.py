# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils.multi_task import (
    preprocess_linear_shared_params,
)


class TestSharedDictLinear(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "type": "linear_ener",
            "type_map": ["O", "H"],
            "shared_dict": {
                "type_map_all": ["O", "H"],
                "shared_descriptor": {
                    "type": "dpa1",
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "sel": 16,
                    "neuron": [4, 8, 16],
                    "axis_neuron": 4,
                    "seed": 1,
                },
            },
            "models": [
                {
                    "type_map": "type_map_all",
                    "descriptor": "shared_descriptor",
                    "fitting_net": {
                        "neuron": [8, 8],
                        "resnet_dt": True,
                        "seed": 1,
                    },
                },
                {
                    "type_map": "type_map_all",
                    "descriptor": "shared_descriptor",
                    "fitting_net": {
                        "neuron": [8, 8],
                        "resnet_dt": True,
                        "seed": 2,
                    },
                },
            ],
            "weights": "mean",
        }

    def test_preprocess_linear_shared_dict(self) -> None:
        model_config, shared_links = preprocess_linear_shared_params(
            copy.deepcopy(self.config)
        )

        self.assertEqual(model_config["models"][0]["type_map"], ["O", "H"])
        self.assertIsInstance(model_config["models"][0]["descriptor"], dict)
        self.assertIn("shared_descriptor", shared_links)
        self.assertEqual(
            [item["model_key"] for item in shared_links["shared_descriptor"]["links"]],
            ["model_0", "model_1"],
        )

    def test_linear_model_shares_descriptor_params(self) -> None:
        model = get_model(copy.deepcopy(self.config))
        self.assertIsNotNone(model.shared_links)

        model.share_params(model.shared_links)

        descriptor_0 = model.atomic_model.models[0].descriptor
        descriptor_1 = model.atomic_model.models[1].descriptor
        self.assertIs(descriptor_0.type_embedding, descriptor_1.type_embedding)


if __name__ == "__main__":
    unittest.main()
