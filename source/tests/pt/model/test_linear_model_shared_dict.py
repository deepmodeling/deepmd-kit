# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.model import (
    get_model,
)


class TestLinearEnergySharedDict(unittest.TestCase):
    def test_shared_dict_descriptor_and_type_map(self) -> None:
        config = {
            "type": "linear_ener",
            "shared_dict": {
                "type_map_all": ["O", "H"],
                "dpa1_descriptor": {
                    "type": "dpa1",
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "sel": 4,
                    "neuron": [4, 8, 16],
                    "axis_neuron": 4,
                    "seed": 1,
                },
            },
            "models": [
                {
                    "type_map": "type_map_all",
                    "descriptor": "dpa1_descriptor",
                    "fitting_net": {
                        "neuron": [8, 8, 8],
                        "resnet_dt": True,
                        "seed": 1,
                    },
                },
                {
                    "type_map": "type_map_all",
                    "descriptor": "dpa1_descriptor",
                    "fitting_net": {
                        "neuron": [8, 8, 8],
                        "resnet_dt": True,
                        "seed": 2,
                    },
                },
            ],
            "weights": "mean",
        }

        model = get_model(config)

        self.assertEqual(model.get_type_map(), ["O", "H"])
        self.assertIsNotNone(model.shared_links)
        self.assertIn("dpa1_descriptor", model.shared_links)
        self.assertEqual(len(model.atomic_model.models), 2)
        descriptor0 = model.atomic_model.models[0].descriptor
        descriptor1 = model.atomic_model.models[1].descriptor
        self.assertIs(descriptor1.type_embedding, descriptor0.type_embedding)
        self.assertIs(descriptor1.se_atten, descriptor0.se_atten)

    def test_shared_dict_descriptor_with_top_level_type_map(self) -> None:
        config = {
            "type": "linear_ener",
            "type_map": ["O", "H"],
            "shared_dict": {
                "dpa1_descriptor": {
                    "type": "dpa1",
                    "rcut": 6.0,
                    "rcut_smth": 0.5,
                    "sel": 4,
                    "neuron": [4, 8, 16],
                    "axis_neuron": 4,
                    "seed": 1,
                },
            },
            "models": [
                {
                    "descriptor": "dpa1_descriptor",
                    "fitting_net": {
                        "neuron": [8, 8, 8],
                        "resnet_dt": True,
                        "seed": 1,
                    },
                },
                {
                    "descriptor": "dpa1_descriptor",
                    "fitting_net": {
                        "neuron": [8, 8, 8],
                        "resnet_dt": True,
                        "seed": 2,
                    },
                },
            ],
            "weights": "mean",
        }

        model = get_model(config)

        self.assertEqual(model.get_type_map(), ["O", "H"])
        descriptor0 = model.atomic_model.models[0].descriptor
        descriptor1 = model.atomic_model.models[1].descriptor
        self.assertIs(descriptor1.type_embedding, descriptor0.type_embedding)
        self.assertIs(descriptor1.se_atten, descriptor0.se_atten)
