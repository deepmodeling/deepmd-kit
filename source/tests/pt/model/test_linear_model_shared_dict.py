# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.model import (
    BaseModel,
)


class TestLinearEnergySharedDict(unittest.TestCase):
    def assert_dpa1_descriptor_shared(self, descriptor0, descriptor1) -> None:
        self.assertIs(descriptor1.type_embedding, descriptor0.type_embedding)
        self.assertGreater(len(descriptor0.se_atten._modules), 0)
        for module_name, module in descriptor0.se_atten._modules.items():
            self.assertIs(descriptor1.se_atten._modules[module_name], module)

    def make_dpa1_descriptor(self, seed: int) -> dict:
        return {
            "type": "dpa1",
            "sel": 4,
            "rcut_smth": 0.5,
            "rcut": 6.0,
            "neuron": [4, 8, 16],
            "axis_neuron": 4,
            "seed": seed,
        }

    def make_se_e2_a_descriptor(self, sel: str) -> dict:
        return {
            "type": "se_e2_a",
            "rcut": 6.0,
            "sel": sel,
        }

    def make_fitting_net(self, seed: int) -> dict:
        return {
            "neuron": [8, 8, 8],
            "resnet_dt": True,
            "seed": seed,
        }

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
        self.assert_dpa1_descriptor_shared(descriptor0, descriptor1)

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
        self.assert_dpa1_descriptor_shared(descriptor0, descriptor1)

    def test_shared_dict_hybrid_descriptor_component(self) -> None:
        config = {
            "type": "linear_ener",
            "type_map": ["O", "H"],
            "shared_dict": {
                "dpa1_descriptor": self.make_dpa1_descriptor(seed=1),
            },
            "models": [
                {
                    "descriptor": {
                        "type": "hybrid",
                        "list": [
                            "dpa1_descriptor",
                            self.make_dpa1_descriptor(seed=2),
                        ],
                    },
                    "fitting_net": self.make_fitting_net(seed=1),
                },
                {
                    "descriptor": {
                        "type": "hybrid",
                        "list": [
                            "dpa1_descriptor",
                            self.make_dpa1_descriptor(seed=3),
                        ],
                    },
                    "fitting_net": self.make_fitting_net(seed=2),
                },
            ],
            "weights": "mean",
        }

        model = get_model(config)

        self.assertIn("dpa1_descriptor", model.shared_links)
        descriptor0 = model.atomic_model.models[0].descriptor.descrpt_list[0]
        descriptor1 = model.atomic_model.models[1].descriptor.descrpt_list[0]
        self.assert_dpa1_descriptor_shared(descriptor0, descriptor1)

    @patch("deepmd.pt.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_shared_dict_update_sel_round_trip(self, sel_mock) -> None:
        sel_mock.return_value = 0.25, [10, 20]
        config = {
            "type": "linear_ener",
            "shared_dict": {
                "type_map_all": ["O", "H"],
                "shared_descriptor": {
                    "type": "se_e2_a",
                    "rcut": 6.0,
                    "sel": "auto",
                },
            },
            "models": [
                {
                    "type_map": "type_map_all",
                    "descriptor": {
                        "type": "hybrid",
                        "list": [
                            "shared_descriptor",
                            {
                                "type": "se_e2_a",
                                "rcut": 6.0,
                                "sel": "auto:1.5",
                            },
                        ],
                    },
                    "fitting_net": {
                        "neuron": [8, 8, 8],
                        "resnet_dt": True,
                        "seed": 1,
                    },
                },
            ],
            "weights": "mean",
        }

        updated, min_nbor_dist = BaseModel.update_sel(None, None, config)

        self.assertEqual(min_nbor_dist, 0.25)
        self.assertEqual(updated["type_map"], ["O", "H"])
        self.assertEqual(
            updated["models"][0]["descriptor"]["list"][0], "shared_descriptor"
        )
        self.assertEqual(updated["shared_dict"]["shared_descriptor"]["sel"], [12, 24])
        self.assertEqual(updated["models"][0]["descriptor"]["list"][1]["sel"], [16, 32])

    @patch("deepmd.pt.utils.update_sel.UpdateSel.get_nbor_stat")
    def test_shared_dict_update_sel_string_and_inline_descriptors(
        self, sel_mock
    ) -> None:
        sel_mock.return_value = 0.25, [10, 20]
        config = {
            "type": "linear_ener",
            "type_map": ["O", "H"],
            "shared_dict": {
                "shared_descriptor": self.make_se_e2_a_descriptor(sel="auto"),
            },
            "models": [
                {
                    "descriptor": "shared_descriptor",
                    "fitting_net": self.make_fitting_net(seed=1),
                },
                {
                    "descriptor": self.make_se_e2_a_descriptor(sel="auto:1.5"),
                    "fitting_net": self.make_fitting_net(seed=2),
                },
            ],
            "weights": "mean",
        }

        updated, min_nbor_dist = BaseModel.update_sel(None, None, config)

        self.assertEqual(min_nbor_dist, 0.25)
        self.assertEqual(updated["models"][0]["descriptor"], "shared_descriptor")
        self.assertEqual(updated["shared_dict"]["shared_descriptor"]["sel"], [12, 24])
        self.assertEqual(updated["models"][1]["descriptor"]["sel"], [16, 32])

    def test_shared_dict_fitting_net(self) -> None:
        config = {
            "type": "linear_ener",
            "type_map": ["O", "H"],
            "shared_dict": {
                "shared_fit": {
                    "neuron": [8, 8, 8],
                    "resnet_dt": True,
                    "seed": 1,
                },
            },
            "models": [
                {
                    "descriptor": {
                        "type": "dpa1",
                        "sel": 4,
                        "rcut_smth": 0.5,
                        "rcut": 6.0,
                        "neuron": [4, 8, 16],
                        "axis_neuron": 4,
                        "seed": 1,
                    },
                    "fitting_net": "shared_fit",
                },
                {
                    "descriptor": {
                        "type": "dpa1",
                        "sel": 4,
                        "rcut_smth": 0.5,
                        "rcut": 6.0,
                        "neuron": [4, 8, 16],
                        "axis_neuron": 4,
                        "seed": 2,
                    },
                    "fitting_net": "shared_fit",
                },
            ],
            "weights": "mean",
        }

        model = get_model(config)

        self.assertIsNotNone(model.shared_links)
        self.assertIn("shared_fit", model.shared_links)
        fitting0 = model.atomic_model.models[0].fitting_net
        fitting1 = model.atomic_model.models[1].fitting_net
        self.assertGreater(len(fitting0._modules), 0)
        for module_name, module in fitting0._modules.items():
            self.assertIs(fitting1._modules[module_name], module)

    def test_shared_dict_requires_sub_model_type_map_without_top_level(self) -> None:
        config = {
            "type": "linear_ener",
            "shared_dict": {
                "shared_descriptor": self.make_dpa1_descriptor(seed=1),
            },
            "models": [
                {
                    "type_map": ["O", "H"],
                    "descriptor": "shared_descriptor",
                    "fitting_net": self.make_fitting_net(seed=1),
                },
                {
                    "descriptor": "shared_descriptor",
                    "fitting_net": self.make_fitting_net(seed=2),
                },
            ],
            "weights": "mean",
        }

        with self.assertRaisesRegex(
            ValueError, "Linear sub-model 1 must define type_map"
        ):
            get_model(config)

    def test_shared_dict_rejects_inconsistent_sub_model_type_map(self) -> None:
        config = {
            "type": "linear_ener",
            "shared_dict": {
                "shared_descriptor": self.make_dpa1_descriptor(seed=1),
            },
            "models": [
                {
                    "type_map": ["O", "H"],
                    "descriptor": "shared_descriptor",
                    "fitting_net": self.make_fitting_net(seed=1),
                },
                {
                    "type_map": ["H", "O"],
                    "descriptor": "shared_descriptor",
                    "fitting_net": self.make_fitting_net(seed=2),
                },
            ],
            "weights": "mean",
        }

        with self.assertRaisesRegex(
            ValueError,
            "Linear sub-model 1 type_map differs from sub-model 0",
        ):
            get_model(config)

    def test_shared_dict_rejects_inconsistent_type_map_with_top_level(
        self,
    ) -> None:
        config = {
            "type": "linear_ener",
            "type_map": ["O", "H"],
            "shared_dict": {
                "shared_descriptor": self.make_dpa1_descriptor(seed=1),
            },
            "models": [
                {
                    "type_map": ["O", "H"],
                    "descriptor": "shared_descriptor",
                    "fitting_net": self.make_fitting_net(seed=1),
                },
                {
                    "type_map": ["H", "O"],
                    "descriptor": "shared_descriptor",
                    "fitting_net": self.make_fitting_net(seed=2),
                },
            ],
            "weights": "mean",
        }

        with self.assertRaisesRegex(
            ValueError,
            "Linear sub-model 1 type_map .* shared descriptor 'shared_descriptor'",
        ):
            get_model(config)
