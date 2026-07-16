# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test the JAX standard-model factory defaults the fitting type to energy.

dpmodel and TF2 both default ``fitting_net.type`` to ``"ener"`` (and tolerate a
missing ``fitting_net`` block) when constructing a standard model.  The JAX
factory used a bare ``data["fitting_net"].pop("type")``, so a config that omits
``fitting_net`` or leaves ``type`` unset raised ``KeyError`` on JAX only.  In
normal training this is masked because argcheck normalization fills the default
before the factory runs; the factory is exposed when called with a raw dict.
"""

import unittest
from unittest.mock import (
    patch,
)

from deepmd.jax.model.base_model import (
    BaseModel,
)
from deepmd.jax.model.ener_model import (
    EnergyModel,
)
from deepmd.jax.model.model import (
    get_model,
)
from deepmd.utils.argcheck import (
    model_args,
)


def _base_config() -> dict:
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [20, 20],
            "rcut_smth": 0.5,
            "rcut": 6.0,
            "neuron": [3, 6],
            "axis_neuron": 2,
            "precision": "float64",
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [5, 5],
            "precision": "float64",
            "seed": 1,
        },
    }


def _base_sezm_config() -> dict:
    """Return the smallest config needed to exercise DPA4 factory routing."""
    return {
        "type": "dpa4",
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "dpa4",
            "random_gamma": False,
            "use_amp": False,
        },
        "fitting_net": {"type": "dpa4_ener"},
    }


class TestJAXModelFactoryFittingDefault(unittest.TestCase):
    def test_fitting_net_without_type_defaults_to_ener(self) -> None:
        # fitting_net present but no "type": must default to energy.
        data = _base_config()
        model = get_model(data)
        self.assertIsInstance(model, EnergyModel)

    def test_fitting_net_omitted_defaults_to_ener(self) -> None:
        # fitting_net block omitted entirely: must default to energy.
        data = _base_config()
        del data["fitting_net"]
        model = get_model(data)
        self.assertIsInstance(model, EnergyModel)

    def test_explicit_fitting_type_preserved(self) -> None:
        # Control: an explicit type is still honored.
        data = _base_config()
        data["fitting_net"]["type"] = "ener"
        model = get_model(data)
        self.assertIsInstance(model, EnergyModel)


class TestJAXSeZMModelFactory(unittest.TestCase):
    @patch("deepmd.jax.model.model.get_standard_model", side_effect=lambda data: data)
    def test_null_blocks_receive_dpa4_defaults(self, _get_standard_model) -> None:
        data = _base_sezm_config()
        data["descriptor"] = None
        data["fitting_net"] = None

        normalized = get_model(data)

        self.assertEqual(normalized["descriptor"]["type"], "dpa4")
        self.assertEqual(normalized["fitting_net"]["type"], "dpa4_ener")

    def test_rejects_unsupported_features(self) -> None:
        cases = (
            ("spin", {}),
            ("bridging_method", "linear"),
            ("lora", {}),
            ("use_compile", True),
            ("preset_out_bias", [0.0]),
        )
        for key, value in cases:
            with self.subTest(key=key):
                data = _base_sezm_config()
                data[key] = value
                with self.assertRaises(NotImplementedError):
                    get_model(data)

        data = _base_sezm_config()
        data["descriptor"]["add_chg_spin_ebd"] = True
        with self.assertRaises(NotImplementedError):
            get_model(data)

        for key in ("random_gamma", "use_amp"):
            with self.subTest(descriptor_option=key):
                data = _base_sezm_config()
                data["descriptor"][key] = True
                with self.assertRaisesRegex(NotImplementedError, key):
                    get_model(data)

    def test_rejects_incompatible_descriptor_and_fitting_types(self) -> None:
        data = _base_sezm_config()
        data["descriptor"]["type"] = "se_e2_a"
        with self.assertRaises(ValueError):
            get_model(data)

        data = _base_sezm_config()
        data["fitting_net"]["type"] = "ener"
        with self.assertRaises(ValueError):
            get_model(data)

    def test_rejects_mismatched_exclude_types(self) -> None:
        data = _base_sezm_config()
        data["descriptor"]["exclude_types"] = [[0, 1]]
        data["pair_exclude_types"] = [[1, 1]]

        with self.assertRaises(ValueError):
            get_model(data)

    @patch(
        "deepmd.dpmodel.model.dp_model.BaseDescriptor.update_sel",
        return_value=({"type": "dpa4", "sel": 16}, 0.75),
    )
    def test_model_aliases_route_through_update_sel(self, update_sel) -> None:
        """Neighbor-stat preprocessing recognizes every public DPA4 alias."""
        for model_type in ("dpa4", "DPA4", "sezm", "SeZM"):
            with self.subTest(model_type=model_type):
                local_jdata = {
                    "type": model_type,
                    "descriptor": {"type": "dpa4", "sel": "auto"},
                }

                updated, min_nbor_dist = BaseModel.update_sel(
                    object(), ["O", "H"], local_jdata
                )

                self.assertEqual(updated["descriptor"]["sel"], 16)
                self.assertEqual(min_nbor_dist, 0.75)
        self.assertEqual(update_sel.call_count, 4)

    @patch("deepmd.jax.model.model.get_standard_model", side_effect=lambda data: data)
    def test_descriptor_exclude_types_feed_standard_model(
        self,
        _get_standard_model,
    ) -> None:
        data = _base_sezm_config()
        data["descriptor"] = {
            "type": "SeZM",
            "exclude_types": [[0, 1]],
        }
        data["fitting_net"]["type"] = "sezm_ener"

        normalized = get_model(data)

        self.assertEqual(normalized["pair_exclude_types"], [[0, 1]])
        self.assertEqual(normalized["descriptor"]["exclude_types"], [[0, 1]])

    @patch("deepmd.jax.model.model.get_standard_model", side_effect=lambda data: data)
    def test_normalized_descriptor_exclusions_override_empty_default(
        self,
        _get_standard_model,
    ) -> None:
        """Argcheck's empty model-level default is not an explicit mismatch."""
        data = _base_sezm_config()
        data["descriptor"]["exclude_types"] = [[0, 1]]
        data = model_args().normalize_value(data, trim_pattern="_.*")

        normalized = get_model(data)

        self.assertEqual(normalized["pair_exclude_types"], [[0, 1]])
        self.assertEqual(normalized["descriptor"]["exclude_types"], [[0, 1]])


if __name__ == "__main__":
    unittest.main()
