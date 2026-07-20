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

from deepmd.jax.model.ener_model import (
    EnergyModel,
)
from deepmd.jax.model.model import (
    get_model,
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


if __name__ == "__main__":
    unittest.main()
