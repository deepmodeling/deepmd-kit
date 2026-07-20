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
from pathlib import (
    Path,
)

import numpy as np

from deepmd.jax.env import (
    jnp,
    nnx,
)
from deepmd.jax.model.ener_model import (
    EnergyModel,
)
from deepmd.jax.model.model import (
    get_model,
)

from ..common.stat_file import (
    assert_energy_stat_cache_round_trip,
    energy_model_params,
    energy_stat_sample,
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


def test_jax_array_assignment_preserves_variable_for_shape_change() -> None:
    """Backend array assignment updates an existing NNX variable container."""
    descriptor = get_model(energy_model_params()).get_descriptor()
    variable = descriptor.davg
    new_shape = (3, *variable.shape[1:])

    descriptor.davg = jnp.zeros(new_shape, dtype=variable.dtype)

    assert descriptor.davg is variable
    assert descriptor.davg.shape == new_shape


def test_jax_energy_cache_round_trip_uses_fitting_outputs_only(
    tmp_path: Path,
) -> None:
    models = []

    def model_factory():
        model = get_model(energy_model_params())
        models.append(model)
        return model

    def sample_factory():
        return [
            {
                key: jnp.asarray(value) if isinstance(value, np.ndarray) else value
                for key, value in sample.items()
            }
            for sample in energy_stat_sample()
        ]

    assert_energy_stat_cache_round_trip(
        model_factory,
        tmp_path / "stat.hdf5",
        sample_factory=sample_factory,
    )
    for model in models:
        descriptor = model.get_descriptor()
        fitting = model.get_fitting_net()
        for value in (
            *descriptor.get_stat_mean_and_stddev(),
            fitting.fparam_avg,
            fitting.fparam_inv_std,
            fitting.aparam_avg,
            fitting.aparam_inv_std,
            model.atomic_model.out_bias,
            model.atomic_model.out_std,
        ):
            assert isinstance(value, nnx.Variable)


if __name__ == "__main__":
    unittest.main()
