# SPDX-License-Identifier: LGPL-3.0-or-later
"""Preset output bias through the JAX model factory and bias change."""

import unittest

import numpy as np

from deepmd.jax.env import (
    jnp,
)
from deepmd.jax.model.ener_model import (
    EnergyModel,
)
from deepmd.jax.model.model import (
    get_model,
)

from ..common.stat_file import (
    energy_model_params,
    energy_stat_sample,
)


class TestPresetOutBias(unittest.TestCase):
    def setUp(self) -> None:
        params = energy_model_params()
        params["preset_out_bias"] = {"energy": {"H": 5.0}}
        self.model = get_model(params)
        self.sampled = [
            {
                key: jnp.asarray(value) if isinstance(value, np.ndarray) else value
                for key, value in sample.items()
            }
            for sample in energy_stat_sample()
        ]

    def out_bias(self) -> np.ndarray:
        return np.asarray(self.model.get_out_bias()).reshape(-1)

    def test_preset_pinned_in_both_modes(self) -> None:
        self.assertEqual(
            self.model.atomic_model.preset_out_bias, {"energy": [None, [5.0]]}
        )
        # frame 1 holds two O atoms with energy 2, so O is fitted to 1 while H is pinned
        self.model.change_out_bias(self.sampled, bias_adjust_mode="set-by-statistic")
        np.testing.assert_allclose(self.out_bias(), [1.0, 5.0])
        self.model.change_out_bias(self.sampled, bias_adjust_mode="change-by-statistic")
        bias_changed = self.out_bias()
        np.testing.assert_allclose(bias_changed[1], 5.0)
        self.model.change_out_bias(self.sampled, bias_adjust_mode="change-by-statistic")
        np.testing.assert_allclose(self.out_bias(), bias_changed, atol=1e-10)

    def test_serialize_round_trip(self) -> None:
        self.model.change_out_bias(self.sampled, bias_adjust_mode="set-by-statistic")
        loaded = EnergyModel.deserialize(self.model.serialize())
        self.assertEqual(loaded.atomic_model.preset_out_bias, {"energy": [None, [5.0]]})
        np.testing.assert_allclose(
            np.asarray(loaded.get_out_bias()), np.asarray(self.model.get_out_bias())
        )
