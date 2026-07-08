# SPDX-License-Identifier: LGPL-3.0-or-later
"""Property-model inference on the dpmodel backend.

A property model could be built and serialized but not evaluated: the generic
``DeepEval`` factory dispatched only energy/DOS/dipole/polar/WFC and raised
``RuntimeError("Unknown model type")`` for a property model, and the backend
``DeepEval`` did not expose the ``get_var_name``/``get_task_dim``/
``get_intensive`` getters that ``DeepProperty`` needs.  This test covers the
full ``serialize -> .dp -> DeepEval`` round trip.
"""

import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.dpmodel.utils.serialization import (
    save_dp_model,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.infer.deep_property import (
    DeepProperty,
)


def _property_model_config() -> dict:
    return {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [20, 20],
            "rcut_smth": 1.8,
            "rcut": 6.0,
            "neuron": [2, 4, 8],
            "resnet_dt": False,
            "axis_neuron": 8,
            "precision": "float64",
            "type_one_side": True,
            "seed": 1,
        },
        "fitting_net": {
            "type": "property",
            "neuron": [4, 4, 4],
            "property_name": "foo",
            "task_dim": 3,
            "resnet_dt": True,
            "numb_fparam": 0,
            "precision": "float64",
            "seed": 1,
        },
    }


class TestDeepPropertyDPModel(unittest.TestCase):
    def setUp(self) -> None:
        model = get_model_dp(_property_model_config())
        self.tmpdir = tempfile.TemporaryDirectory()
        model_file = str(Path(self.tmpdir.name) / "property.dp")
        save_dp_model(model_file, {"model": model.serialize()})
        self.dp = DeepEval(model_file)
        # two frames to guard against single-frame reduction bugs.
        rng = np.random.default_rng(0)
        self.coords = rng.random([2, 6, 3]) * 4.0
        self.cells = np.tile((np.eye(3) * 10.0).reshape(1, 9), (2, 1))
        self.atypes = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_model_type_dispatch(self) -> None:
        # The generic factory must resolve a property model to DeepProperty
        # (raised "Unknown model type" before the fix).
        self.assertIsInstance(self.dp, DeepProperty)

    def test_getters(self) -> None:
        self.assertEqual(self.dp.get_var_name(), "foo")
        self.assertEqual(self.dp.get_task_dim(), 3)
        self.assertIsInstance(self.dp.get_intensive(), bool)

    def test_eval_global_only(self) -> None:
        # atomic=False must return the global property via the reduced output,
        # without requiring the atomic key (KeyError before the fix).
        (prop,) = self.dp.eval(self.coords, self.cells, self.atypes, atomic=False)
        self.assertEqual(prop.shape, (2, self.dp.get_task_dim()))

    def test_eval_global_matches_atomic_sum(self) -> None:
        # For an extensive property the global value is the sum of the atomic
        # contributions per frame.
        self.assertFalse(self.dp.get_intensive())
        prop, atomic_prop = self.dp.eval(
            self.coords, self.cells, self.atypes, atomic=True
        )
        self.assertEqual(atomic_prop.shape, (2, 6, self.dp.get_task_dim()))
        np.testing.assert_allclose(prop, np.sum(atomic_prop, axis=1))


if __name__ == "__main__":
    unittest.main()
