# SPDX-License-Identifier: LGPL-3.0-or-later
"""Property-model inference on the TF2 backend (SavedModel export).

Mirrors the dpmodel/JAX property tests.  The TF2 evaluator wraps a
``TF2SavedModelWrapper`` with no live model, so the property variable name,
output dimension and intensiveness must be carried through the SavedModel for
the generic ``DeepEval`` factory to dispatch to ``DeepProperty`` and reshape the
output.  Placed under ``consistent`` (and named ``tf2``) so the TF2-only CI job,
which runs ``source/tests/consistent -k tf2``, collects it.
"""

import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.infer.deep_property import (
    DeepProperty,
)

from .common import (
    INSTALLED_TF2,
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


@unittest.skipUnless(INSTALLED_TF2, "TF2 backend is not available")
class TestDeepEvalPropertyTF2(unittest.TestCase):
    def setUp(self) -> None:
        from deepmd.tf2.utils.serialization import (
            deserialize_to_file,
        )

        config = _property_model_config()
        model = get_model_dp(config)
        self.tmpdir = tempfile.TemporaryDirectory()
        model_file = str(Path(self.tmpdir.name) / "property.savedmodeltf")
        deserialize_to_file(
            model_file,
            {"model": model.serialize(), "model_def_script": {"model": config}},
        )
        self.dp = DeepEval(model_file)
        # two frames to guard against single-frame reduction bugs.
        rng = np.random.default_rng(0)
        self.coords = rng.random([2, 6, 3]) * 4.0
        self.cells = np.tile((np.eye(3) * 10.0).reshape(1, 9), (2, 1))
        self.atypes = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_model_type_dispatch(self) -> None:
        # The generic factory must resolve a property .savedmodeltf to
        # DeepProperty (raised "Unknown model type" before the fix).
        self.assertIsInstance(self.dp, DeepProperty)

    def test_getters_survive_export(self) -> None:
        self.assertEqual(self.dp.get_var_name(), "foo")
        self.assertEqual(self.dp.get_task_dim(), 3)
        self.assertIsInstance(self.dp.get_intensive(), bool)

    def test_eval_global_only(self) -> None:
        # atomic=False must return the global property via the reduced output.
        (prop,) = self.dp.eval(self.coords, self.cells, self.atypes, atomic=False)
        self.assertEqual(prop.shape, (2, self.dp.get_task_dim()))

    def test_eval_global_matches_atomic_sum(self) -> None:
        prop, atomic_prop = self.dp.eval(
            self.coords, self.cells, self.atypes, atomic=True
        )
        self.assertEqual(atomic_prop.shape, (2, 6, self.dp.get_task_dim()))
        np.testing.assert_allclose(prop, np.sum(atomic_prop, axis=1))


if __name__ == "__main__":
    unittest.main()
