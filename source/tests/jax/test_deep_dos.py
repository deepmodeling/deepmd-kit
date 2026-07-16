# SPDX-License-Identifier: LGPL-3.0-or-later
"""Global-DOS-only inference on the JAX backend (StableHLO export).

Mirrors the dpmodel DOS inference test.  The JAX evaluator wraps an ``HLO``
object with no live model, so ``numb_dos`` must be carried through the StableHLO
export for ``DeepDOS.eval`` to reshape the reduced DOS output correctly.
"""

import tempfile
import unittest
from pathlib import (
    Path,
)

import numpy as np

from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.infer.deep_dos import (
    DeepDOS,
)
from deepmd.jax.utils.serialization import (
    deserialize_to_file,
)


def _dos_model_config() -> dict:
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
            "type": "dos",
            "numb_dos": 2,
            "neuron": [4, 4, 4],
            "resnet_dt": True,
            "numb_fparam": 0,
            "precision": "float64",
            "seed": 1,
        },
    }


class TestDeepDOSJAX(unittest.TestCase):
    def setUp(self) -> None:
        config = _dos_model_config()
        model = get_model_dp(config)
        self.tmpdir = tempfile.TemporaryDirectory()
        model_file = str(Path(self.tmpdir.name) / "dos.hlo")
        deserialize_to_file(
            model_file,
            {"model": model.serialize(), "model_def_script": {"model": config}},
        )
        self.dp = DeepDOS(model_file)
        rng = np.random.default_rng(0)
        self.coords = rng.random([1, 6, 3]) * 4.0
        self.cells = (np.eye(3) * 10.0).reshape(1, 9)
        self.atypes = np.array([[0, 1, 1, 0, 1, 1]], dtype=np.int32)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_numb_dos_survives_export(self) -> None:
        self.assertEqual(self.dp.get_numb_dos(), 2)

    def test_neighbor_count_survives_export(self) -> None:
        """Load the exported HLO and expose its serialized selection width."""
        self.assertEqual(self.dp.deep_eval.dp.get_nnei(), 40)

    def test_global_dos_only(self) -> None:
        (dos,) = self.dp.eval(self.coords, self.cells, self.atypes, atomic=False)
        self.assertEqual(dos.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
