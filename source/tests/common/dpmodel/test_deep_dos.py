# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test global-DOS-only inference on the dpmodel backend.

``DeepDOS.eval`` used to read the atomic ``dos`` output unconditionally and sum
it to obtain the global DOS.  The dpmodel (and JAX) backends only return the
atomic ``OUT`` variables when ``atomic=True``; for ``atomic=False`` they return
the reduced ``dos_redu`` instead, so reading ``results["dos"]`` raised
``KeyError``.  A global-DOS-only path (e.g. ``dp test`` without atomic DOS
labels) must use the reduced output.
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
from deepmd.infer.deep_dos import (
    DeepDOS,
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


class TestDeepDOSDPModel(unittest.TestCase):
    def setUp(self) -> None:
        model = get_model_dp(_dos_model_config())
        self.tmpdir = tempfile.TemporaryDirectory()
        model_file = str(Path(self.tmpdir.name) / "dos.dp")
        save_dp_model(model_file, {"model": model.serialize()})
        self.dp = DeepDOS(model_file)
        rng = np.random.default_rng(0)
        self.coords = rng.random([1, 6, 3]) * 4.0
        self.cells = (np.eye(3) * 10.0).reshape(1, 9)
        self.atypes = np.array([[0, 1, 1, 0, 1, 1]], dtype=np.int32)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_global_dos_only(self) -> None:
        # atomic=False must return the global DOS via the reduced output,
        # without requiring the atomic `dos` key.
        (dos,) = self.dp.eval(self.coords, self.cells, self.atypes, atomic=False)
        self.assertEqual(dos.shape, (1, self.dp.get_numb_dos()))

    def test_global_matches_atomic_sum(self) -> None:
        # The reduced global DOS must equal the sum of the atomic DOS.
        (dos,) = self.dp.eval(self.coords, self.cells, self.atypes, atomic=False)
        _, atomic_dos = self.dp.eval(self.coords, self.cells, self.atypes, atomic=True)
        np.testing.assert_allclose(dos, np.sum(atomic_dos, axis=1))


if __name__ == "__main__":
    unittest.main()
