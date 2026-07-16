# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for pt_expt property-model inference via the DeepProperty interface.

Verifies the full pipeline:
    PropertyModel.serialize() -> deserialize_to_file(.pte) -> DeepEval(.pte)

A pt_expt property checkpoint could be trained but not evaluated: the pt_expt
DeepEval.model_type did not dispatch property models (raising "Unknown model
type") and did not expose the property metadata getters. This checks the
dispatch and the get_var_name / get_task_dim / get_intensive getters.
"""

import os
import tempfile
import unittest

import numpy as np
import torch

from deepmd.infer import (
    DeepEval,
)
from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    PropertyFittingNet,
)
from deepmd.pt_expt.model import (
    PropertyModel,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestDeepEvalProperty(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rcut = 4.0
        cls.rcut_smth = 0.5
        cls.sel = [8, 6]
        cls.nt = 2
        cls.task_dim = 3
        cls.type_map = ["foo", "bar"]

        ds = DescrptSeA(cls.rcut, cls.rcut_smth, cls.sel)
        ft = PropertyFittingNet(
            cls.nt,
            ds.get_dim_out(),
            task_dim=cls.task_dim,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        )
        cls.model = PropertyModel(ds, ft, type_map=cls.type_map).to(torch.float64)
        cls.model.eval()

        cls.tmpfile = tempfile.NamedTemporaryFile(suffix=".pte", delete=False)
        cls.tmpfile.close()
        deserialize_to_file(cls.tmpfile.name, {"model": cls.model.serialize()})
        cls.dp = DeepEval(cls.tmpfile.name)

    @classmethod
    def tearDownClass(cls) -> None:
        os.unlink(cls.tmpfile.name)

    def test_model_type_is_property(self) -> None:
        self.assertIs(self.dp.deep_eval.model_type, DeepProperty)
        self.assertIsInstance(self.dp, DeepProperty)

    def test_get_var_name(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_var_name(), self.model.get_var_name())

    def test_get_task_dim(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_task_dim(), self.task_dim)

    def test_get_intensive(self) -> None:
        self.assertEqual(self.dp.deep_eval.get_intensive(), self.model.get_intensive())

    def test_eval_shape(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.random([1, 4, 3]) * 3.0
        cells = (np.eye(3) * 10.0).reshape(1, 9)
        atypes = np.array([[0, 1, 0, 1]], dtype=np.int32)
        prop = self.dp.eval(coords, cells, atypes)[0]
        self.assertEqual(prop.shape[0], 1)
        self.assertEqual(prop.reshape(1, -1).shape[1], self.task_dim)


if __name__ == "__main__":
    unittest.main()
