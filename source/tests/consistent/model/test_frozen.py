# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
    parameterized,
)
from .common import (
    ModelTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.model import BaseModel as FrozenModelPT

else:
    FrozenModelPT = None
if INSTALLED_TF:
    from deepmd.tf.model.model import Model as FrozenModelTF
else:
    FrozenModelTF = None

from deepmd.utils.argcheck import (
    model_args,
)

from ...infer.case import (
    get_cases,
)

pt_model = "deeppot_for_consistent_frozen.pth"
tf_model = "deeppot_for_consistent_frozen.pb"
dp_model = "deeppot_for_consistent_frozen.dp"


def setUpModule() -> None:
    case = get_cases()["se_e2_a"]
    case.get_model(".dp", dp_model)
    case.get_model(".pb", tf_model)
    case.get_model(".pth", pt_model)


def tearDownModule() -> None:
    for model_file in (dp_model, pt_model, tf_model):
        try:
            os.remove(model_file)
        except FileNotFoundError:
            pass


@parameterized((pt_model, tf_model, dp_model))
class TestFrozen(CommonTest, ModelTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (model_file,) = self.param
        if not INSTALLED_PT and model_file.endswith(".pth"):
            raise unittest.SkipTest("PyTorch is not installed")
        if not INSTALLED_TF and model_file.endswith(".pb"):
            raise unittest.SkipTest("TensorFlow is not installed")
        return {
            "type": "frozen",
            "model_file": model_file,
        }

    tf_class = FrozenModelTF
    dp_class = None
    pt_class = FrozenModelPT
    args = model_args()

    def skip_dp(self) -> bool:
        return True

    def setUp(self) -> None:
        CommonTest.setUp(self)

        self.ntypes = 2
        self.coords = np.array(
            [
                12.83,
                2.56,
                2.18,
                12.09,
                2.87,
                2.74,
                00.25,
                3.32,
                1.68,
                3.36,
                3.00,
                1.81,
                3.51,
                2.51,
                2.60,
                4.27,
                3.22,
                1.56,
            ],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, -1, 3)
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32).reshape(1, -1)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        ).reshape(1, 9)
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

        # TF requires the atype to be sort
        idx_map = np.argsort(self.atype.ravel())
        self.atype = self.atype[:, idx_map]
        self.coords = self.coords[:, idx_map]

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
        return self.build_tf_model(
            obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            suffix,
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_model(
            dp_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_model(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        # shape not matched. ravel...
        if backend is self.RefBackend.DP:
            return (ret["energy_redu"].ravel(), ret["energy"].ravel())
        elif backend is self.RefBackend.PT:
            return (ret["energy"].ravel(), ret["atom_energy"].ravel())
        elif backend is self.RefBackend.TF:
            return (ret[0].ravel(), ret[1].ravel())
        raise ValueError(f"Unknown backend: {backend}")
