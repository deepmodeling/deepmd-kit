# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.descriptor.se_r import DescrptSeR as DescrptSeRDP
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
    DescriptorTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.descriptor.se_r import DescrptSeR as DescrptSeRPT
else:
    DescrptSeAPT = None
if INSTALLED_TF:
    from deepmd.tf.descriptor.se_r import DescrptSeR as DescrptSeRTF
else:
    DescrptSeATF = None
from deepmd.utils.argcheck import (
    descrpt_se_r_args,
)


@parameterized(
    (True, False),  # resnet_dt
    (True, False),  # type_one_side
    ([], [[0, 1]]),  # excluded_types
    ("float32", "float64"),  # precision
)
class TestSeA(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return {
            "sel": [9, 10],
            "rcut_smth": 5.80,
            "rcut": 6.00,
            "neuron": [6, 12, 24],
            "resnet_dt": resnet_dt,
            "type_one_side": type_one_side,
            "exclude_types": excluded_types,
            "precision": precision,
            "seed": 1145141919810,
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or CommonTest.skip_pt

    @property
    def skip_dp(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or CommonTest.skip_dp

    tf_class = DescrptSeRTF
    dp_class = DescrptSeRDP
    pt_class = DescrptSeRPT
    args = descrpt_se_r_args()

    def setUp(self):
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
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

    def build_tf(self, obj: Any, suffix: str) -> Tuple[list, dict]:
        return self.build_tf_descriptor(
            obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
            suffix,
        )

    def eval_dp(self, dp_obj: Any) -> Any:
        return self.eval_dp_descriptor(
            dp_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_pt(self, pt_obj: Any) -> Any:
        return self.eval_pt_descriptor(
            pt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> Tuple[np.ndarray, ...]:
        return (ret[0],)

    @property
    def rtol(self) -> float:
        """Relative tolerance for comparing the return value."""
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")

    @property
    def atol(self) -> float:
        """Absolute tolerance for comparing the return value."""
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
