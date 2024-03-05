# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA as DescrptSeADP
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
    from deepmd.pt.model.descriptor.se_a import DescrptSeA as DescrptSeAPT
else:
    DescrptSeAPT = None
if INSTALLED_TF:
    from deepmd.tf.descriptor.se_a import DescrptSeA as DescrptSeATF
else:
    DescrptSeATF = None
from deepmd.utils.argcheck import (
    descrpt_se_a_args,
)


@parameterized(
    (True, False),  # resnet_dt
    (True, False),  # type_one_side
    ([], [[0, 1]]),  # excluded_types
    ("float32", "float64"),  # precision
    (0.0, 1e-8, 1e-2),  # env_protection
)
class TestSeA(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return {
            "sel": [9, 10],
            "rcut_smth": 5.80,
            "rcut": 6.00,
            "neuron": [6, 12, 24],
            "axis_neuron": 3,
            "resnet_dt": resnet_dt,
            "type_one_side": type_one_side,
            "exclude_types": excluded_types,
            "env_protection": env_protection,
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
            env_protection,
        ) = self.param
        return CommonTest.skip_pt

    @property
    def skip_dp(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return CommonTest.skip_dp

    @property
    def skip_tf(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        return env_protection != 0.0

    tf_class = DescrptSeATF
    dp_class = DescrptSeADP
    pt_class = DescrptSeAPT
    args = descrpt_se_a_args()

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
        # TF se_e2_a type_one_side=False requires atype sorted
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
            env_protection,
        ) = self.param
        if not type_one_side:
            idx = np.argsort(self.atype)
            self.atype = self.atype[idx]
            self.coords = self.coords.reshape(-1, 3)[idx].ravel()

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
            env_protection,
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
            env_protection,
        ) = self.param
        if precision == "float64":
            return 1e-10
        elif precision == "float32":
            return 1e-4
        else:
            raise ValueError(f"Unknown precision: {precision}")
