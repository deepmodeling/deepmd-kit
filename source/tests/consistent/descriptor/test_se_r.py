# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.descriptor.se_r import DescrptSeR as DescrptSeRDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_PT_EXPT,
    INSTALLED_TF,
    INSTALLED_TF2,
    CommonTest,
    parameterized_cases,
)
from .common import (
    DescriptorAPITest,
    DescriptorTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.descriptor.se_r import DescrptSeR as DescrptSeRPT
else:
    DescrptSeRPT = None
if INSTALLED_PT_EXPT:
    from deepmd.pt_expt.descriptor.se_r import DescrptSeR as DescrptSeRPTExpt
else:
    DescrptSeRPTExpt = None
if INSTALLED_TF:
    from deepmd.tf.descriptor.se_r import DescrptSeR as DescrptSeRTF
else:
    DescrptSeRTF = None
if INSTALLED_TF2:
    from deepmd.tf2.descriptor.se_e2_r import DescrptSeR as DescrptSeRTF2
else:
    DescrptSeRTF2 = None
from deepmd.utils.argcheck import (
    descrpt_se_r_args,
)

if INSTALLED_JAX:
    from deepmd.jax.descriptor.se_e2_r import DescrptSeR as DescrptSeRJAX
else:
    DescrptSeRJAX = None
if INSTALLED_ARRAY_API_STRICT:
    from ...array_api_strict.descriptor.se_e2_r import (
        DescrptSeR as DescrptSeRArrayAPIStrict,
    )
else:
    DescrptSeRArrayAPIStrict = None


SE_R_CASE_FIELDS = (
    "resnet_dt",
    "type_one_side",
    "excluded_types",
    "precision",
)

SE_R_BASELINE_CASE = {
    "resnet_dt": True,
    "type_one_side": True,
    "excluded_types": [],
    "precision": "float64",
}


def se_r_case(**overrides: Any) -> tuple:
    case = SE_R_BASELINE_CASE | overrides
    return tuple(case[field] for field in SE_R_CASE_FIELDS)


SE_R_CURATED_CASES = (
    se_r_case(),
    se_r_case(resnet_dt=False),
    se_r_case(type_one_side=False),
    se_r_case(excluded_types=[[0, 1]]),
    se_r_case(precision="float32"),
)

SE_R_DESCRIPTOR_API_CURATED_CASES = (
    se_r_case(),
    se_r_case(resnet_dt=False),
    se_r_case(type_one_side=False),
    se_r_case(excluded_types=[[0, 1]]),
)


@parameterized_cases(*SE_R_CURATED_CASES)
class TestSeR(CommonTest, DescriptorTest, unittest.TestCase):
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
            "activation_function": "relu",
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
    def skip_pt_expt(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or CommonTest.skip_pt_expt

    @property
    def skip_dp(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or CommonTest.skip_dp

    @property
    def skip_jax(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or not INSTALLED_JAX

    @property
    def skip_array_api_strict(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or not INSTALLED_ARRAY_API_STRICT

    @property
    def skip_tf2(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or not INSTALLED_TF2

    tf_class = DescrptSeRTF
    tf2_class = DescrptSeRTF2
    dp_class = DescrptSeRDP
    pt_class = DescrptSeRPT
    pt_expt_class = DescrptSeRPTExpt
    jax_class = DescrptSeRJAX
    array_api_strict_class = DescrptSeRArrayAPIStrict
    args = descrpt_se_r_args()

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
        )
        self.atype = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
        self.box = np.array(
            [13.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 13.0],
            dtype=GLOBAL_NP_FLOAT_PRECISION,
        )
        self.natoms = np.array([6, 6, 2, 4], dtype=np.int32)

    def build_tf(self, obj: Any, suffix: str) -> tuple[list, dict]:
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

    def eval_pt_expt(self, pt_expt_obj: Any) -> Any:
        return self.eval_pt_expt_descriptor(
            pt_expt_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_tf2(self, tf2_obj: Any) -> Any:
        return self.eval_tf2_descriptor(
            tf2_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_jax(self, jax_obj: Any) -> Any:
        return self.eval_jax_descriptor(
            jax_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        return self.eval_array_api_strict_descriptor(
            array_api_strict_obj,
            self.natoms,
            self.coords,
            self.atype,
            self.box,
        )

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
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


@parameterized_cases(*SE_R_DESCRIPTOR_API_CURATED_CASES)
class TestSeRDescriptorAPI(DescriptorAPITest, unittest.TestCase):
    """Test consistency of BaseDescriptor API methods across backends."""

    dp_class = DescrptSeRDP
    pt_class = DescrptSeRPT
    pt_expt_class = DescrptSeRPTExpt
    args = descrpt_se_r_args()

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
            "activation_function": "relu",
        }

    @property
    def skip_pt(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or not INSTALLED_PT

    @property
    def skip_pt_expt(self) -> bool:
        (
            resnet_dt,
            type_one_side,
            excluded_types,
            precision,
        ) = self.param
        return not type_one_side or not INSTALLED_PT_EXPT
