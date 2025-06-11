# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
)

import numpy as np

from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DescrptHybridDP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)

from ..common import (
    INSTALLED_ARRAY_API_STRICT,
    INSTALLED_JAX,
    INSTALLED_PT,
    INSTALLED_TF,
    CommonTest,
)
from .common import (
    DescriptorTest,
)

if INSTALLED_PT:
    from deepmd.pt.model.descriptor.hybrid import DescrptHybrid as DescrptHybridPT
else:
    DescrptHybridPT = None
if INSTALLED_TF:
    from deepmd.tf.descriptor.hybrid import DescrptHybrid as DescrptHybridTF
else:
    DescrptHybridTF = None
if INSTALLED_JAX:
    from deepmd.jax.descriptor.hybrid import DescrptHybrid as DescrptHybridJAX
else:
    DescrptHybridJAX = None
if INSTALLED_ARRAY_API_STRICT:
    from ...array_api_strict.descriptor.hybrid import (
        DescrptHybrid as DescrptHybridStrict,
    )
else:
    DescrptHybridStrict = None
from deepmd.utils.argcheck import (
    descrpt_hybrid_args,
)


class TestHybrid(CommonTest, DescriptorTest, unittest.TestCase):
    @property
    def data(self) -> dict:
        return {
            "list": [
                {
                    "type": "se_e2_r",
                    # test the case that sel are different!
                    "sel": [10, 10],
                    "rcut_smth": 5.80,
                    "rcut": 6.00,
                    "neuron": [6, 12, 24],
                    "resnet_dt": False,
                    "type_one_side": True,
                    "precision": "float64",
                    "seed": 20240229,
                },
                {
                    "type": "se_e2_a",
                    "sel": [9, 11],
                    "rcut_smth": 2.80,
                    "rcut": 3.00,
                    "neuron": [6, 12, 24],
                    "axis_neuron": 3,
                    "resnet_dt": True,
                    "type_one_side": True,
                    "precision": "float64",
                    "seed": 20240229,
                },
            ]
        }

    tf_class = DescrptHybridTF
    dp_class = DescrptHybridDP
    pt_class = DescrptHybridPT
    jax_class = DescrptHybridJAX
    array_api_strict_class = DescrptHybridStrict
    args = descrpt_hybrid_args()

    skip_jax = not INSTALLED_JAX
    skip_array_api_strict = not INSTALLED_ARRAY_API_STRICT

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

    def eval_array_api_strict(self, array_api_strict_obj: Any) -> Any:
        return self.eval_array_api_strict_descriptor(
            array_api_strict_obj,
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

    def extract_ret(self, ret: Any, backend) -> tuple[np.ndarray, ...]:
        return (ret[0], ret[1])
