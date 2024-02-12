# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from typing import (
    Any,
    ClassVar,
    Tuple,
)

import numpy as np

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA as DescrptSeADP
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.pt.model.descriptor.se_a import DescrptSeA as DescrptSeAPT
from deepmd.tf.descriptor.se_a import DescrptSeA as DescrptSeATF
from deepmd.utils.argcheck import (
    descrpt_se_a_args,
)

from .common import (
    CommonTest,
    DescriptorTest,
)


class CommonTestSeATest(CommonTest, DescriptorTest):
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

    def extract_ret(self, ret: Any, backend) -> Any:
        return ret[0]


class TestSeATypeOneSide(CommonTestSeATest, unittest.TestCase):
    data: ClassVar[dict] = {
        "sel": [46, 92],
        "rcut_smth": 5.80,
        "rcut": 6.00,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": True,
        "type_one_side": True,
        "seed": 1145141919810,
    }


class TestSeATypeTwoSide(CommonTestSeATest, unittest.TestCase):
    data: ClassVar[dict] = {
        "sel": [46, 92],
        "rcut_smth": 5.80,
        "rcut": 6.00,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": False,
        "type_one_side": False,
        "seed": 1145141919810,
    }
    skip_dp = True
    skip_pt = True


class TestSeAExcludeTypeOneSide(CommonTestSeATest, unittest.TestCase):
    data: ClassVar[dict] = {
        "sel": [46, 92],
        "rcut_smth": 5.80,
        "rcut": 6.00,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": False,
        "type_one_side": True,
        "exclude_types": [[0, 1]],
        "seed": 1145141919810,
    }
    unittest.skip("Unsupported by native model")
    skip_dp = True
    skip_pt = True


class TestSeAExcludeTypeTwoSide(CommonTestSeATest, unittest.TestCase):
    data: ClassVar[dict] = {
        "sel": [46, 92],
        "rcut_smth": 5.80,
        "rcut": 6.00,
        "neuron": [25, 50, 100],
        "axis_neuron": 16,
        "resnet_dt": False,
        "type_one_side": False,
        "exclude_types": [[0, 1]],
        "seed": 1145141919810,
    }
    skip_dp = True
    skip_pt = True
