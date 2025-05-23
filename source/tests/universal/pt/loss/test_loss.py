# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.loss import (
    DOSLoss,
    EnergySpinLoss,
    EnergyStdLoss,
    PropertyLoss,
    TensorLoss,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.loss.loss import (
    LossTest,
)
from ...dpmodel.loss.test_loss import (
    LossParamDosList,
    LossParamEnergyList,
    LossParamEnergySpinList,
    LossParamPropertyList,
    LossParamTensorList,
)
from ..backend import (
    PTTestCase,
)


@parameterized(
    (
        *[(param_func, EnergyStdLoss) for param_func in LossParamEnergyList],
        *[(param_func, EnergySpinLoss) for param_func in LossParamEnergySpinList],
        *[(param_func, DOSLoss) for param_func in LossParamDosList],
        *[(param_func, TensorLoss) for param_func in LossParamTensorList],
        *[(param_func, PropertyLoss) for param_func in LossParamPropertyList],
    )  # class_param & class
)
class TestLossPT(unittest.TestCase, LossTest, PTTestCase):
    def setUp(self):
        (LossParam, Loss) = self.param[0]
        LossTest.setUp(self)
        self.module_class = Loss
        self.input_dict = LossParam()
        self.key_to_pref_map = self.input_dict.pop("key_to_pref_map")
        self.module = Loss(**self.input_dict)
        self.skip_test_jit = True

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()
