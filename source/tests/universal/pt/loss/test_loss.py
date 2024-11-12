# SPDX-License-Identifier: LGPL-3.0-or-later
# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from collections import (
    OrderedDict,
)

from deepmd.pt.loss import (
    EnergyStdLoss,
)

from ....consistent.common import (
    parameterize_func,
    parameterized,
)
from ...common.cases.loss.loss import (
    LossTest,
)

# from ...dpmodel.fitting.test_fitting import (
#     FittingParamDipole,
#     FittingParamDos,
#     FittingParamEnergy,
#     FittingParamPolar,
#     FittingParamProperty,
# )
from ..backend import (
    PTTestCase,
)


def LossParamEnergy(
    starter_learning_rate=1.0,
    pref_e=1.0,
    pref_f=1.0,
    pref_v=1.0,
    pref_ae=1.0,
):
    key_to_pref_map = {
        "energy": pref_e,
        "force": pref_f,
        "virial": pref_v,
        "atom_ener": pref_ae,
    }
    input_dict = {
        "key_to_pref_map": key_to_pref_map,
        "starter_learning_rate": starter_learning_rate,
        "start_pref_e": pref_e,
        "limit_pref_e": pref_e / 2,
        "start_pref_f": pref_f,
        "limit_pref_f": pref_f / 2,
        "start_pref_v": pref_v,
        "limit_pref_v": pref_v / 2,
        "start_pref_ae": pref_ae,
        "limit_pref_ae": pref_ae / 2,
    }
    return input_dict


LossParamEnergyList = parameterize_func(
    LossParamEnergy,
    OrderedDict(
        {
            "pref_e": (1.0, 0.0),
            "pref_f": (1.0, 0.0),
            "pref_v": (1.0, 0.0),
            "pref_ae": (1.0, 0.0),
        }
    ),
)
# to get name for the default function
LossParamEnergy = LossParamEnergyList[0]


@parameterized(
    (
        *[(param_func, EnergyStdLoss) for param_func in LossParamEnergyList],
    )  # class_param & class
)
class TestFittingPT(unittest.TestCase, LossTest, PTTestCase):
    def setUp(self):
        (LossParam, Loss) = self.param[0]
        LossTest.setUp(self)
        self.module_class = Loss
        self.input_dict = LossParam()
        self.key_to_pref_map = self.input_dict.pop("key_to_pref_map")
        self.module = Loss(**self.input_dict)
        self.skip_test_jit = True
