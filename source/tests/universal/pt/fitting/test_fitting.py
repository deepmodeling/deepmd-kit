# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.task import (
    DipoleFittingNet,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFittingNet,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.fitting.fitting import (
    FittingTest,
)
from ...dpmodel.fitting.test_fitting import (
    FittingParamDipole,
    FittingParamDos,
    FittingParamEnergy,
    FittingParamPolar,
)
from ..backend import (
    PTTestCase,
)


@parameterized(
    (
        (FittingParamEnergy, EnergyFittingNet),
        (FittingParamDos, DOSFittingNet),
        (FittingParamDipole, DipoleFittingNet),
        (FittingParamPolar, PolarFittingNet),
    ),  # class_param & class
    (True, False),  # mixed_types
)
class TestFittingPT(unittest.TestCase, FittingTest, PTTestCase):
    def setUp(self):
        ((FittingParam, Fitting), self.mixed_types) = self.param
        FittingTest.setUp(self)
        self.module_class = Fitting
        self.input_dict = FittingParam(
            self.nt, self.dim_descrpt, self.mixed_types, self.dim_embed, ["O", "H"]
        )
        self.module = Fitting(**self.input_dict)
