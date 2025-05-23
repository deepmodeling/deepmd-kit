# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.task import (
    DipoleFittingNet,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFittingNet,
    PropertyFittingNet,
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
    FittingParamProperty,
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
        (FittingParamProperty, PropertyFittingNet),
    ),  # class_param & class
    (True, False),  # mixed_types
)
class TestFittingPT(unittest.TestCase, FittingTest, PTTestCase):
    def setUp(self) -> None:
        ((FittingParam, Fitting), self.mixed_types) = self.param
        FittingTest.setUp(self)
        self.module_class = Fitting
        self.input_dict = FittingParam(
            self.nt,
            self.dim_descrpt,
            self.mixed_types,
            ["O", "H"],
            embedding_width=self.dim_embed,
        )
        self.module = Fitting(**self.input_dict)

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()
