# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.fitting import (
    DipoleFitting,
    DOSFittingNet,
    EnergyFittingNet,
    PolarFitting,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.fitting.fitting import (
    FittingTest,
)
from ..backend import (
    DPTestCase,
)


@parameterized(
    (True, False),  # mixed_types
)
class TestFittingEnergyDP(unittest.TestCase, FittingTest, DPTestCase):
    def setUp(self):
        (self.mixed_types,) = self.param
        FittingTest.setUp(self)
        self.module_class = EnergyFittingNet
        self.module = EnergyFittingNet(**self.input_dict)


@parameterized(
    (True, False),  # mixed_types
)
class TestFittingDosDP(unittest.TestCase, FittingTest, DPTestCase):
    def setUp(self):
        (self.mixed_types,) = self.param
        FittingTest.setUp(self)
        self.module_class = DOSFittingNet
        self.module = DOSFittingNet(**self.input_dict)


@parameterized(
    (True, False),  # mixed_types
)
class TestFittingDipoleDP(unittest.TestCase, FittingTest, DPTestCase):
    def setUp(self):
        (self.mixed_types,) = self.param
        FittingTest.setUp(self)
        self.input_dict.update({"embedding_width": self.dim_embed})
        self.module_class = DipoleFitting
        self.module = DipoleFitting(**self.input_dict)


@parameterized(
    (True, False),  # mixed_types
)
class TestFittingPolarDP(unittest.TestCase, FittingTest, DPTestCase):
    def setUp(self):
        (self.mixed_types,) = self.param
        FittingTest.setUp(self)
        self.input_dict.update({"embedding_width": self.dim_embed})
        self.module_class = PolarFitting
        self.module = PolarFitting(**self.input_dict)
