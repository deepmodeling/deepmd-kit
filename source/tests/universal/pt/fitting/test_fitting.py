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
from ..backend import (
    PTTestCase,
)


@parameterized(
    (True, False),  # mixed_types
)
class TestFittingEnergyPT(unittest.TestCase, FittingTest, PTTestCase):
    def setUp(self):
        (self.mixed_types,) = self.param
        FittingTest.setUp(self)
        self.module_class = EnergyFittingNet
        self.module = EnergyFittingNet(**self.input_dict)


@parameterized(
    (True, False),  # mixed_types
)
class TestFittingDosPT(unittest.TestCase, FittingTest, PTTestCase):
    def setUp(self):
        (self.mixed_types,) = self.param
        FittingTest.setUp(self)
        self.module_class = DOSFittingNet
        self.module = DOSFittingNet(**self.input_dict)


@parameterized(
    (True, False),  # mixed_types
)
class TestFittingDipolePT(unittest.TestCase, FittingTest, PTTestCase):
    def setUp(self):
        (self.mixed_types,) = self.param
        FittingTest.setUp(self)
        self.input_dict.update({"embedding_width": self.dim_embed})
        self.module_class = DipoleFittingNet
        self.module = DipoleFittingNet(**self.input_dict)


@parameterized(
    (True, False),  # mixed_types
)
class TestFittingPolarPT(unittest.TestCase, FittingTest, PTTestCase):
    def setUp(self):
        (self.mixed_types,) = self.param
        FittingTest.setUp(self)
        self.input_dict.update({"embedding_width": self.dim_embed})
        self.module_class = PolarFittingNet
        self.module = PolarFittingNet(**self.input_dict)
