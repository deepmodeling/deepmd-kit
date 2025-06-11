# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptDPA3,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
    DescrptSeTTebd,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.descriptor.descriptor import (
    DescriptorTest,
)
from ...dpmodel.descriptor.test_descriptor import (
    DescriptorParamDPA1,
    DescriptorParamDPA2,
    DescriptorParamDPA3,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamSeA,
    DescriptorParamSeR,
    DescriptorParamSeT,
    DescriptorParamSeTTebd,
)
from ..backend import (
    PTTestCase,
)


@parameterized(
    (
        (DescriptorParamSeA, DescrptSeA),
        (DescriptorParamSeR, DescrptSeR),
        (DescriptorParamSeT, DescrptSeT),
        (DescriptorParamSeTTebd, DescrptSeTTebd),
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamDPA3, DescrptDPA3),
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    )  # class_param & class
)
class TestDescriptorPT(unittest.TestCase, DescriptorTest, PTTestCase):
    def setUp(self) -> None:
        DescriptorTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        self.module_class = Descrpt
        self.input_dict = DescriptorParam(
            self.nt, self.rcut, self.rcut_smth, self.sel, ["O", "H"]
        )
        self.module = Descrpt(**self.input_dict)

    @classmethod
    def tearDownClass(cls) -> None:
        PTTestCase.tearDownClass()
