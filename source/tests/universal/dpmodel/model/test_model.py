# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.descriptor import (
    DescrptDPA1,
    DescrptDPA2,
    DescrptHybrid,
    DescrptSeA,
    DescrptSeR,
    DescrptSeT,
)
from deepmd.dpmodel.fitting import (
    EnergyFittingNet,
)
from deepmd.dpmodel.model import (
    EnergyModel,
)

from ....consistent.common import (
    parameterized,
)
from ...common.cases.model.model import (
    EnerModelTest,
)
from ..backend import (
    DPTestCase,
)
from ..descriptor.test_descriptor import (
    DescriptorParamDPA1,
    DescriptorParamDPA2,
    DescriptorParamHybrid,
    DescriptorParamHybridMixed,
    DescriptorParamSeA,
    DescriptorParamSeR,
    DescriptorParamSeT,
)


@parameterized(
    (
        (DescriptorParamSeA, DescrptSeA),
        (DescriptorParamSeR, DescrptSeR),
        (DescriptorParamSeT, DescrptSeT),
        (DescriptorParamDPA1, DescrptDPA1),
        (DescriptorParamDPA2, DescrptDPA2),
        (DescriptorParamHybrid, DescrptHybrid),
        (DescriptorParamHybridMixed, DescrptHybrid),
    )  # class_param & class
)
class TestEnergyModelDP(unittest.TestCase, EnerModelTest, DPTestCase):
    def setUp(self):
        EnerModelTest.setUp(self)
        (DescriptorParam, Descrpt) = self.param[0]
        # set special precision
        if Descrpt in [DescrptDPA2]:
            self.epsilon_dict["test_smooth"] = 1e-8
        self.input_dict_ds = DescriptorParam(
            len(self.expected_type_map),
            self.expected_rcut,
            self.expected_rcut / 2,
            self.expected_sel,
            self.expected_type_map,
        )
        ds = Descrpt(**self.input_dict_ds)
        ft = EnergyFittingNet(
            **self.input_dict_ft,
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.module = EnergyModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
        self.output_def = self.module.model_output_def().get_data()
        self.expected_has_message_passing = ds.has_message_passing()
        self.skip_test_autodiff = True
