# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.ener_model import (
    EnergyModel,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
)

from ...common.cases.model.ener_model import (
    EnerModelTest,
)
from ..backend import (
    PTTestCase,
)


class TestEnergyModelDP(unittest.TestCase, EnerModelTest, PTTestCase):
    @property
    def modules_to_test(self):
        # for Model, we can test script module API
        modules = [
            *PTTestCase.modules_to_test.fget(self),
            self.script_module,
        ]
        return modules

    def setUp(self):
        EnerModelTest.setUp(self)
        ds = DescrptSeA(
            rcut=self.expected_rcut,
            rcut_smth=self.expected_rcut / 2,
            sel=self.expected_sel,
        )
        ft = EnergyFittingNet(
            ntypes=len(self.expected_type_map),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.module = EnergyModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
