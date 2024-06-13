# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting.ener_fitting import (
    EnergyFittingNet,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)

from ...common.cases.model.ener_model import (
    EnerModelTest,
)
from ..backend import (
    DPTestCase,
)


class TestEnergyModelDP(unittest.TestCase, EnerModelTest, DPTestCase):
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
