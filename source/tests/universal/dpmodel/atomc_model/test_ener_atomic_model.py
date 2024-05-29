# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.atomic_model.dp_atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting.ener_fitting import (
    EnergyFittingNet,
)

from ...common.cases.atomic_model.ener_model import (
    EnerAtomicModelTest,
)
from ..backend import (
    DPTestCase,
)


class TestEnergyAtomicModelDP(unittest.TestCase, EnerAtomicModelTest, DPTestCase):
    def setUp(self):
        EnerAtomicModelTest.setUp(self)
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
        self.module = DPAtomicModel(
            ds,
            ft,
            type_map=self.expected_type_map,
        )
