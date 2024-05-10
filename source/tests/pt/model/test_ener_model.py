# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.pt.model.atomic_model.energy_atomic_model import (
    DPEnergyAtomicModel,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.ener_model import (
    EnergyModel,
)
from deepmd.pt.model.task.ener import (
    EnergyFittingNet,
)

from .utils import (
    AtomicModelTestCase,
    ModelTestCase,
)


class TestEnerModel(unittest.TestCase, ModelTestCase):
    def setUp(self) -> None:
        self.expected_rcut = 5.0
        self.expected_type_map = ["foo", "bar"]
        self.expected_dim_fparam = 0
        self.expected_dim_aparam = 0
        self.expected_sel_type = [0, 1]
        self.expected_aparam_nall = False
        self.expected_model_output_type = ["energy", "mask"]
        self.expected_sel = [8, 12]
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
        self.module = EnergyModel(ds, ft, type_map=self.expected_type_map)


class TestEnerAtomicModel(unittest.TestCase, AtomicModelTestCase):
    def setUp(self) -> None:
        self.expected_rcut = 5.0
        self.expected_type_map = ["foo", "bar"]
        self.expected_dim_fparam = 0
        self.expected_dim_aparam = 0
        self.expected_sel_type = [0, 1]
        self.expected_aparam_nall = False
        self.expected_model_output_type = ["energy", "mask"]
        self.expected_sel = [8, 12]
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
        self.module = DPEnergyAtomicModel(ds, ft, type_map=self.expected_type_map)
