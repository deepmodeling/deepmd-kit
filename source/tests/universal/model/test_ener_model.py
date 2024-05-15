# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

from deepmd.dpmodel.descriptor.se_e2_a import DescrptSeA as DescrptSeADP
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.dpmodel.model.ener_model import EnergyModel as EnergyModelDP

from ..utils import (
    INSTALLED_PT,
)

if INSTALLED_PT:
    from deepmd.pt.model.model.ener_model import (
        EnergyModel as EnergyModelPT,
    )

from .utils import (
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
        ds = DescrptSeADP(
            rcut=self.expected_rcut,
            rcut_smth=self.expected_rcut / 2,
            sel=self.expected_sel,
        )
        ft = EnergyFittingNetDP(
            ntypes=len(self.expected_type_map),
            dim_descrpt=ds.get_dim_out(),
            mixed_types=ds.mixed_types(),
        )
        self.dp_module = EnergyModelDP(
            ds,
            ft,
            type_map=self.expected_type_map,
        )

        if INSTALLED_PT:
            self.pt_module = EnergyModelPT.deserialize(self.dp_module.serialize())
