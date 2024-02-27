# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model import (
    DPModel,
)

from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDPModel(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]
        md0 = DPModel(ds, ft, type_map=type_map)
        md1 = DPModel.deserialize(md0.serialize())

        ret0 = md0.call_lower(self.coord_ext, self.atype_ext, self.nlist)
        ret1 = md1.call_lower(self.coord_ext, self.atype_ext, self.nlist)

        np.testing.assert_allclose(ret0["energy"], ret1["energy"])
        np.testing.assert_allclose(ret0["energy_redu"], ret1["energy_redu"])
