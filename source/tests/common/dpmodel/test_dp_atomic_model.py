# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)

from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDPAtomicModel(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
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

        for atom_excl, pair_excl in itertools.product([[], [1]], [[], [[0, 1]]]):
            md0 = DPAtomicModel(ds, ft, type_map=type_map)
            md0.reinit_atom_exclude(atom_excl)
            md0.reinit_pair_exclude(pair_excl)
            md1 = DPAtomicModel.deserialize(md0.serialize())

            ret0 = md0.forward_common_atomic(self.coord_ext, self.atype_ext, self.nlist)
            ret1 = md1.forward_common_atomic(self.coord_ext, self.atype_ext, self.nlist)

            np.testing.assert_allclose(ret0["energy"], ret1["energy"])
