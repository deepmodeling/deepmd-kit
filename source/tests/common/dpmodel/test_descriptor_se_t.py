# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDescrptSeT(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        em0 = DescrptSeT(self.rcut, self.rcut_smth, self.sel)
        em0.davg = davg
        em0.dstd = dstd
        em1 = DescrptSeT.deserialize(em0.serialize())
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist)
        mm1 = em1.call(self.coord_ext, self.atype_ext, self.nlist)
        for ii in [0, 4]:
            np.testing.assert_allclose(mm0[ii], mm1[ii])
