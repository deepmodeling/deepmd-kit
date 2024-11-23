# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptDPA1,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDescrptDPA1(unittest.TestCase, TestCaseSingleFrameWithNlist):
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

        em0 = DescrptDPA1(self.rcut, self.rcut_smth, self.sel, ntypes=2)
        em0.davg = davg
        em0.dstd = dstd
        em1 = DescrptDPA1.deserialize(em0.serialize())
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist)
        mm1 = em1.call(self.coord_ext, self.atype_ext, self.nlist)
        for ii in [0, 1, 4]:
            np.testing.assert_allclose(mm0[ii], mm1[ii])

    def test_multiple_frames(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        em0 = DescrptDPA1(self.rcut, self.rcut_smth, self.sel, ntypes=2)
        em0.davg = davg
        em0.dstd = dstd
        two_coord_ext = np.concatenate([self.coord_ext, self.coord_ext], axis=0)
        two_atype_ext = np.concatenate([self.atype_ext, self.atype_ext], axis=0)
        two_nlist = np.concatenate([self.nlist, self.nlist], axis=0)

        mm0 = em0.call(two_coord_ext, two_atype_ext, two_nlist)
        for ii in [0, 1, 4]:
            np.testing.assert_allclose(mm0[ii][0], mm0[ii][2], err_msg=f"{ii} 0~2")
            np.testing.assert_allclose(mm0[ii][1], mm0[ii][3], err_msg=f"{ii} 1~3")
