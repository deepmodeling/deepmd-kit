# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptDPA2,
)

from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDescrptDPA2(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        davg_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd = 0.1 + np.abs(dstd)
        dstd_2 = 0.1 + np.abs(dstd_2)

        em0 = DescrptDPA2(
            ntypes=self.nt,
            repinit_rcut=self.rcut,
            repinit_rcut_smth=self.rcut_smth,
            repinit_nsel=self.sel_mix,
            repformer_rcut=self.rcut / 2,
            repformer_rcut_smth=self.rcut_smth,
            repformer_nsel=nnei // 2,
        )

        em0.repinit.mean = davg
        em0.repinit.stddev = dstd
        em0.repformers.mean = davg_2
        em0.repformers.stddev = dstd_2
        em1 = DescrptDPA2.deserialize(em0.serialize())
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist, self.mapping)
        mm1 = em1.call(self.coord_ext, self.atype_ext, self.nlist, self.mapping)
        for ii in [0, 1, 4]:
            np.testing.assert_allclose(mm0[ii], mm1[ii])
