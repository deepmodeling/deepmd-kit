# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
from case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)

from deepmd.dpmodel.utils import (
    EnvMat,
)


class TestEnvMat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ):
        rng = np.random.default_rng()
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)
        em0 = EnvMat(self.rcut, self.rcut_smth)
        em1 = EnvMat.deserialize(em0.serialize())
        mm0, ww0 = em0.call(self.coord_ext, self.atype_ext, self.nlist, davg, dstd)
        mm1, ww1 = em1.call(self.coord_ext, self.atype_ext, self.nlist, davg, dstd)
        np.testing.assert_allclose(mm0, mm1)
        np.testing.assert_allclose(ww0, ww1)
