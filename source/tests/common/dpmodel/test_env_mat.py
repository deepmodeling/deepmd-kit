# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils import (
    EnvMat,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestEnvMat(unittest.TestCase, TestCaseSingleFrameWithNlist):
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
        em0 = EnvMat(self.rcut, self.rcut_smth)
        em1 = EnvMat.deserialize(em0.serialize())
        mm0, diff0, ww0 = em0.call(
            self.coord_ext, self.atype_ext, self.nlist, davg, dstd
        )
        mm1, diff1, ww1 = em1.call(
            self.coord_ext, self.atype_ext, self.nlist, davg, dstd
        )
        np.testing.assert_allclose(mm0, mm1)
        np.testing.assert_allclose(diff0, diff1)
        np.testing.assert_allclose(ww0, ww1)
