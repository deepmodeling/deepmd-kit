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
    TestCaseSingleFrameWithNlistWithVirtual,
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


class TestEnvMatWithVirtualCenter(
    unittest.TestCase, TestCaseSingleFrameWithNlistWithVirtual
):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlistWithVirtual.setUp(self)

    def test_normalization_keeps_virtual_centers_zero(self) -> None:
        """Virtual centers must not borrow normalization data from a real type."""
        nf, nloc, nnei = self.nlist.shape
        virtual_center = self.atype_ext[:, :nloc] < 0

        for radial_only, width in ((False, 4), (True, 1)):
            with self.subTest(radial_only=radial_only):
                # Nonzero values make accidental ``-1`` indexing observable: NumPy
                # would otherwise select the final real-type row silently.
                davg = np.arange(1, self.nt * nnei * width + 1, dtype=np.float64)
                davg = davg.reshape(self.nt, nnei, width)
                dstd = np.full_like(davg, 2.0)

                env_mat, diff, switch = EnvMat(self.rcut, self.rcut_smth).call(
                    self.coord_ext,
                    self.atype_ext,
                    self.nlist,
                    davg,
                    dstd,
                    radial_only=radial_only,
                )

                np.testing.assert_allclose(env_mat[virtual_center], 0.0)
                np.testing.assert_allclose(diff[virtual_center], 0.0)
                np.testing.assert_allclose(switch[virtual_center], 0.0)
