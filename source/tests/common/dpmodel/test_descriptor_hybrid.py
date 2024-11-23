# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.descriptor.hybrid import (
    DescrptHybrid,
)
from deepmd.dpmodel.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.dpmodel.descriptor.se_r import (
    DescrptSeR,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDescrptHybrid(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        unittest.TestCase.setUp(self)
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_get_parameters(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)
        ddsub0 = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
        )
        ddsub0.davg = davg
        ddsub0.dstd = dstd
        ddsub1 = DescrptDPA1(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=np.sum(self.sel).item() - 1,
            ntypes=len(self.sel),
        )
        ddsub1.davg = davg[:, :6]
        ddsub1.dstd = dstd[:, :6]
        ddsub2 = DescrptSeR(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth - 0.1,
            sel=[3, 1],
        )
        ddsub2.davg = davg[:, :4, :1]
        ddsub2.dstd = dstd[:, :4, :1]
        em0 = DescrptHybrid(list=[ddsub0, ddsub1, ddsub2])
        self.assertAlmostEqual(em0.get_env_protection(), 0.0)
        self.assertAlmostEqual(em0.get_rcut_smth(), self.rcut_smth - 0.1)
        ddsub3 = DescrptSeR(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth - 0.1,
            sel=[3, 1],
            env_protection=0.1,
        )
        em0 = DescrptHybrid(list=[ddsub0, ddsub1, ddsub3])
        with self.assertRaises(ValueError):
            self.assertAlmostEqual(em0.get_env_protection(), 0.0)

    def test_self_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        ddsub0 = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
        )
        ddsub0.davg = davg
        ddsub0.dstd = dstd
        ddsub1 = DescrptDPA1(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=np.sum(self.sel).item() - 1,
            ntypes=len(self.sel),
        )
        ddsub1.davg = davg[:, :6]
        ddsub1.dstd = dstd[:, :6]
        ddsub2 = DescrptSeR(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth / 2,
            sel=[3, 1],
        )
        ddsub2.davg = davg[:, :4, :1]
        ddsub2.dstd = dstd[:, :4, :1]
        em0 = DescrptHybrid(list=[ddsub0, ddsub1, ddsub2])

        em1 = DescrptHybrid.deserialize(em0.serialize())
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist)
        mm1 = em1.call(self.coord_ext, self.atype_ext, self.nlist)
        for ii in [0, 1]:
            np.testing.assert_allclose(mm0[ii], mm1[ii])
