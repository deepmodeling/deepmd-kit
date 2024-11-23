# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptDPA2,
)
from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)

from ...seed import (
    GLOBAL_SEED,
)
from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestDescrptDPA2(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_self_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        davg_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd = 0.1 + np.abs(dstd)
        dstd_2 = 0.1 + np.abs(dstd_2)

        repinit = RepinitArgs(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            nsel=self.sel_mix,
        )
        repformer = RepformerArgs(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth,
            nsel=nnei // 2,
        )

        em0 = DescrptDPA2(
            ntypes=self.nt,
            repinit=repinit,
            repformer=repformer,
        )

        em0.repinit.mean = davg
        em0.repinit.stddev = dstd
        em0.repformers.mean = davg_2
        em0.repformers.stddev = dstd_2
        em1 = DescrptDPA2.deserialize(em0.serialize())
        mm0 = em0.call(self.coord_ext, self.atype_ext, self.nlist, self.mapping)
        mm1 = em1.call(self.coord_ext, self.atype_ext, self.nlist, self.mapping)
        desired_shape = [
            (nf, nloc, em0.get_dim_out()),  # descriptor
            (nf, nloc, em0.get_dim_emb(), 3),  # rot_mat
            (nf, nloc, nnei // 2, em0.repformers.g2_dim),  # g2
            (nf, nloc, nnei // 2, 3),  # h2
            (nf, nloc, nnei // 2),  # sw
        ]
        for ii in [0, 1, 2, 3, 4]:
            np.testing.assert_equal(mm0[ii].shape, desired_shape[ii])
            np.testing.assert_allclose(mm0[ii], mm1[ii])
