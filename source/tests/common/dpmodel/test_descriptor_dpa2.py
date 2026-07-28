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
from deepmd.dpmodel.descriptor.repformers import (
    DescrptBlockRepformers,
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
        # this test exercises the legacy dense body's full 5-tuple (real
        # g2/h2, not the thinner dense-ABI adapter's `None`/`None`); force
        # the dense route explicitly rather than relying on the
        # graph-eligibility of this particular config (`uses_graph_lower`
        # would otherwise route `.call()` through `_call_graph_adapter`,
        # see DescrptDPA2.uses_graph_lower).
        em0.disable_graph_lower()
        em1 = DescrptDPA2.deserialize(em0.serialize())
        em1.disable_graph_lower()
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


class TestDescrptBlockRepformersAccessors(unittest.TestCase):
    def test_get_rcut_smth(self) -> None:
        block = DescrptBlockRepformers(
            rcut=6.0,
            rcut_smth=5.0,
            sel=40,
            ntypes=2,
            nlayers=3,
        )
        self.assertEqual(block.get_rcut_smth(), 5.0)

    def test_get_env_protection(self) -> None:
        block = DescrptBlockRepformers(
            rcut=6.0,
            rcut_smth=5.0,
            sel=40,
            ntypes=2,
            nlayers=3,
            env_protection=1.0,
        )
        self.assertEqual(block.get_env_protection(), 1.0)

    def test_get_env_protection_default(self) -> None:
        block = DescrptBlockRepformers(
            rcut=6.0,
            rcut_smth=5.0,
            sel=40,
            ntypes=2,
            nlayers=3,
        )
        self.assertEqual(block.get_env_protection(), 0.0)
