# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import paddle

from deepmd.pd.model.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.pd.model.descriptor.hybrid import (
    DescrptHybrid,
)
from deepmd.pd.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pd.model.descriptor.se_r import (
    DescrptSeR,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.utils import (
    to_paddle_tensor,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PD_FLOAT_PRECISION


class TestDescrptHybrid(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_jit(
        self,
    ):
        ddsub0 = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ddsub1 = DescrptSeR(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        dd0 = DescrptHybrid(list=[ddsub0, ddsub1])
        dd1 = DescrptHybrid.deserialize(dd0.serialize())
        dd0 = paddle.jit.to_static(dd0)
        dd1 = paddle.jit.to_static(dd1)

    def test_get_parameters(
        self,
    ):
        nf, nloc, nnei = self.nlist.shape
        ddsub0 = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
        )
        ddsub1 = DescrptDPA1(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=np.sum(self.sel).item() - 1,
            ntypes=len(self.sel),
        )
        ddsub2 = DescrptSeR(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth - 0.1,
            sel=[3, 1],
        )
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

    def test_hybrid_mixed_and_no_mixed(self):
        coord_ext = to_paddle_tensor(self.coord_ext)
        atype_ext = to_paddle_tensor(self.atype_ext)
        nlist1 = to_paddle_tensor(self.nlist)
        nlist2 = to_paddle_tensor(-np.sort(-self.nlist, axis=-1))
        ddsub0 = DescrptSeA(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=self.sel,
        )
        ddsub1 = DescrptDPA1(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            sel=np.sum(self.sel).item() - 1,
            ntypes=len(self.sel),
        )
        ddsub2 = DescrptSeR(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth,
            sel=[3, 1],
        )
        dd = DescrptHybrid(list=[ddsub0, ddsub1, ddsub2])
        ret = dd(
            coord_ext,
            atype_ext,
            nlist2,
        )
        ret0 = ddsub0(
            coord_ext,
            atype_ext,
            nlist1,
        )
        ret1 = ddsub1(coord_ext, atype_ext, nlist2[:, :, :-1])
        ret2 = ddsub2(coord_ext, atype_ext, nlist1[:, :, [0, 1, 2, self.sel[0]]])
        assert paddle.allclose(
            ret[0],
            paddle.concat([ret0[0], ret1[0], ret2[0]], axis=2),
        )
