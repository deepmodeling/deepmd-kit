# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.model.descriptor.hybrid import (
    DescrptHybrid,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.descriptor.se_r import (
    DescrptSeR,
)
from deepmd.pt.utils import (
    env,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


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
            old_impl=False,
        )
        ddsub1 = DescrptSeR(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        dd0 = DescrptHybrid(list=[ddsub0, ddsub1])
        dd1 = DescrptHybrid.deserialize(dd0.serialize())
        dd0 = torch.jit.script(dd0)
        dd1 = torch.jit.script(dd1)
