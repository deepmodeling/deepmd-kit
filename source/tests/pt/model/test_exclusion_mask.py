# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.pt.model.descriptor.se_a import (
    DescrptBlockSeA,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


# to be merged with the tf test case
class TestExcludeMask(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_build_type_exclude_mask(self):
        exclude_types = [[0, 1]]
        expected_mask = np.array(
            [
                [1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 0, 1],
                [0, 0, 1, 1, 1, 1, 1],
            ]
        ).reshape(self.nf, self.nloc, sum(self.sel))
        des = DescrptBlockSeA(
            self.rcut, self.rcut_smth, self.sel, exclude_types=exclude_types
        )
        mask = des.build_type_exclude_mask(
            to_torch_tensor(self.nlist),
            to_torch_tensor(self.atype_ext),
        )
        np.testing.assert_equal(to_numpy_array(mask), expected_mask)
