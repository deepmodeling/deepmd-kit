# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor.exclude_mask import (
    ExcludeMask,
)

from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


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
                [1, 1, 1, 1, 1, 1, 1],
            ]
        ).reshape(self.nf, self.nloc, sum(self.sel))
        des = ExcludeMask(self.nt, exclude_types=exclude_types)
        mask = des.build_type_exclude_mask(
            self.nlist,
            self.atype_ext,
        )
        np.testing.assert_equal(mask, expected_mask)
