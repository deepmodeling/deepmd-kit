# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.exclude_mask import (
    AtomExcludeMask,
    PairExcludeMask,
)

from .case_single_frame_with_nlist import (
    TestCaseSingleFrameWithNlist,
)


class TestAtomExcludeMask(unittest.TestCase):
    def test_build_type_exclude_mask(self) -> None:
        nf = 2
        nt = 3
        exclude_types = [0, 2]
        atype = np.array(
            [
                [0, 2, 1, 2, 0, 1, 0],
                [1, 2, 0, 0, 2, 2, 1],
            ],
            dtype=np.int32,
        ).reshape([nf, -1])
        expected_mask = np.array(
            [
                [0, 0, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0, 1],
            ]
        ).reshape([nf, -1])
        des = AtomExcludeMask(nt, exclude_types=exclude_types)
        mask = des.build_type_exclude_mask(atype)
        np.testing.assert_equal(mask, expected_mask)


# to be merged with the tf test case
class TestPairExcludeMask(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_build_type_exclude_mask(self) -> None:
        exclude_types = [[0, 1]]
        expected_mask = np.array(
            [
                [1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 0, 1],
                [0, 0, 1, 1, 1, 1, 1],
                [0, 0, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 0, 1],
            ]
        ).reshape(self.nf, self.nloc, sum(self.sel))
        des = PairExcludeMask(self.nt, exclude_types=exclude_types)
        mask = des.build_type_exclude_mask(
            self.nlist,
            self.atype_ext,
        )
        np.testing.assert_equal(mask, expected_mask)
