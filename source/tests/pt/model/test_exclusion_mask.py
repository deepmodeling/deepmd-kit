# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.exclude_mask import (
    AtomExcludeMask,
    PairExcludeMask,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


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
        mask = des(to_torch_tensor(atype))
        np.testing.assert_equal(to_numpy_array(mask), expected_mask)

    def test_masks_move_with_the_module(self) -> None:
        """The masks follow ``module.to`` and stay out of the state dict."""
        for des in (
            AtomExcludeMask(3, exclude_types=[0, 2]),
            PairExcludeMask(3, exclude_types=[(0, 1)]),
        ):
            self.assertEqual(des.type_mask.device.type, env.DEVICE.type)
            des.to("cpu")
            self.assertEqual(des.type_mask.device.type, "cpu")
            self.assertEqual(des.state_dict(), {})
            # a re-initialization replaces the mask in place
            des.reinit(3, [])
            self.assertEqual(des.state_dict(), {})
            self.assertTrue(bool(des.type_mask.all()))


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
        des = PairExcludeMask(self.nt, exclude_types=exclude_types).to(env.DEVICE)
        mask = des(
            to_torch_tensor(self.nlist),
            to_torch_tensor(self.atype_ext),
        )
        np.testing.assert_equal(to_numpy_array(mask), expected_mask)
