# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import unittest

import numpy as np

from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.exclude_mask import (
    AtomExcludeMask,
    PairExcludeMask,
)

from ...pt.model.test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

torch = importlib.import_module("torch")


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
        mask = des.build_type_exclude_mask(torch.as_tensor(atype, device=env.DEVICE))
        np.testing.assert_equal(mask.detach().cpu().numpy(), expected_mask)


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
            torch.as_tensor(self.nlist, device=env.DEVICE),
            torch.as_tensor(self.atype_ext, device=env.DEVICE),
        )
        np.testing.assert_equal(mask.detach().cpu().numpy(), expected_mask)
