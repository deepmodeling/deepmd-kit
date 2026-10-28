# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest.mock import (
    patch,
)

import numpy as np
import torch

from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.exclude_mask import (
    AtomExcludeMask,
    PairExcludeMask,
)

from ...common.test_mixins import (
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
        mask = des.build_type_exclude_mask(torch.as_tensor(atype, device=env.DEVICE))
        np.testing.assert_equal(mask.detach().cpu().numpy(), expected_mask)
        with patch.object(
            torch.Tensor,
            "numpy",
            side_effect=TypeError("direct conversion is unavailable"),
        ):
            numpy_mask = des.build_type_exclude_mask(atype)
        np.testing.assert_equal(numpy_mask, expected_mask)

    def test_type_mask_is_buffer(self) -> None:
        des = AtomExcludeMask(3, exclude_types=[0])
        assert "type_mask" in des.state_dict()


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
        with patch.object(
            torch.Tensor,
            "numpy",
            side_effect=TypeError("direct conversion is unavailable"),
        ):
            numpy_mask = des.build_type_exclude_mask(self.nlist, self.atype_ext)
        np.testing.assert_equal(
            numpy_mask,
            expected_mask,
        )

    def test_build_edge_exclude_mask_with_numpy_inputs(self) -> None:
        des = PairExcludeMask(self.nt, exclude_types=[[0, 1]])
        edge_index = np.array([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=np.int64)
        atype = np.array([0, 1, 0, 0], dtype=np.int32)

        with patch.object(
            torch.Tensor,
            "numpy",
            side_effect=TypeError("direct conversion is unavailable"),
        ):
            numpy_mask = des.build_edge_exclude_mask(edge_index, atype)
        np.testing.assert_equal(numpy_mask, np.array([0, 0, 1, 1], dtype=np.int32))

    def test_type_mask_is_buffer(self) -> None:
        des = PairExcludeMask(self.nt, exclude_types=[[0, 1]])
        assert "type_mask" in des.state_dict()
