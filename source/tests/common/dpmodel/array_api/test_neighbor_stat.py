# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import array_api_strict as xp

from deepmd.dpmodel.utils.neighbor_stat import (
    NeighborStatOP,
)

from .utils import (
    ArrayAPITest,
)


class TestNeighborStatOP(unittest.TestCase, ArrayAPITest):
    def test_virtual_atoms_are_masked_before_reductions(self) -> None:
        """Virtual-pair masking and neighbor reductions follow the Array API."""
        coord = xp.reshape(
            xp.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ],
                dtype=xp.float64,
            ),
            (1, -1),
        )
        atype = xp.asarray([[0, -1, 0, 1]], dtype=xp.int64)
        expected_min_rr2 = xp.asarray([[1.0, xp.inf, 1.0, 4.0]], dtype=xp.float64)

        for mixed_types in (False, True):
            with self.subTest(mixed_types=mixed_types):
                min_rr2, max_nnei = NeighborStatOP(2, 1.1, mixed_types).call(
                    coord, atype, None
                )
                expected_max_nnei = xp.asarray(
                    [[1]] if mixed_types else [[1, 0]], dtype=xp.int64
                )

                self.assertTrue(bool(xp.all(min_rr2 == expected_min_rr2)))
                self.assertTrue(bool(xp.all(max_nnei == expected_max_nnei)))
                self.assert_namespace_equal(min_rr2, coord)
                self.assert_namespace_equal(max_nnei, atype)
                self.assert_device_equal(min_rr2, coord)
                self.assert_device_equal(max_nnei, atype)
                self.assert_dtype_equal(min_rr2, coord)
                self.assert_dtype_equal(max_nnei, atype)
                self.assertEqual(min_rr2.shape, (1, 4))
                self.assertEqual(max_nnei.shape, expected_max_nnei.shape)
