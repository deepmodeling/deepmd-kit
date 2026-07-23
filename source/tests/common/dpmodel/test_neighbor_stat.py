# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils.neighbor_stat import (
    NeighborStatOP,
)


class TestNeighborStatOP(unittest.TestCase):
    def test_virtual_atoms_do_not_affect_statistics(self) -> None:
        """Ignore virtual atoms as both neighbor-stat centers and neighbors."""
        # Atom 1 is virtual and overlaps atom 0.  Without a neighbor mask it
        # drives the minimum distance to zero; without a center mask it sees both
        # type-0 atoms and inflates their maximum neighbor count from one to two.
        coord = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ).reshape(1, -1)
        atype = np.array([[0, -1, 0, 1]], dtype=np.int64)
        expected_min_rr2 = np.array([[1.0, np.inf, 1.0, 4.0]])

        for cell in (None, 10.0 * np.eye(3).reshape(1, 9)):
            for mixed_types in (False, True):
                with self.subTest(cell=cell is not None, mixed_types=mixed_types):
                    min_rr2, max_nnei = NeighborStatOP(
                        ntypes=2,
                        rcut=1.1,
                        mixed_types=mixed_types,
                    ).call(coord, atype, cell)

                    np.testing.assert_allclose(min_rr2, expected_min_rr2)
                    expected_max_nnei = [[1]] if mixed_types else [[1, 0]]
                    np.testing.assert_array_equal(max_nnei, expected_max_nnei)
