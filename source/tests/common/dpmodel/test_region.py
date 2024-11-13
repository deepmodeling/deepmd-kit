# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils import (
    inter2phys,
    to_face_distance,
)

from ...seed import (
    GLOBAL_SEED,
)


class TestRegion(unittest.TestCase):
    def setUp(self) -> None:
        self.cell = np.array(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]],
        )
        self.cell = np.reshape(self.cell, [1, 1, -1, 3])
        self.cell = np.tile(self.cell, [4, 5, 1, 1])
        self.prec = 1e-8

    def test_inter_to_phys(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        inter = rng.normal(size=[4, 5, 3, 3])
        phys = inter2phys(inter, self.cell)
        for ii in range(4):
            for jj in range(5):
                expected_phys = np.matmul(inter[ii, jj], self.cell[ii, jj])
                np.testing.assert_allclose(
                    phys[ii, jj], expected_phys, rtol=self.prec, atol=self.prec
                )

    def test_to_face_dist(self) -> None:
        cell0 = self.cell[0][0]
        vol = np.linalg.det(cell0)
        # area of surfaces xy, xz, yz
        sxy = np.linalg.norm(np.cross(cell0[0], cell0[1]))
        sxz = np.linalg.norm(np.cross(cell0[0], cell0[2]))
        syz = np.linalg.norm(np.cross(cell0[1], cell0[2]))
        # vol / area gives distance
        dz = vol / sxy
        dy = vol / sxz
        dx = vol / syz
        expected = np.array([dx, dy, dz])
        dists = to_face_distance(self.cell)
        for ii in range(4):
            for jj in range(5):
                np.testing.assert_allclose(
                    dists[ii][jj], expected, rtol=self.prec, atol=self.prec
                )
