# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import paddle

from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.preprocess import (
    Region3D,
)
from deepmd.pd.utils.region import (
    inter2phys,
    to_face_distance,
)

from ...seed import (
    GLOBAL_SEED,
)

dtype = paddle.float64


class TestRegion(unittest.TestCase):
    def setUp(self):
        self.cell = paddle.to_tensor(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype, place="cpu"
        )
        self.cell = self.cell.unsqueeze(0).unsqueeze(0)
        self.cell = paddle.tile(self.cell, [4, 5, 1, 1])
        self.prec = 1e-8

    def test_inter_to_phys(self):
        generator = paddle.seed(GLOBAL_SEED)
        inter = paddle.rand([4, 5, 3, 3], dtype=dtype).to(device="cpu")
        phys = inter2phys(inter, self.cell)
        for ii in range(4):
            for jj in range(5):
                expected_phys = paddle.matmul(inter[ii, jj], self.cell[ii, jj])
                assert paddle.allclose(
                    phys[ii, jj], expected_phys, rtol=self.prec, atol=self.prec
                )

    def test_to_face_dist(self):
        cell0 = self.cell[0][0].numpy()
        vol = np.linalg.det(cell0)
        # area of surfaces xy, xz, yz
        sxy = np.linalg.norm(np.cross(cell0[0], cell0[1]))
        sxz = np.linalg.norm(np.cross(cell0[0], cell0[2]))
        syz = np.linalg.norm(np.cross(cell0[1], cell0[2]))
        # vol / area gives distance
        dz = vol / sxy
        dy = vol / sxz
        dx = vol / syz
        dists = to_face_distance(self.cell)
        expected = paddle.to_tensor([dx, dy, dz], dtype=dists.dtype).to(device="cpu")
        for ii in range(4):
            for jj in range(5):
                assert paddle.allclose(
                    dists[ii][jj], expected, rtol=self.prec, atol=self.prec
                )


class TestLegacyRegion(unittest.TestCase):
    def setUp(self):
        self.cell = paddle.to_tensor(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype, place=env.DEVICE
        )
        self.prec = 1e-6

    def test_inter_to_phys(self):
        generator = paddle.seed(GLOBAL_SEED)
        inter = paddle.rand([3, 3], dtype=dtype).to(device=env.DEVICE)
        reg = Region3D(self.cell)
        phys = reg.inter2phys(inter)
        expected_phys = paddle.matmul(inter, self.cell)
        assert paddle.allclose(phys, expected_phys, rtol=self.prec, atol=self.prec)

    def test_inter_to_inter(self):
        generator = paddle.seed(GLOBAL_SEED)
        inter = paddle.rand([3, 3], dtype=dtype).to(device=env.DEVICE)
        reg = Region3D(self.cell)
        new_inter = reg.phys2inter(reg.inter2phys(inter))
        assert paddle.allclose(inter, new_inter, rtol=self.prec, atol=self.prec)

    def test_to_face_dist(self):
        pass
