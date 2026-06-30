# SPDX-License-Identifier: LGPL-3.0-or-later


import tensorflow as tf

from deepmd.jax.jax2tf.region import (
    inter2phys,
    to_face_distance,
)

GLOBAL_SEED = 20241109
DTYPE = tf.float64


class TestRegion(tf.test.TestCase):
    def setUp(self) -> None:
        self.cell = tf.constant(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=DTYPE
        )
        self.cell = tf.reshape(self.cell, [1, 1, -1, 3])
        self.cell = tf.tile(self.cell, [4, 5, 1, 1])
        self.prec = 1e-8

    def test_inter_to_phys(self) -> None:
        rng = tf.random.Generator.from_seed(GLOBAL_SEED)
        inter = rng.normal(shape=[4, 5, 3, 3], dtype=DTYPE)
        phys = inter2phys(inter, self.cell)
        for ii in range(4):
            for jj in range(5):
                expected_phys = tf.matmul(inter[ii, jj], self.cell[ii, jj])
                self.assertAllClose(
                    phys[ii, jj], expected_phys, rtol=self.prec, atol=self.prec
                )

    def test_to_face_dist(self) -> None:
        cell0 = self.cell[0][0]
        vol = tf.linalg.det(cell0)
        # area of surfaces xy, xz, yz
        sxy = tf.linalg.norm(tf.linalg.cross(cell0[0], cell0[1]))
        sxz = tf.linalg.norm(tf.linalg.cross(cell0[0], cell0[2]))
        syz = tf.linalg.norm(tf.linalg.cross(cell0[1], cell0[2]))
        # vol / area gives distance
        dz = vol / sxy
        dy = vol / sxz
        dx = vol / syz
        expected = tf.stack([dx, dy, dz])
        dists = to_face_distance(self.cell)
        for ii in range(4):
            for jj in range(5):
                self.assertAllClose(
                    dists[ii][jj], expected, rtol=self.prec, atol=self.prec
                )
