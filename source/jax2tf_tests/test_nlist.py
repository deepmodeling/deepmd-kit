# SPDX-License-Identifier: LGPL-3.0-or-later


import tensorflow as tf

from deepmd.jax.jax2tf.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.jax.jax2tf.region import (
    inter2phys,
)

DTYPE = tf.float64


class TestNeighList(tf.test.TestCase):
    def setUp(self) -> None:
        self.nf = 3
        self.nloc = 3
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = tf.constant(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=DTYPE
        )
        self.icoord = tf.constant([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.1]], dtype=DTYPE)
        self.atype = tf.constant([-1, 0, 1], dtype=tf.int32)
        [self.cell, self.icoord, self.atype] = [
            tf.expand_dims(ii, 0) for ii in [self.cell, self.icoord, self.atype]
        ]
        self.coord = tf.reshape(inter2phys(self.icoord, self.cell), [-1, self.nloc * 3])
        self.cell = tf.reshape(self.cell, [-1, 9])
        [self.cell, self.coord, self.atype] = [
            tf.tile(ii, [self.nf, 1]) for ii in [self.cell, self.coord, self.atype]
        ]
        self.rcut = 1.01
        self.prec = 1e-10
        self.nsel = [10, 10]
        self.ref_nlist = tf.constant(
            [
                [-1] * sum(self.nsel),
                [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1],
            ],
            dtype=tf.int64,
        )

    def test_build_notype(self) -> None:
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel),
            distinguish_types=False,
        )
        self.assertAllClose(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = tf.gather(mapping[0], tf.where(nlist_mask, 0, nlist[0]))
        nlist_loc = tf.where(
            nlist_mask,
            tf.fill(tf.shape(nlist_loc), tf.cast(-1, nlist_loc.dtype)),
            nlist_loc,
        )
        self.assertAllClose(
            tf.sort(nlist_loc, axis=-1),
            tf.sort(self.ref_nlist, axis=-1),
        )

    def test_build_type(self) -> None:
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            self.nsel,
            distinguish_types=True,
        )
        self.assertAllClose(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = tf.gather(mapping[0], tf.where(nlist_mask, 0, nlist[0]))
        nlist_loc = tf.where(
            nlist_mask,
            tf.fill(tf.shape(nlist_loc), tf.cast(-1, nlist_loc.dtype)),
            nlist_loc,
        )
        for ii in range(2):
            self.assertAllClose(
                tf.sort(tf.split(nlist_loc, self.nsel, axis=-1)[ii], axis=-1),
                tf.sort(tf.split(self.ref_nlist, self.nsel, axis=-1)[ii], axis=-1),
            )

    def test_extend_coord(self) -> None:
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        # expected ncopy x nloc
        self.assertEqual(list(ecoord.shape), [self.nf, self.nall * 3])
        self.assertEqual(list(eatype.shape), [self.nf, self.nall])
        self.assertEqual(list(mapping.shape), [self.nf, self.nall])
        # check the nloc part is identical with original coord
        self.assertAllClose(
            ecoord[:, : self.nloc * 3], self.coord, rtol=self.prec, atol=self.prec
        )
        # check the shift vectors are aligned with grid
        shift_vec = (
            tf.reshape(ecoord, [-1, self.ns, self.nloc, 3])
            - tf.reshape(self.coord, [-1, self.nloc, 3])[:, None, :, :]
        )
        shift_vec = tf.reshape(shift_vec, [-1, self.nall, 3])
        # hack!!! assumes identical cell across frames
        shift_vec = tf.matmul(
            shift_vec, tf.linalg.inv(tf.reshape(self.cell, [self.nf, 3, 3])[0])
        )
        # nf x nall x 3
        shift_vec = tf.round(shift_vec)
        # check: identical shift vecs
        self.assertAllClose(shift_vec[0], shift_vec[1], rtol=self.prec, atol=self.prec)
        # check: shift idx aligned with grid
        mm, _, cc = tf.unique_with_counts(shift_vec[0][:, 0])
        self.assertAllClose(
            tf.sort(mm),
            tf.constant([-2, -1, 0, 1, 2], dtype=DTYPE),
            rtol=self.prec,
            atol=self.prec,
        )
        self.assertAllClose(
            cc,
            tf.constant([self.ns * self.nloc // 5] * 5, dtype=tf.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, _, cc = tf.unique_with_counts(shift_vec[1][:, 1])
        self.assertAllClose(
            tf.sort(mm),
            tf.constant([-2, -1, 0, 1, 2], dtype=DTYPE),
            rtol=self.prec,
            atol=self.prec,
        )
        self.assertAllClose(
            cc,
            tf.constant([self.ns * self.nloc // 5] * 5, dtype=tf.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, _, cc = tf.unique_with_counts(shift_vec[1][:, 2])
        self.assertAllClose(
            tf.sort(mm),
            tf.constant([-1, 0, 1], dtype=DTYPE),
            rtol=self.prec,
            atol=self.prec,
        )
        self.assertAllClose(
            cc,
            tf.constant([self.ns * self.nloc // 3] * 3, dtype=tf.int32),
            rtol=self.prec,
            atol=self.prec,
        )
