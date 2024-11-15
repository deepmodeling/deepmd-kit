# SPDX-License-Identifier: LGPL-3.0-or-later


import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from deepmd.jax.jax2tf.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.jax.jax2tf.region import (
    inter2phys,
)

dtype = tnp.float64


class TestNeighList(tf.test.TestCase):
    def setUp(self) -> None:
        self.nf = 3
        self.nloc = 3
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = tnp.array([[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype)
        self.icoord = tnp.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.1]], dtype=dtype)
        self.atype = tnp.array([-1, 0, 1], dtype=tnp.int32)
        [self.cell, self.icoord, self.atype] = [
            tnp.expand_dims(ii, 0) for ii in [self.cell, self.icoord, self.atype]
        ]
        self.coord = inter2phys(self.icoord, self.cell).reshape([-1, self.nloc * 3])
        self.cell = self.cell.reshape([-1, 9])
        [self.cell, self.coord, self.atype] = [
            tnp.tile(ii, [self.nf, 1]) for ii in [self.cell, self.coord, self.atype]
        ]
        self.rcut = 1.01
        self.prec = 1e-10
        self.nsel = [10, 10]
        self.ref_nlist = tnp.array(
            [
                [-1] * sum(self.nsel),
                [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1],
            ]
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
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc = tnp.where(nlist_mask, tnp.full_like(nlist_loc, -1), nlist_loc)
        self.assertAllClose(
            tnp.sort(nlist_loc, axis=-1),
            tnp.sort(self.ref_nlist, axis=-1),
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
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc = tnp.where(nlist_mask, tnp.full_like(nlist_loc, -1), nlist_loc)
        for ii in range(2):
            self.assertAllClose(
                tnp.sort(tnp.split(nlist_loc, self.nsel, axis=-1)[ii], axis=-1),
                tnp.sort(tnp.split(self.ref_nlist, self.nsel, axis=-1)[ii], axis=-1),
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
            ecoord.reshape([-1, self.ns, self.nloc, 3])
            - self.coord.reshape([-1, self.nloc, 3])[:, None, :, :]
        )
        shift_vec = shift_vec.reshape([-1, self.nall, 3])
        # hack!!! assumes identical cell across frames
        shift_vec = tnp.matmul(
            shift_vec, tf.linalg.inv(self.cell.reshape([self.nf, 3, 3])[0])
        )
        # nf x nall x 3
        shift_vec = tnp.round(shift_vec)
        # check: identical shift vecs
        self.assertAllClose(shift_vec[0], shift_vec[1], rtol=self.prec, atol=self.prec)
        # check: shift idx aligned with grid
        mm, _, cc = tf.unique_with_counts(shift_vec[0][:, 0])
        self.assertAllClose(
            tnp.sort(mm),
            tnp.array([-2, -1, 0, 1, 2], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        self.assertAllClose(
            cc,
            tnp.array([self.ns * self.nloc // 5] * 5, dtype=tnp.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, _, cc = tf.unique_with_counts(shift_vec[1][:, 1])
        self.assertAllClose(
            tnp.sort(mm),
            tnp.array([-2, -1, 0, 1, 2], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        self.assertAllClose(
            cc,
            tnp.array([self.ns * self.nloc // 5] * 5, dtype=tnp.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, _, cc = tf.unique_with_counts(shift_vec[1][:, 2])
        self.assertAllClose(
            tnp.sort(mm),
            tnp.array([-1, 0, 1], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        self.assertAllClose(
            cc,
            tnp.array([self.ns * self.nloc // 3] * 3, dtype=tnp.int32),
            rtol=self.prec,
            atol=self.prec,
        )
