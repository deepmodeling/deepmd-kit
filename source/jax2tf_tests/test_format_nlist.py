# SPDX-License-Identifier: LGPL-3.0-or-later
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

from deepmd.jax.jax2tf.format_nlist import (
    format_nlist,
)
from deepmd.jax.jax2tf.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)

GLOBAL_SEED = 20241110


class TestFormatNlist(tf.test.TestCase):
    def setUp(self) -> None:
        self.nf = 3
        self.nloc = 3
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = tnp.array(
            [[[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]]], dtype=tnp.float64
        )
        self.icoord = tnp.array(
            [[[0.035, 0.062, 0.064], [0.085, 0.058, 0.021], [0.537, 0.553, 0.124]]],
            dtype=tnp.float64,
        )
        self.atype = tnp.array([[1, 0, 1]], dtype=tnp.int32)
        self.nsel = [10, 10]
        self.rcut = 1.01

        self.ecoord, self.eatype, mapping = extend_coord_with_ghosts(
            self.icoord, self.atype, self.cell, self.rcut
        )
        self.nlist = build_neighbor_list(
            self.ecoord,
            self.eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel),
            distinguish_types=False,
        )

    def test_format_nlist_equal(self) -> None:
        nlist = format_nlist(self.ecoord, self.nlist, sum(self.nsel), self.rcut)
        self.assertAllEqual(nlist, self.nlist)

    def test_format_nlist_less(self) -> None:
        nlist = build_neighbor_list(
            self.ecoord,
            self.eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel) - 5,
            distinguish_types=False,
        )
        nlist = format_nlist(self.ecoord, nlist, sum(self.nsel), self.rcut)
        self.assertAllEqual(nlist, self.nlist)

    def test_format_nlist_large(self) -> None:
        nlist = build_neighbor_list(
            self.ecoord,
            self.eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel) + 5,
            distinguish_types=False,
        )
        # random shuffle
        shuffle_idx = tf.random.shuffle(tf.range(nlist.shape[2]))
        nlist = tnp.take(nlist, shuffle_idx, axis=2)
        nlist = format_nlist(self.ecoord, nlist, sum(self.nsel), self.rcut)
        # we only need to ensure the result is correct, no need to check the order
        self.assertAllEqual(tnp.sort(nlist, axis=-1), tnp.sort(self.nlist, axis=-1))

    def test_format_nlist_larger_rcut(self) -> None:
        nlist = build_neighbor_list(
            self.ecoord,
            self.eatype,
            self.nloc,
            self.rcut * 2,
            40,
            distinguish_types=False,
        )
        # random shuffle
        shuffle_idx = tf.random.shuffle(tf.range(nlist.shape[2]))
        nlist = tnp.take(nlist, shuffle_idx, axis=2)
        nlist = format_nlist(self.ecoord, nlist, sum(self.nsel), self.rcut)
        # we only need to ensure the result is correct, no need to check the order
        self.assertAllEqual(tnp.sort(nlist, axis=-1), tnp.sort(self.nlist, axis=-1))
