# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.utils import (
    inter2phys,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
)
from deepmd.tf.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    default_tf_session_config,
    tf,
)
from deepmd.tf.utils.nlist import (
    extend_coord_with_ghosts,
)


class TestNeighList(unittest.TestCase):
    def setUp(self) -> None:
        self.nf = 3
        self.nloc = 2
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = np.array(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=GLOBAL_NP_FLOAT_PRECISION
        )
        self.icoord = np.array(
            [[0, 0, 0], [0.5, 0.5, 0.1]], dtype=GLOBAL_NP_FLOAT_PRECISION
        )
        self.atype = np.array([0, 1], dtype=int)
        [self.cell, self.icoord, self.atype] = [
            np.expand_dims(ii, 0) for ii in [self.cell, self.icoord, self.atype]
        ]
        self.coord = inter2phys(self.icoord, self.cell).reshape([-1, self.nloc * 3])
        self.cell = self.cell.reshape([-1, 9])
        [self.cell, self.coord, self.atype] = [
            np.tile(ii, [self.nf, 1]) for ii in [self.cell, self.coord, self.atype]
        ]
        self.rcut = 1.01
        self.prec = 1e-10
        self.nsel = [10, 10]
        self.ref_nlist = np.array(
            [
                [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
                [0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1],
            ]
        )

    def test_extend_coord(self) -> None:
        t_coord = tf.placeholder(
            GLOBAL_TF_FLOAT_PRECISION, [None, None], name="i_coord"
        )
        t_atype = tf.placeholder(tf.int32, [None, None], name="i_atype")
        t_cell = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None, None], name="i_cell")
        t_pbc = tf.placeholder(tf.bool, [], name="i_pbc")
        t_ecoord, t_eatype, t_mapping = extend_coord_with_ghosts(
            t_coord, t_atype, t_cell, self.rcut, t_pbc
        )
        with tf.Session(config=default_tf_session_config) as sess:
            ecoord, eatype, mapping = sess.run(
                [t_ecoord, t_eatype, t_mapping],
                feed_dict={
                    t_coord: self.coord,
                    t_atype: self.atype,
                    t_cell: self.cell,
                    t_pbc: self.cell is not None,
                },
            )
        # expected ncopy x nloc
        self.assertEqual(list(ecoord.shape), [self.nf, self.nall * 3])
        self.assertEqual(list(eatype.shape), [self.nf, self.nall])
        self.assertEqual(list(mapping.shape), [self.nf, self.nall])
        # check the nloc part is identical with original coord
        np.testing.assert_allclose(
            ecoord[:, : self.nloc * 3], self.coord, rtol=self.prec, atol=self.prec
        )
        # check the shift vectors are aligned with grid
        shift_vec = (
            ecoord.reshape([-1, self.ns, self.nloc, 3])
            - self.coord.reshape([-1, self.nloc, 3])[:, None, :, :]
        )
        shift_vec = shift_vec.reshape([-1, self.nall, 3])
        # hack!!! assumes identical cell across frames
        shift_vec = np.matmul(
            shift_vec, np.linalg.inv(self.cell.reshape([self.nf, 3, 3])[0])
        )
        # nf x nall x 3
        shift_vec = np.round(shift_vec)
        # check: identical shift vecs
        np.testing.assert_allclose(
            shift_vec[0], shift_vec[1], rtol=self.prec, atol=self.prec
        )
        # check: shift idx aligned with grid
        mm, cc = np.unique(shift_vec[0][:, 0], axis=-1, return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-2, -1, 0, 1, 2], dtype=GLOBAL_NP_FLOAT_PRECISION),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([30, 30, 30, 30, 30], dtype=np.int64),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = np.unique(shift_vec[1][:, 1], axis=-1, return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-2, -1, 0, 1, 2], dtype=GLOBAL_NP_FLOAT_PRECISION),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([30, 30, 30, 30, 30], dtype=GLOBAL_NP_FLOAT_PRECISION),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = np.unique(shift_vec[1][:, 2], axis=-1, return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-1, 0, 1], dtype=GLOBAL_NP_FLOAT_PRECISION),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([50, 50, 50], dtype=np.int64),
            rtol=self.prec,
            atol=self.prec,
        )
