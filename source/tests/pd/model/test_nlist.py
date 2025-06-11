# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import paddle

from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.nlist import (
    build_directional_neighbor_list,
    build_multiple_neighbor_list,
    build_neighbor_list,
    extend_coord_with_ghosts,
    get_multiple_nlist_key,
)
from deepmd.pd.utils.region import (
    inter2phys,
)

dtype = paddle.float64


class TestNeighList(unittest.TestCase):
    def setUp(self):
        self.nf = 3
        self.nloc = 3
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = paddle.to_tensor(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype, place=env.DEVICE
        )
        self.icoord = paddle.to_tensor(
            [[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.1]], dtype=dtype, place=env.DEVICE
        )
        self.atype = paddle.to_tensor([-1, 0, 1], dtype=paddle.int64).to(
            device=env.DEVICE
        )
        [self.cell, self.icoord, self.atype] = [
            ii.unsqueeze(0) for ii in [self.cell, self.icoord, self.atype]
        ]
        self.coord = inter2phys(self.icoord, self.cell).reshape([-1, self.nloc * 3])
        self.cell = self.cell.reshape([-1, 9])
        [self.cell, self.coord, self.atype] = [
            paddle.tile(ii, [self.nf, 1]) for ii in [self.cell, self.coord, self.atype]
        ]
        self.rcut = 1.01
        self.prec = 1e-10
        self.nsel = [10, 10]
        # genrated by preprocess.build_neighbor_list
        # ref_nlist, _, _ = legacy_build_neighbor_list(
        #   2, ecoord[0], eatype[0],
        #   self.rcut,
        #   paddle.to_tensor([10,20], dtype=paddle.int64),
        #   mapping[0], type_split=True, )
        self.ref_nlist = paddle.to_tensor(
            [
                [-1] * sum(self.nsel),
                [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1],
            ],
            place=env.DEVICE,
        )

    def test_build_notype(self):
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        # test normal sel
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel),
            distinguish_types=False,
        )
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        np.testing.assert_allclose(
            paddle.sort(nlist_loc, axis=-1).numpy(),
            paddle.sort(self.ref_nlist, axis=-1).numpy(),
        )
        # test a very large sel
        nlist = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            self.rcut,
            sum(self.nsel) + 300,  # +300, real nnei==224
            distinguish_types=False,
        )
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        np.testing.assert_allclose(
            paddle.sort(nlist_loc, descending=True, axis=-1)[
                :, : sum(self.nsel)
            ].numpy(),
            paddle.sort(self.ref_nlist, descending=True, axis=-1).numpy(),
        )

    def test_build_type(self):
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
        np.testing.assert_allclose(nlist[0].numpy(), nlist[1].numpy())
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        for ii in range(2):
            np.testing.assert_allclose(
                paddle.sort(
                    paddle.split(nlist_loc, (self.nsel), axis=-1)[ii], axis=-1
                ).numpy(),
                paddle.sort(
                    paddle.split(self.ref_nlist, (self.nsel), axis=-1)[ii], axis=-1
                ).numpy(),
            )

    def test_build_multiple_nlist(self):
        rcuts = [1.01, 2.01]
        nsels = [20, 80]
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, max(rcuts)
        )
        nlist1 = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            rcuts[1],
            nsels[1] - 1,
            distinguish_types=False,
        )
        pad = -1 * paddle.ones([self.nf, self.nloc, 1], dtype=nlist1.dtype).to(
            device=nlist1.place
        )
        nlist2 = paddle.concat([nlist1, pad], axis=-1)
        nlist0 = build_neighbor_list(
            ecoord,
            eatype,
            self.nloc,
            rcuts[0],
            nsels[0],
            distinguish_types=False,
        )
        nlists = build_multiple_neighbor_list(ecoord, nlist1, rcuts, nsels)
        for dd in range(2):
            self.assertEqual(
                nlists[get_multiple_nlist_key(rcuts[dd], nsels[dd])].shape[-1],
                nsels[dd],
            )
        np.testing.assert_allclose(
            nlists[get_multiple_nlist_key(rcuts[0], nsels[0])].numpy(),
            nlist0.numpy(),
        )
        np.testing.assert_allclose(
            nlists[get_multiple_nlist_key(rcuts[1], nsels[1])].numpy(),
            nlist2.numpy(),
        )

    def test_extend_coord(self):
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        # expected ncopy x nloc
        self.assertEqual(list(ecoord.shape), [self.nf, self.nall * 3])
        self.assertEqual(list(eatype.shape), [self.nf, self.nall])
        self.assertEqual(list(mapping.shape), [self.nf, self.nall])
        # check the nloc part is identical with original coord
        np.testing.assert_allclose(
            ecoord[:, : self.nloc * 3].numpy(),
            self.coord.numpy(),
            rtol=self.prec,
            atol=self.prec,
        )
        # check the shift vectors are aligned with grid
        shift_vec = (
            ecoord.reshape([-1, self.ns, self.nloc, 3])
            - self.coord.reshape([-1, self.nloc, 3])[:, None, :, :]
        )
        shift_vec = shift_vec.reshape([-1, self.nall, 3])
        # hack!!! assumes identical cell across frames
        shift_vec = paddle.matmul(
            shift_vec, paddle.linalg.inv(self.cell.reshape([self.nf, 3, 3])[0])
        )
        # nf x nall x 3
        shift_vec = paddle.round(shift_vec)
        # check: identical shift vecs
        np.testing.assert_allclose(
            shift_vec[0].numpy(), shift_vec[1].numpy(), rtol=self.prec, atol=self.prec
        )
        # check: shift idx aligned with grid
        mm, cc = paddle.unique(shift_vec[0][:, 0], axis=-1, return_counts=True)
        np.testing.assert_allclose(
            mm.numpy(),
            paddle.to_tensor([-2, -1, 0, 1, 2], dtype=dtype)
            .to(device=env.DEVICE)
            .numpy(),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc.numpy(),
            paddle.to_tensor(
                [self.ns * self.nloc // 5] * 5, dtype=paddle.int64, place=env.DEVICE
            ).numpy(),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = paddle.unique(shift_vec[1][:, 1], axis=-1, return_counts=True)
        np.testing.assert_allclose(
            mm.numpy(),
            paddle.to_tensor([-2, -1, 0, 1, 2], dtype=dtype).to(device=env.DEVICE),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc.numpy(),
            paddle.to_tensor(
                [self.ns * self.nloc // 5] * 5, dtype=paddle.int64, place=env.DEVICE
            ),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = paddle.unique(shift_vec[1][:, 2], axis=-1, return_counts=True)
        np.testing.assert_allclose(
            mm.numpy(),
            paddle.to_tensor([-1, 0, 1], dtype=dtype).to(device=env.DEVICE).numpy(),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc.numpy(),
            paddle.to_tensor(
                [self.ns * self.nloc // 3] * 3, dtype=paddle.int64, place=env.DEVICE
            ).numpy(),
            rtol=self.prec,
            atol=self.prec,
        )

    def test_build_directional_nlist(self):
        """Directional nlist is tested against the standard nlist implementation."""
        ecoord, eatype, mapping = extend_coord_with_ghosts(
            self.coord, self.atype, self.cell, self.rcut
        )
        for distinguish_types, mysel in zip([True, False], [sum(self.nsel), 300]):
            # full neighbor list
            nlist_full = build_neighbor_list(
                ecoord,
                eatype,
                self.nloc,
                self.rcut,
                sum(self.nsel),
                distinguish_types=distinguish_types,
            )
            # central as part of the system
            nlist = build_directional_neighbor_list(
                ecoord[:, 3:6],
                eatype[:, 1:2],
                paddle.concat(
                    [
                        ecoord[:, 0:3],
                        paddle.zeros(
                            [self.nf, 3],
                            dtype=dtype,
                        ).to(device=env.DEVICE),  # placeholder
                        ecoord[:, 6:],
                    ],
                    axis=1,
                ),
                paddle.concat(
                    [
                        eatype[:, 0:1],
                        -1
                        * paddle.ones(
                            [self.nf, 1],
                            dtype="int64",
                        ).to(device=env.DEVICE),  # placeholder
                        eatype[:, 2:],
                    ],
                    axis=1,
                ),
                self.rcut,
                mysel,
                distinguish_types=distinguish_types,
            )
            np.testing.assert_allclose(nlist[0].numpy(), nlist[1].numpy())
            np.testing.assert_allclose(nlist[0].numpy(), nlist[2].numpy())
            np.testing.assert_allclose(
                paddle.sort(nlist[0], descending=True, axis=-1)[
                    :, : sum(self.nsel)
                ].numpy(),
                paddle.sort(nlist_full[0][1:2], descending=True, axis=-1).numpy(),
            )
