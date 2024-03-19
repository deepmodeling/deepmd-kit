# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import torch

from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    build_multiple_neighbor_list,
    build_neighbor_list,
    extend_coord_with_ghosts,
    get_multiple_nlist_key,
)
from deepmd.pt.utils.region import (
    inter2phys,
)

dtype = torch.float64


class TestNeighList(unittest.TestCase):
    def setUp(self):
        self.nf = 3
        self.nloc = 3
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = torch.tensor(
            [[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype, device=env.DEVICE
        )
        self.icoord = torch.tensor(
            [[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.1]], dtype=dtype, device=env.DEVICE
        )
        self.atype = torch.tensor([-1, 0, 1], dtype=torch.int, device=env.DEVICE)
        [self.cell, self.icoord, self.atype] = [
            ii.unsqueeze(0) for ii in [self.cell, self.icoord, self.atype]
        ]
        self.coord = inter2phys(self.icoord, self.cell).view([-1, self.nloc * 3])
        self.cell = self.cell.view([-1, 9])
        [self.cell, self.coord, self.atype] = [
            torch.tile(ii, [self.nf, 1]) for ii in [self.cell, self.coord, self.atype]
        ]
        self.rcut = 1.01
        self.prec = 1e-10
        self.nsel = [10, 10]
        # genrated by preprocess.build_neighbor_list
        # ref_nlist, _, _ = legacy_build_neighbor_list(
        #   2, ecoord[0], eatype[0],
        #   self.rcut,
        #   torch.tensor([10,20], dtype=torch.long),
        #   mapping[0], type_split=True, )
        self.ref_nlist = torch.tensor(
            [
                [-1] * sum(self.nsel),
                [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 2, 2, 2, 2, -1, -1, -1, -1, -1, -1],
                [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 2, 2, 2, 2, 2, 2, -1, -1, -1, -1],
            ],
            device=env.DEVICE,
        )

    def test_build_notype(self):
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
        torch.testing.assert_close(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        torch.testing.assert_close(
            torch.sort(nlist_loc, dim=-1)[0],
            torch.sort(self.ref_nlist, dim=-1)[0],
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
        torch.testing.assert_close(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        for ii in range(2):
            torch.testing.assert_close(
                torch.sort(torch.split(nlist_loc, self.nsel, dim=-1)[ii], dim=-1)[0],
                torch.sort(torch.split(self.ref_nlist, self.nsel, dim=-1)[ii], dim=-1)[
                    0
                ],
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
        pad = -1 * torch.ones(
            [self.nf, self.nloc, 1], dtype=nlist1.dtype, device=nlist1.device
        )
        nlist2 = torch.cat([nlist1, pad], dim=-1)
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
        torch.testing.assert_close(
            nlists[get_multiple_nlist_key(rcuts[0], nsels[0])],
            nlist0,
        )
        torch.testing.assert_close(
            nlists[get_multiple_nlist_key(rcuts[1], nsels[1])],
            nlist2,
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
        torch.testing.assert_close(
            ecoord[:, : self.nloc * 3], self.coord, rtol=self.prec, atol=self.prec
        )
        # check the shift vectors are aligned with grid
        shift_vec = (
            ecoord.view([-1, self.ns, self.nloc, 3])
            - self.coord.view([-1, self.nloc, 3])[:, None, :, :]
        )
        shift_vec = shift_vec.view([-1, self.nall, 3])
        # hack!!! assumes identical cell across frames
        shift_vec = torch.matmul(
            shift_vec, torch.linalg.inv(self.cell.view([self.nf, 3, 3])[0])
        )
        # nf x nall x 3
        shift_vec = torch.round(shift_vec)
        # check: identical shift vecs
        torch.testing.assert_close(
            shift_vec[0], shift_vec[1], rtol=self.prec, atol=self.prec
        )
        # check: shift idx aligned with grid
        mm, cc = torch.unique(shift_vec[0][:, 0], dim=-1, return_counts=True)
        torch.testing.assert_close(
            mm,
            torch.tensor([-2, -1, 0, 1, 2], dtype=dtype, device=env.DEVICE),
            rtol=self.prec,
            atol=self.prec,
        )
        torch.testing.assert_close(
            cc,
            torch.tensor(
                [self.ns * self.nloc // 5] * 5, dtype=torch.long, device=env.DEVICE
            ),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = torch.unique(shift_vec[1][:, 1], dim=-1, return_counts=True)
        torch.testing.assert_close(
            mm,
            torch.tensor([-2, -1, 0, 1, 2], dtype=dtype, device=env.DEVICE),
            rtol=self.prec,
            atol=self.prec,
        )
        torch.testing.assert_close(
            cc,
            torch.tensor(
                [self.ns * self.nloc // 5] * 5, dtype=torch.long, device=env.DEVICE
            ),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = torch.unique(shift_vec[1][:, 2], dim=-1, return_counts=True)
        torch.testing.assert_close(
            mm,
            torch.tensor([-1, 0, 1], dtype=dtype, device=env.DEVICE),
            rtol=self.prec,
            atol=self.prec,
        )
        torch.testing.assert_close(
            cc,
            torch.tensor(
                [self.ns * self.nloc // 3] * 3, dtype=torch.long, device=env.DEVICE
            ),
            rtol=self.prec,
            atol=self.prec,
        )
