# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)
from deepmd.dpmodel.utils import (
    build_multiple_neighbor_list,
    build_neighbor_list,
    extend_coord_with_ghosts,
    get_multiple_nlist_key,
    inter2phys,
)


class TestDPModelFormatNlist(unittest.TestCase):
    def setUp(self) -> None:
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 5
        self.nf, self.nt = 1, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
                [2.3, 0, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall * 3])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.expected_nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.atype_ext = np.array([0, 0, 1, 0, 1], dtype=int).reshape([1, self.nall])
        self.rcut_smth = 0.4
        self.rcut = 2.1

        nf, nloc, nnei = self.expected_nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
        )
        type_map = ["foo", "bar"]
        self.md = EnergyModel(ds, ft, type_map=type_map)

    def test_nlist_eq(self) -> None:
        # n_nnei == nnei
        nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)

    def test_nlist_st(self) -> None:
        # n_nnei < nnei
        nlist = np.array(
            [
                [1, 3, -1, 2],
                [0, -1, -1, 2],
                [0, 1, -1, -1],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)

    def test_nlist_lt(self) -> None:
        # n_nnei > nnei
        nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1, -1, 4],
                [0, -1, 4, -1, -1, 2, -1, 3, -1],
                [0, 1, -1, -1, -1, 4, -1, -1, 3],
            ],
            dtype=np.int64,
        ).reshape([1, self.nloc, -1])
        nlist1 = self.md.format_nlist(
            self.coord_ext,
            self.atype_ext,
            nlist,
        )
        np.testing.assert_allclose(self.expected_nlist, nlist1)


dtype = np.float64


class TestNeighList(unittest.TestCase):
    def setUp(self) -> None:
        self.nf = 3
        self.nloc = 3
        self.ns = 5 * 5 * 3
        self.nall = self.ns * self.nloc
        self.cell = np.array([[1, 0, 0], [0.4, 0.8, 0], [0.1, 0.3, 2.1]], dtype=dtype)
        self.icoord = np.array([[0, 0, 0], [0, 0, 0], [0.5, 0.5, 0.1]], dtype=dtype)
        self.atype = np.array([-1, 0, 1], dtype=np.int32)
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
        np.testing.assert_allclose(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        np.testing.assert_allclose(
            np.sort(nlist_loc, axis=-1),
            np.sort(self.ref_nlist, axis=-1),
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
        np.testing.assert_allclose(nlist[0], nlist[1])
        nlist_mask = nlist[0] == -1
        nlist_loc = mapping[0][nlist[0]]
        nlist_loc[nlist_mask] = -1
        for ii in range(2):
            np.testing.assert_allclose(
                np.sort(np.split(nlist_loc, self.nsel, axis=-1)[ii], axis=-1),
                np.sort(np.split(self.ref_nlist, self.nsel, axis=-1)[ii], axis=-1),
            )

    def test_build_multiple_nlist(self) -> None:
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
        pad = -1 * np.ones([self.nf, self.nloc, 1], dtype=nlist1.dtype)
        nlist2 = np.concatenate([nlist1, pad], axis=-1)
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
            nlists[get_multiple_nlist_key(rcuts[0], nsels[0])],
            nlist0,
        )
        np.testing.assert_allclose(
            nlists[get_multiple_nlist_key(rcuts[1], nsels[1])],
            nlist2,
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
        mm, cc = np.unique(shift_vec[0][:, 0], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-2, -1, 0, 1, 2], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([self.ns * self.nloc // 5] * 5, dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = np.unique(shift_vec[1][:, 1], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-2, -1, 0, 1, 2], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([self.ns * self.nloc // 5] * 5, dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )
        mm, cc = np.unique(shift_vec[1][:, 2], return_counts=True)
        np.testing.assert_allclose(
            mm,
            np.array([-1, 0, 1], dtype=dtype),
            rtol=self.prec,
            atol=self.prec,
        )
        np.testing.assert_allclose(
            cc,
            np.array([self.ns * self.nloc // 3] * 3, dtype=np.int32),
            rtol=self.prec,
            atol=self.prec,
        )
