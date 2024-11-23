# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


class TestCaseSingleFrameWithoutNlist:
    def setUp(self) -> None:
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nf, self.nt = 1, 2
        self.coord = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        ).reshape([1, self.nloc * 3])
        self.atype = np.array([0, 0, 1], dtype=int).reshape([1, self.nloc])
        self.cell = 2.0 * np.eye(3).reshape([1, 9])
        # sel = [5, 2]
        self.sel = [16, 8]
        self.rcut = 2.2
        self.rcut_smth = 0.4
        self.atol = 1e-12


class TestCaseSingleFrameWithNlist:
    def setUp(self) -> None:
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 2, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall, 3])
        self.coord = self.coord_ext[:, : self.nloc, :]
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        self.mapping = np.array([0, 1, 2, 0], dtype=int).reshape([1, self.nall])
        self.atype = self.atype_ext[:, : self.nloc]
        # sel = [5, 2]
        self.sel = [5, 2]
        self.sel_mix = [7]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut = 2.2
        self.rcut_smth = 0.4
        self.atol = 1e-12

        # permutations
        self.perm = np.array([2, 0, 1, 3], dtype=np.int32)
        inv_perm = np.array([1, 2, 0, 3], dtype=np.int32)
        # permute the coord and atype
        self.coord_ext = np.concatenate(
            [self.coord_ext, self.coord_ext[:, self.perm, :]], axis=0
        ).reshape(self.nf, self.nall * 3)
        self.atype_ext = np.concatenate(
            [self.atype_ext, self.atype_ext[:, self.perm]], axis=0
        )
        self.mapping = np.concatenate(
            [self.mapping, self.mapping[:, self.perm]], axis=0
        )
        # permute the nlist
        nlist1 = self.nlist[:, self.perm[: self.nloc], :]
        mask = nlist1 == -1
        nlist1 = inv_perm[nlist1]
        nlist1 = np.where(mask, -1, nlist1)
        self.nlist = np.concatenate([self.nlist, nlist1], axis=0)


class TestCaseSingleFrameWithNlistWithVirtual:
    def setUp(self) -> None:
        # nloc == 3, nall == 4
        self.nloc = 4
        self.nall = 5
        self.nf, self.nt = 2, 2
        self.coord_ext = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -2, 0],
            ],
            dtype=np.float64,
        ).reshape([1, self.nall, 3])
        self.atype_ext = np.array([0, -1, 0, 1, 0], dtype=int).reshape([1, self.nall])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.nlist = np.array(
            [
                [2, 4, -1, -1, -1, 3, -1],
                [-1, -1, -1, -1, -1, -1, -1],
                [0, -1, -1, -1, -1, 3, -1],
                [0, 2, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut = 2.2
        self.rcut_smth = 0.4
        # permutations
        self.perm = np.array([3, 0, 1, 2, 4], dtype=np.int32)
        inv_perm = np.argsort(self.perm)
        # permute the coord and atype
        self.coord_ext = np.concatenate(
            [self.coord_ext, self.coord_ext[:, self.perm, :]], axis=0
        ).reshape(self.nf, self.nall * 3)
        self.atype_ext = np.concatenate(
            [self.atype_ext, self.atype_ext[:, self.perm]], axis=0
        )
        # permute the nlist
        nlist1 = self.nlist[:, self.perm[: self.nloc], :]
        mask = nlist1 == -1
        nlist1 = inv_perm[nlist1]
        nlist1 = np.where(mask, -1, nlist1)
        self.nlist = np.concatenate([self.nlist, nlist1], axis=0)
        self.get_real_mapping = np.array([[0, 2, 3], [0, 1, 3]], dtype=np.int32)
        self.atol = 1e-12
