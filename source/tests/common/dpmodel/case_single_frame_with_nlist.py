# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


class TestCaseSingleFrameWithNlist:
    def setUp(self):
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
        self.atype_ext = np.array([0, 0, 1, 0], dtype=int).reshape([1, self.nall])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.rcut = 0.4
        self.rcut_smth = 2.2
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
        # permute the nlist
        nlist1 = self.nlist[:, self.perm[: self.nloc], :]
        mask = nlist1 == -1
        nlist1 = inv_perm[nlist1]
        nlist1 = np.where(mask, -1, nlist1)
        self.nlist = np.concatenate([self.nlist, nlist1], axis=0)
