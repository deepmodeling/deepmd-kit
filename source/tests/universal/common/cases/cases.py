# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np


# originally copied from source/tests/pt/model/test_env_mat.py
class TestCaseSingleFrameWithNlist:
    def setUp(self) -> None:
        # nloc == 3, nall == 4
        self.nloc = 3
        self.nall = 4
        self.nf, self.nt = 2, 2
        self.dim_descrpt = 100
        self.dim_embed = 20
        rng = np.random.default_rng()
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
        self.mapping = np.array([0, 1, 2, 0], dtype=int).reshape([1, self.nall])
        # sel = [5, 2]
        self.sel = [5, 2]
        self.sel_mix = [7]
        self.natoms = [3, 3, 2, 1]
        self.nlist = np.array(
            [
                [1, 3, -1, -1, -1, 2, -1],
                [0, -1, -1, -1, -1, 2, -1],
                [0, 1, -1, -1, -1, -1, -1],
            ],
            dtype=int,
        ).reshape([1, self.nloc, sum(self.sel)])
        self.mock_descriptor = rng.normal(size=(1, self.nloc, self.dim_descrpt))
        self.mock_gr = rng.normal(size=(1, self.nloc, self.dim_embed, 3))
        self.mock_energy_bias = rng.normal(size=(self.nt, 1))
        self.rcut = 2.2
        self.rcut_smth = 0.4
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
        self.mock_descriptor = np.concatenate(
            [self.mock_descriptor, self.mock_descriptor[:, self.perm[: self.nloc], :]],
            axis=0,
        )
        self.mock_gr = np.concatenate(
            [self.mock_gr, self.mock_gr[:, self.perm[: self.nloc], :, :]], axis=0
        )

        # permute the nlist
        nlist1 = self.nlist[:, self.perm[: self.nloc], :]
        mask = nlist1 == -1
        nlist1 = inv_perm[nlist1]
        nlist1 = np.where(mask, -1, nlist1)
        self.nlist = np.concatenate([self.nlist, nlist1], axis=0)
        self.atol = 1e-12


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
        self.sel_mix = [7]
        self.natoms = [3, 3, 2, 1]
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
        self.sel_mix = [24]
        self.natoms = [3, 3, 2, 1]
        self.rcut = 2.2
        self.rcut_smth = 0.4
        self.atol = 1e-12
