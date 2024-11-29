# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import paddle

from deepmd.dpmodel.utils import (
    EnvMat,
)
from deepmd.pd.model.descriptor.env_mat import (
    prod_env_mat,
)
from deepmd.pd.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)

dtype = env.GLOBAL_PD_FLOAT_PRECISION


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
        self.atype_ext = np.array([0, 0, 1, 0], dtype="int64").reshape([1, self.nall])
        self.mapping = np.array([0, 1, 2, 0], dtype="int64").reshape([1, self.nall])
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
            dtype="int64",
        ).reshape([1, self.nloc, sum(self.sel)])
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

        # permute the nlist
        nlist1 = self.nlist[:, self.perm[: self.nloc], :]
        mask = nlist1 == -1
        nlist1 = inv_perm[nlist1]
        nlist1 = np.where(mask, -1, nlist1)
        self.nlist = np.concatenate([self.nlist, nlist1], axis=0)
        self.atol = 1e-12


class TestCaseSingleFrameWithNlistWithVirtual:
    def setUp(self):
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
        self.atype_ext = np.array([0, -1, 0, 1, 0], dtype="int64").reshape(
            [1, self.nall]
        )
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
            dtype="int64",
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
    def setUp(self):
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
        self.atype = np.array([0, 0, 1], dtype="int64").reshape([1, self.nloc])
        self.cell = 2.0 * np.eye(3).reshape([1, 9])
        # sel = [5, 2]
        self.sel = [16, 8]
        self.sel_mix = [24]
        self.natoms = [3, 3, 2, 1]
        self.rcut = 2.2
        self.rcut_smth = 0.4
        self.atol = 1e-12


# to be merged with the tf test case
class TestEnvMat(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)
        em0 = EnvMat(self.rcut, self.rcut_smth)
        mm0, diff0, ww0 = em0.call(
            self.coord_ext, self.atype_ext, self.nlist, davg, dstd
        )
        mm1, diff1, ww1 = prod_env_mat(
            paddle.to_tensor(self.coord_ext, dtype=dtype).to(device=env.DEVICE),
            paddle.to_tensor(self.nlist, dtype="int64").to(device=env.DEVICE),
            paddle.to_tensor(self.atype_ext[:, :nloc], dtype="int64").to(
                device=env.DEVICE
            ),
            paddle.to_tensor(davg).to(device=env.DEVICE),
            paddle.to_tensor(dstd).to(device=env.DEVICE),
            self.rcut,
            self.rcut_smth,
        )
        np.testing.assert_allclose(mm0, mm1.detach().cpu().numpy())
        np.testing.assert_allclose(diff0, diff1.detach().cpu().numpy())
        np.testing.assert_allclose(ww0, ww1.detach().cpu().numpy())
        np.testing.assert_allclose(mm0[0][self.perm[: self.nloc]], mm0[1])
