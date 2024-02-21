# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch
from scipy.stats import (
    special_ortho_group,
)

from deepmd.dpmodel.fitting import DipoleFitting as DPDipoleFitting
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.task.dipole import (
    DipoleFittingNet,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.nlist import (
    extend_input_and_build_neighbor_list,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDipoleFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)
        self.rng = np.random.default_rng()
        self.nf, self.nloc, _ = self.nlist.shape
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)

    def test_consistency(
        self,
    ):
        rd0, gr, _, _, _ = self.dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
        )
        atype = torch.tensor(
            self.atype_ext[:, : self.nloc], dtype=int, device=env.DEVICE
        )

        for mixed_types, nfp, nap in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
        ):
            ft0 = DipoleFittingNet(
                "foo",
                self.nt,
                self.dd0.dim_out,
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
            ).to(env.DEVICE)
            ft1 = DPDipoleFitting.deserialize(ft0.serialize())
            ft2 = DipoleFittingNet.deserialize(ft1.serialize())

            if nfp > 0:
                ifp = torch.tensor(
                    self.rng.normal(size=(self.nf, nfp)), dtype=dtype, device=env.DEVICE
                )
            else:
                ifp = None
            if nap > 0:
                iap = torch.tensor(
                    self.rng.normal(size=(self.nf, self.nloc, nap)),
                    dtype=dtype,
                    device=env.DEVICE,
                )
            else:
                iap = None

            ret0 = ft0(rd0, atype, gr, fparam=ifp, aparam=iap)
            ret1 = ft1(
                rd0.detach().cpu().numpy(),
                atype.detach().cpu().numpy(),
                gr.detach().cpu().numpy(),
                fparam=to_numpy_array(ifp),
                aparam=to_numpy_array(iap),
            )
            ret2 = ft2(rd0, atype, gr, fparam=ifp, aparam=iap)
            np.testing.assert_allclose(
                to_numpy_array(ret0["foo"]),
                ret1["foo"],
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["foo"]),
                to_numpy_array(ret2["foo"]),
            )

    def test_jit(
        self,
    ):
        for mixed_types, nfp, nap in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
        ):
            ft0 = DipoleFittingNet(
                "foo",
                self.nt,
                self.dd0.dim_out,
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
            ).to(env.DEVICE)
            torch.jit.script(ft0)


class TestEquivalence(unittest.TestCase):
    def setUp(self) -> None:
        self.natoms = 5
        self.rcut = 4
        self.rcut_smth = 0.5
        self.sel = [46, 92, 4]
        self.nf = 1
        self.coord = 2 * torch.rand([self.natoms, 3], dtype=dtype, device=env.DEVICE)
        self.shift = torch.tensor([4, 4, 4], dtype=dtype, device=env.DEVICE)
        self.atype = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=env.DEVICE)
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE)
        self.cell = (self.cell + self.cell.T) + 5.0 * torch.eye(3, device=env.DEVICE)

    def test_rot(self):
        atype = self.atype.reshape(1, 5)
        rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype, device=env.DEVICE)
        coord_rot = torch.matmul(self.coord, rmat)
        rng = np.random.default_rng()
        for mixed_types, nfp, nap in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
        ):
            ft0 = DipoleFittingNet(
                "foo",
                3,  # ntype
                self.dd0.dim_out,  # dim_descrpt
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
            ).to(env.DEVICE)
            if nfp > 0:
                ifp = torch.tensor(
                    rng.normal(size=(self.nf, nfp)), dtype=dtype, device=env.DEVICE
                )
            else:
                ifp = None
            if nap > 0:
                iap = torch.tensor(
                    rng.normal(size=(self.nf, self.natoms, nap)),
                    dtype=dtype,
                    device=env.DEVICE,
                )
            else:
                iap = None

            res = []
            for xyz in [self.coord, coord_rot]:
                (
                    extended_coord,
                    extended_atype,
                    _,
                    nlist,
                ) = extend_input_and_build_neighbor_list(
                    xyz + self.shift, atype, self.rcut, self.sel, not mixed_types
                )

                rd0, gr0, _, _, _ = self.dd0(
                    extended_coord,
                    extended_atype,
                    nlist,
                )

                ret0 = ft0(rd0, extended_atype, gr0, fparam=ifp, aparam=iap)
                res.append(ret0["foo"])

            np.testing.assert_allclose(
                to_numpy_array(res[1]), to_numpy_array(torch.matmul(res[0], rmat))
            )

    def test_permu(self):
        coord = torch.matmul(self.coord, self.cell)
        ft0 = DipoleFittingNet(
            "foo",
            3,  # ntype
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=False,
        ).to(env.DEVICE)
        res = []
        for idx_perm in [[0, 1, 2, 3, 4], [1, 0, 4, 3, 2]]:
            atype = self.atype[idx_perm].reshape(1, 5)
            (
                extended_coord,
                extended_atype,
                _,
                nlist,
            ) = extend_input_and_build_neighbor_list(
                coord[idx_perm], atype, self.rcut, self.sel, True
            )

            rd0, gr0, _, _, _ = self.dd0(
                extended_coord,
                extended_atype,
                nlist,
            )

            ret0 = ft0(rd0, extended_atype, gr0, fparam=0, aparam=0)
            res.append(ret0["foo"])

        np.testing.assert_allclose(
            to_numpy_array(res[0][:, idx_perm]), to_numpy_array(res[1])
        )

    def test_trans(self):
        atype = self.atype.reshape(1, 5)
        coord_s = torch.matmul(
            torch.remainder(
                torch.matmul(self.coord + self.shift, torch.linalg.inv(self.cell)), 1.0
            ),
            self.cell,
        )
        ft0 = DipoleFittingNet(
            "foo",
            3,  # ntype
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=True,
        ).to(env.DEVICE)
        res = []
        for xyz in [self.coord, coord_s]:
            (
                extended_coord,
                extended_atype,
                _,
                nlist,
            ) = extend_input_and_build_neighbor_list(
                xyz, atype, self.rcut, self.sel, False
            )

            rd0, gr0, _, _, _ = self.dd0(
                extended_coord,
                extended_atype,
                nlist,
            )

            ret0 = ft0(rd0, extended_atype, gr0, fparam=0, aparam=0)
            res.append(ret0["foo"])

        np.testing.assert_allclose(to_numpy_array(res[0]), to_numpy_array(res[1]))


if __name__ == "__main__":
    unittest.main()
