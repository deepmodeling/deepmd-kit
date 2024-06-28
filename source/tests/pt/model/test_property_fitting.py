# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import unittest

import numpy as np
import torch
from scipy.stats import (
    special_ortho_group,
)

from deepmd.dpmodel.fitting import PropertyFittingNet as DPProperFittingNet
from deepmd.infer.deep_property import (
    DeepProperty,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.property_model import (
    PropertyModel,
)
from deepmd.pt.model.task.property import (
    PropertyFittingNet,
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


class TestPropertyFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)
        self.rng = np.random.default_rng()
        self.nf, self.nloc, _ = self.nlist.shape
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.scale = self.rng.uniform(0, 1, self.nt).tolist()

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

        for mixed_types, nfp, nap, fit_diag, scale, bias_atom_p in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
            [True, False],
            [None, self.scale],
            [
                np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
                np.array([[11, 12, 13, 4, 15], [16, 17, 18, 9, 20]]),
            ],
        ):
            ft0 = PropertyFittingNet(
                self.nt,
                self.dd0.dim_out,
                task_dim=5,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
                fit_diag=fit_diag,
                scale=scale,
                bias_atom_p=bias_atom_p,
            ).to(env.DEVICE)

            ft1 = DPProperFittingNet.deserialize(ft0.serialize())
            ft2 = PropertyFittingNet.deserialize(ft0.serialize())
            ft3 = DPProperFittingNet.deserialize(ft1.serialize())

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
            ret3 = ft3(
                rd0.detach().cpu().numpy(),
                atype.detach().cpu().numpy(),
                gr.detach().cpu().numpy(),
                fparam=to_numpy_array(ifp),
                aparam=to_numpy_array(iap),
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["property"]),
                ret1["property"],
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["property"]),
                to_numpy_array(ret2["property"]),
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["property"]),
                ret3["property"],
            )

    def test_jit(
        self,
    ):
        for mixed_types, nfp, nap, fit_diag in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
            [True, False],
        ):
            ft0 = PropertyFittingNet(
                self.nt,
                self.dd0.dim_out,
                task_dim=5,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
                fit_diag=fit_diag,
            ).to(env.DEVICE)
            torch.jit.script(ft0)


class TestInvariance(unittest.TestCase):
    def setUp(self) -> None:
        self.natoms = 5
        self.rcut = 4
        self.rcut_smth = 0.5
        self.sel = [46, 92, 4]
        self.nf = 1
        self.nt = 3
        self.rng = np.random.default_rng()
        self.coord = 2 * torch.rand([self.natoms, 3], dtype=dtype, device=env.DEVICE)
        self.shift = torch.tensor([4, 4, 4], dtype=dtype, device=env.DEVICE)
        self.atype = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=env.DEVICE)
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE)
        self.cell = (self.cell + self.cell.T) + 5.0 * torch.eye(3, device=env.DEVICE)
        self.scale = self.rng.uniform(0, 1, self.nt).tolist()

    def test_rot(self):
        atype = self.atype.reshape(1, 5)
        rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype, device=env.DEVICE)
        coord_rot = torch.matmul(self.coord, rmat)

        for mixed_types, nfp, nap, fit_diag, scale in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
            [True, False],
            [None, self.scale],
        ):
            ft0 = PropertyFittingNet(
                self.nt,
                self.dd0.dim_out,  # dim_descrpt
                task_dim=9,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=True,
                fit_diag=fit_diag,
                scale=scale,
            ).to(env.DEVICE)
            if nfp > 0:
                ifp = torch.tensor(
                    self.rng.normal(size=(self.nf, nfp)), dtype=dtype, device=env.DEVICE
                )
            else:
                ifp = None
            if nap > 0:
                iap = torch.tensor(
                    self.rng.normal(size=(self.nf, self.natoms, nap)),
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
                    xyz + self.shift, atype, self.rcut, self.sel, mixed_types
                )

                rd0, gr0, _, _, _ = self.dd0(
                    extended_coord,
                    extended_atype,
                    nlist,
                )

                ret0 = ft0(rd0, extended_atype, gr0, fparam=ifp, aparam=iap)
                res.append(ret0["property"])
            np.testing.assert_allclose(
                to_numpy_array(res[1]),
                to_numpy_array(res[0]),
            )

    def test_permu(self):
        coord = torch.matmul(self.coord, self.cell)
        for fit_diag, scale in itertools.product([True, False], [None, self.scale]):
            ft0 = PropertyFittingNet(
                self.nt,
                self.dd0.dim_out,
                task_dim=8,
                numb_fparam=0,
                numb_aparam=0,
                mixed_types=True,
                fit_diag=fit_diag,
                scale=scale,
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
                    coord[idx_perm], atype, self.rcut, self.sel, False
                )

                rd0, gr0, _, _, _ = self.dd0(
                    extended_coord,
                    extended_atype,
                    nlist,
                )

                ret0 = ft0(rd0, extended_atype, gr0, fparam=None, aparam=None)
                res.append(ret0["property"])

            np.testing.assert_allclose(
                to_numpy_array(res[0][:, idx_perm]),
                to_numpy_array(res[1]),
            )

    def test_trans(self):
        atype = self.atype.reshape(1, 5)
        coord_s = torch.matmul(
            torch.remainder(
                torch.matmul(self.coord + self.shift, torch.linalg.inv(self.cell)), 1.0
            ),
            self.cell,
        )
        for fit_diag, scale in itertools.product([True, False], [None, self.scale]):
            ft0 = PropertyFittingNet(
                self.nt,
                self.dd0.dim_out,
                task_dim=11,
                numb_fparam=0,
                numb_aparam=0,
                mixed_types=True,
                fit_diag=fit_diag,
                scale=scale,
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
                res.append(ret0["property"])

            np.testing.assert_allclose(to_numpy_array(res[0]), to_numpy_array(res[1]))


class TestPropertyModel(unittest.TestCase):
    def setUp(self):
        self.natoms = 5
        self.rcut = 4.0
        self.nt = 3
        self.rcut_smth = 0.5
        self.sel = [46, 92, 4]
        self.nf = 1
        self.coord = 2 * torch.rand([self.natoms, 3], dtype=dtype, device="cpu")
        cell = torch.rand([3, 3], dtype=dtype, device="cpu")
        self.cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        self.atype = torch.IntTensor([0, 0, 0, 1, 1], device="cpu")
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.ft0 = PropertyFittingNet(
            self.nt,
            self.dd0.dim_out,
            task_dim=3,
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=True,
            intensive=True,
        ).to(env.DEVICE)
        self.type_mapping = ["O", "H", "B"]
        self.model = PropertyModel(self.dd0, self.ft0, self.type_mapping)
        self.file_path = "model_output.pth"

    def test_deepproperty_infer(self):
        atype = self.atype.view(self.nf, self.natoms)
        coord = self.coord.reshape(1, 5, 3)
        cell = self.cell.reshape(1, 9)
        jit_md = torch.jit.script(self.model)
        torch.jit.save(jit_md, self.file_path)
        load_md = DeepProperty(self.file_path)
        load_md.eval(coords=coord, atom_types=atype, cells=cell, atomic=True)
        load_md.eval(coords=coord, atom_types=atype, cells=cell, atomic=False)

    def tearDown(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


if __name__ == "__main__":
    unittest.main()
