# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import unittest

import numpy as np
import torch
from scipy.stats import (
    special_ortho_group,
)

from deepmd.dpmodel.fitting import DipoleFitting as DPDipoleFitting
from deepmd.infer.deep_dipole import (
    DeepDipole,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.dipole_model import (
    DipoleModel,
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
    to_torch_tensor,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


def finite_difference(f, x, a, delta=1e-6):
    in_shape = x.shape
    y0 = f(x, a)
    out_shape = y0.shape
    res = np.empty(out_shape + in_shape)
    for idx in np.ndindex(*in_shape):
        diff = np.zeros(in_shape)
        diff[idx] += delta
        y1p = f(x + diff, a)
        y1n = f(x - diff, a)
        res[(Ellipsis, *idx)] = (y1p - y1n) / (2 * delta)
    return res


class TestDipoleFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.rng = np.random.default_rng(GLOBAL_SEED)
        self.nf, self.nloc, _ = self.nlist.shape
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)

    def test_consistency(
        self,
    ) -> None:
        rd0, gr, _, _, _ = self.dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
            torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
            torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
        )
        atype = torch.tensor(
            self.atype_ext[:, : self.nloc], dtype=int, device=env.DEVICE
        )

        for nfp, nap in itertools.product(
            [0, 3],
            [0, 4],
        ):
            ft0 = DipoleFittingNet(
                self.nt,
                self.dd0.dim_out,
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=self.dd0.mixed_types(),
                seed=GLOBAL_SEED,
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
                to_numpy_array(ret0["dipole"]),
                ret1["dipole"],
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["dipole"]),
                to_numpy_array(ret2["dipole"]),
            )

    def test_jit(
        self,
    ) -> None:
        for mixed_types, nfp, nap in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
        ):
            ft0 = DipoleFittingNet(
                self.nt,
                self.dd0.dim_out,
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)
            torch.jit.script(ft0)


class TestEquivalence(unittest.TestCase):
    def setUp(self) -> None:
        self.natoms = 5
        self.rcut = 4
        self.rcut_smth = 0.5
        self.sel = [46, 92, 4]
        self.nf = 1
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        self.coord = 2 * torch.rand(
            [self.natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        self.shift = torch.tensor([4, 4, 4], dtype=dtype, device=env.DEVICE)
        self.atype = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=env.DEVICE)
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.cell = torch.rand(
            [3, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        self.cell = (self.cell + self.cell.T) + 5.0 * torch.eye(3, device=env.DEVICE)

    def test_rot(self) -> None:
        atype = self.atype.reshape(1, 5)
        rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype, device=env.DEVICE)
        coord_rot = torch.matmul(self.coord, rmat)
        # use larger cell to rotate only coord and shift to the center of cell
        cell_rot = 10.0 * torch.eye(3, dtype=dtype, device=env.DEVICE)
        rng = np.random.default_rng(GLOBAL_SEED)
        for nfp, nap in itertools.product(
            [0, 3],
            [0, 4],
        ):
            ft0 = DipoleFittingNet(
                3,  # ntype
                self.dd0.dim_out,  # dim_descrpt
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=self.dd0.mixed_types(),
                seed=GLOBAL_SEED,
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
                    xyz + self.shift,
                    atype,
                    self.rcut,
                    self.sel,
                    self.dd0.mixed_types(),
                    box=cell_rot,
                )

                rd0, gr0, _, _, _ = self.dd0(
                    extended_coord,
                    extended_atype,
                    nlist,
                )

                ret0 = ft0(rd0, atype, gr0, fparam=ifp, aparam=iap)
                res.append(ret0["dipole"])

            np.testing.assert_allclose(
                to_numpy_array(res[1]), to_numpy_array(torch.matmul(res[0], rmat))
            )

    def test_permu(self) -> None:
        coord = torch.matmul(self.coord, self.cell)
        ft0 = DipoleFittingNet(
            3,  # ntype
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=self.dd0.mixed_types(),
            seed=GLOBAL_SEED,
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
                coord[idx_perm],
                atype,
                self.rcut,
                self.sel,
                self.dd0.mixed_types(),
                box=self.cell,
            )

            rd0, gr0, _, _, _ = self.dd0(
                extended_coord,
                extended_atype,
                nlist,
            )

            ret0 = ft0(rd0, atype, gr0, fparam=None, aparam=None)
            res.append(ret0["dipole"])

        np.testing.assert_allclose(
            to_numpy_array(res[0][:, idx_perm]), to_numpy_array(res[1])
        )

    def test_trans(self) -> None:
        atype = self.atype.reshape(1, 5)
        coord_s = torch.matmul(
            torch.remainder(
                torch.matmul(self.coord + self.shift, torch.linalg.inv(self.cell)), 1.0
            ),
            self.cell,
        )
        ft0 = DipoleFittingNet(
            3,  # ntype
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=self.dd0.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        res = []
        for xyz in [self.coord, coord_s]:
            (
                extended_coord,
                extended_atype,
                _,
                nlist,
            ) = extend_input_and_build_neighbor_list(
                xyz, atype, self.rcut, self.sel, self.dd0.mixed_types(), box=self.cell
            )

            rd0, gr0, _, _, _ = self.dd0(
                extended_coord,
                extended_atype,
                nlist,
            )

            ret0 = ft0(rd0, atype, gr0, fparam=None, aparam=None)
            res.append(ret0["dipole"])

        np.testing.assert_allclose(to_numpy_array(res[0]), to_numpy_array(res[1]))


class TestDipoleModel(unittest.TestCase):
    def setUp(self) -> None:
        self.natoms = 5
        self.rcut = 4.0
        self.nt = 3
        self.rcut_smth = 0.5
        self.sel = [46, 92, 4]
        self.nf = 1
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        self.coord = 2 * torch.rand(
            [self.natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE, generator=generator)
        self.cell = (cell + cell.T) + 5.0 * torch.eye(3, device=env.DEVICE)
        self.atype = torch.IntTensor([0, 0, 0, 1, 1], device="cpu").to(env.DEVICE)
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.ft0 = DipoleFittingNet(
            self.nt,
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=self.dd0.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        self.type_mapping = ["O", "H", "B"]
        self.model = DipoleModel(self.dd0, self.ft0, self.type_mapping)
        self.file_path = "model_output.pth"

    def test_auto_diff(self) -> None:
        places = 5
        delta = 1e-5
        atype = self.atype.view(self.nf, self.natoms)

        def ff(coord, atype):
            return (
                self.model(to_torch_tensor(coord), to_torch_tensor(atype))[
                    "global_dipole"
                ]
                .detach()
                .cpu()
                .numpy()
            )

        fdf = -finite_difference(
            ff, to_numpy_array(self.coord), to_numpy_array(atype), delta=delta
        )
        rff = self.model(self.coord, atype)["force"].detach().cpu().numpy()

        np.testing.assert_almost_equal(fdf, rff.transpose(0, 2, 1, 3), decimal=places)

    def test_deepdipole_infer(self) -> None:
        atype = to_numpy_array(self.atype.view(self.nf, self.natoms))
        coord = to_numpy_array(self.coord.reshape(1, 5, 3))
        cell = to_numpy_array(self.cell.reshape(1, 9))
        jit_md = torch.jit.script(self.model)
        torch.jit.save(jit_md, self.file_path)
        load_md = DeepDipole(self.file_path)
        load_md.eval(coords=coord, atom_types=atype, cells=cell, atomic=True)
        load_md.eval(coords=coord, atom_types=atype, cells=cell, atomic=False)
        load_md.eval_full(coords=coord, atom_types=atype, cells=cell, atomic=True)
        load_md.eval_full(coords=coord, atom_types=atype, cells=cell, atomic=False)

    def tearDown(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


if __name__ == "__main__":
    unittest.main()
