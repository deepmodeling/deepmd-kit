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

    def test_eval_shuffle_sel_type(self) -> None:
        # Build a model where only type-0 atoms contribute (exclude types 1 and 2).
        # This tests that eval() returns per-atom results in the correct input atom
        # order even when sel_type is a strict subset of all types.
        ft_sel = DipoleFittingNet(
            self.nt,
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=self.dd0.mixed_types(),
            exclude_types=[1, 2],
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        model_sel = DipoleModel(self.dd0, ft_sel, self.type_mapping)
        jit_md = torch.jit.script(model_sel)
        torch.jit.save(jit_md, self.file_path)
        load_md = DeepDipole(self.file_path)

        atype = to_numpy_array(self.atype)  # [0, 0, 0, 1, 1]
        coord = to_numpy_array(self.coord.reshape(1, self.natoms, 3))
        cell = to_numpy_array(self.cell.reshape(1, 9))

        # Reference result with original atom order
        ref = load_md.eval(coords=coord, atom_types=atype, cells=cell, atomic=True)
        # ref shape: [nframes, natoms, nout]

        # Shuffle atoms
        idx_perm = [1, 0, 4, 3, 2]
        coord_sf = coord.reshape(self.natoms, 3)[idx_perm].reshape(1, -1)
        atype_sf = atype[idx_perm]

        # Result with shuffled atom order
        res_sf = load_md.eval(
            coords=coord_sf, atom_types=atype_sf, cells=cell, atomic=True
        )
        # res_sf shape: [nframes, natoms, nout]

        # sel_mask: which atoms in the original order are selected (type 0)
        sel_mask = np.isin(atype, load_md.get_sel_type())  # [T,T,T,F,F]
        sel_mask_sf = sel_mask[idx_perm]  # selected atoms in shuffled order

        # Extract selected-atom outputs from each result
        ref_sel = ref[:, sel_mask]  # [nframes, nsel, nout]
        at_sf = res_sf[:, sel_mask_sf]  # [nframes, nsel, nout]

        # isel_sf: mapping from shuffled-selected positions to original-selected positions
        orig_sel_idx = np.where(sel_mask)[0]
        shuffled_orig = np.array(idx_perm)[sel_mask_sf]
        isel_sf = np.array(
            [np.where(orig_sel_idx == x)[0][0] for x in shuffled_orig]
        )  # [1, 0, 2]

        # Recover original selected order from shuffled selected
        nat = np.empty_like(at_sf)
        nat[:, isel_sf] = at_sf

        np.testing.assert_almost_equal(
            nat.reshape([-1]), ref_sel.reshape([-1]), decimal=10
        )

    def test_label_order_via_deepmd_data(self) -> None:
        """Verify that labels loaded via DeepmdData(sort_atoms=False) +
        output_natoms_for_type_sel=True align with dp.eval() output.
        Uses a sel_type model (exclude_types=[1,2]) with atype=[0,0,0,1,1]
        shuffled to [0,0,1,1,0] so selected atoms are non-contiguous.
        """
        import shutil
        import tempfile

        from deepmd.utils.data import (
            DeepmdData,
        )

        ft_sel = DipoleFittingNet(
            self.nt,
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=self.dd0.mixed_types(),
            exclude_types=[1, 2],
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        model_sel = DipoleModel(self.dd0, ft_sel, self.type_mapping)
        jit_md = torch.jit.script(model_sel)
        torch.jit.save(jit_md, self.file_path)
        load_md = DeepDipole(self.file_path)

        # Shuffle atoms so selected type-0 atoms are non-contiguous
        # atype=[0,0,0,1,1] → shuffled idx → atype=[0,0,1,1,0]
        idx_perm = np.array([1, 0, 4, 3, 2], dtype=np.intp)
        atype = to_numpy_array(self.atype)  # [0,0,0,1,1]
        coord = to_numpy_array(self.coord.reshape(1, self.natoms, 3))
        cell = to_numpy_array(self.cell.reshape(1, 9))
        atype_sf = atype[idx_perm]
        coord_sf = coord.reshape(self.natoms, 3)[idx_perm].reshape(1, -1)

        sel_mask_sf = np.isin(atype_sf, load_md.get_sel_type())  # type-0 positions

        # Reference: model output for shuffled atoms, filter to sel atoms
        ref_sf = load_md.eval(
            coords=coord_sf, atom_types=atype_sf, cells=cell, atomic=True
        )  # [1, natoms, nout]
        ref_sf_sel = ref_sf[:, sel_mask_sf, :]  # [1, nsel, nout]

        tmpdir = tempfile.mkdtemp()
        try:
            set_dir = os.path.join(tmpdir, "set.000")
            os.makedirs(set_dir)
            np.savetxt(os.path.join(tmpdir, "type.raw"), atype_sf, fmt="%d")
            np.save(
                os.path.join(set_dir, "coord.npy"),
                coord_sf.reshape(1, -1),
            )
            np.save(
                os.path.join(set_dir, "box.npy"),
                cell.reshape(1, -1),
            )
            # Labels: nsel atoms in shuffled atom order (nsel format)
            np.save(
                os.path.join(set_dir, "atomic_dipole.npy"),
                ref_sf_sel.reshape(1, -1),
            )

            data = DeepmdData(
                tmpdir,
                set_prefix="set",
                shuffle_test=False,
                type_map=load_md.get_type_map(),
                sort_atoms=False,
            )
            data.add(
                "atomic_dipole",
                3,
                atomic=True,
                must=True,
                high_prec=False,
                type_sel=load_md.get_sel_type(),
                output_natoms_for_type_sel=True,
            )
            test_data = data.get_test()

            # Loaded label shape: [1, natoms*3]. Filter to sel atoms.
            label_sel = test_data["atom_dipole"].reshape(1, self.natoms, 3)[
                :, sel_mask_sf, :
            ]  # [1, nsel, 3]

            # Round-trip: loaded label must match what was written
            np.testing.assert_almost_equal(
                label_sel.reshape(-1), ref_sf_sel.reshape(-1), decimal=5
            )
        finally:
            shutil.rmtree(tmpdir)

    def tearDown(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)


if __name__ == "__main__":
    unittest.main()
