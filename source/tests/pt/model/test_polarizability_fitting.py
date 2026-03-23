# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import os
import unittest

import numpy as np
import torch
from scipy.stats import (
    special_ortho_group,
)

from deepmd.dpmodel.fitting import PolarFitting as DPPolarFitting
from deepmd.infer.deep_polar import (
    DeepPolar,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model.polar_model import (
    PolarModel,
)
from deepmd.pt.model.task.polarizability import (
    PolarFittingNet,
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

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestPolarFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.rng = np.random.default_rng(GLOBAL_SEED)
        self.nf, self.nloc, _ = self.nlist.shape
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.scale = self.rng.uniform(0, 1, self.nt).tolist()

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

        for nfp, nap, fit_diag, scale in itertools.product(
            [0, 3],
            [0, 4],
            [True, False],
            [None, self.scale],
        ):
            ft0 = PolarFittingNet(
                self.nt,
                self.dd0.dim_out,
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=self.dd0.mixed_types(),
                fit_diag=fit_diag,
                scale=scale,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)
            ft1 = DPPolarFitting.deserialize(ft0.serialize())
            ft2 = PolarFittingNet.deserialize(ft0.serialize())
            ft3 = DPPolarFitting.deserialize(ft1.serialize())

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
                to_numpy_array(ret0["polarizability"]),
                ret1["polarizability"],
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["polarizability"]),
                to_numpy_array(ret2["polarizability"]),
            )
            np.testing.assert_allclose(
                to_numpy_array(ret0["polarizability"]),
                ret3["polarizability"],
            )

    def test_jit(
        self,
    ) -> None:
        for mixed_types, nfp, nap, fit_diag in itertools.product(
            [True, False],
            [0, 3],
            [0, 4],
            [True, False],
        ):
            ft0 = PolarFittingNet(
                self.nt,
                self.dd0.dim_out,
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
                fit_diag=fit_diag,
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
        self.nt = 3
        self.rng = np.random.default_rng(GLOBAL_SEED)
        self.coord = 2 * torch.rand([self.natoms, 3], dtype=dtype, device=env.DEVICE)
        self.shift = torch.tensor([4, 4, 4], dtype=dtype, device=env.DEVICE)
        self.atype = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=env.DEVICE)
        self.dd0 = DescrptSeA(self.rcut, self.rcut_smth, self.sel).to(env.DEVICE)
        self.cell = torch.rand([3, 3], dtype=dtype, device=env.DEVICE)
        self.cell = (self.cell + self.cell.T) + 5.0 * torch.eye(3, device=env.DEVICE)
        self.scale = self.rng.uniform(0, 1, self.nt).tolist()

    def test_rot(self) -> None:
        atype = self.atype.reshape(1, 5)
        rmat = torch.tensor(special_ortho_group.rvs(3), dtype=dtype, device=env.DEVICE)
        coord_rot = torch.matmul(self.coord, rmat)
        # use larger cell to rotate only coord and shift to the center of cell
        cell_rot = 10.0 * torch.eye(3, dtype=dtype, device=env.DEVICE)

        for nfp, nap, fit_diag, scale in itertools.product(
            [0, 3],
            [0, 4],
            [True, False],
            [None, self.scale],
        ):
            ft0 = PolarFittingNet(
                self.nt,
                self.dd0.dim_out,  # dim_descrpt
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=self.dd0.mixed_types(),
                fit_diag=fit_diag,
                scale=scale,
                seed=GLOBAL_SEED,
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
                res.append(ret0["polarizability"])
            np.testing.assert_allclose(
                to_numpy_array(res[1]),
                to_numpy_array(
                    torch.matmul(
                        rmat.T,
                        torch.matmul(res[0], rmat),
                    )
                ),
            )

    def test_permu(self) -> None:
        coord = torch.matmul(self.coord, self.cell)
        for fit_diag, scale in itertools.product([True, False], [None, self.scale]):
            ft0 = PolarFittingNet(
                self.nt,
                self.dd0.dim_out,
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=0,
                numb_aparam=0,
                mixed_types=self.dd0.mixed_types(),
                fit_diag=fit_diag,
                scale=scale,
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
                res.append(ret0["polarizability"])

            np.testing.assert_allclose(
                to_numpy_array(res[0][:, idx_perm]),
                to_numpy_array(res[1]),
            )

    def test_trans(self) -> None:
        atype = self.atype.reshape(1, 5)
        coord_s = torch.matmul(
            torch.remainder(
                torch.matmul(self.coord + self.shift, torch.linalg.inv(self.cell)), 1.0
            ),
            self.cell,
        )
        for fit_diag, scale in itertools.product([True, False], [None, self.scale]):
            ft0 = PolarFittingNet(
                self.nt,
                self.dd0.dim_out,
                embedding_width=self.dd0.get_dim_emb(),
                numb_fparam=0,
                numb_aparam=0,
                mixed_types=self.dd0.mixed_types(),
                fit_diag=fit_diag,
                scale=scale,
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
                    xyz,
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
                res.append(ret0["polarizability"])

            np.testing.assert_allclose(to_numpy_array(res[0]), to_numpy_array(res[1]))


class TestPolarModel(unittest.TestCase):
    def setUp(self) -> None:
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
        self.ft0 = PolarFittingNet(
            self.nt,
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=self.dd0.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        self.type_mapping = ["O", "H", "B"]
        self.model = PolarModel(self.dd0, self.ft0, self.type_mapping)
        self.file_path = "model_output.pth"

    def test_deepdipole_infer(self) -> None:
        atype = self.atype.view(self.nf, self.natoms)
        coord = self.coord.reshape(1, 5, 3)
        cell = self.cell.reshape(1, 9)
        jit_md = torch.jit.script(self.model)
        torch.jit.save(jit_md, self.file_path)
        load_md = DeepPolar(self.file_path)
        load_md.eval(coords=coord, atom_types=atype, cells=cell, atomic=True)
        load_md.eval(coords=coord, atom_types=atype, cells=cell, atomic=False)

    def test_eval_shuffle_sel_type(self) -> None:
        # Build a model where only type-0 atoms contribute (exclude types 1 and 2).
        # This tests that eval() returns per-atom results in the correct input atom
        # order even when sel_type is a strict subset of all types.
        ft_sel = PolarFittingNet(
            self.nt,
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=self.dd0.mixed_types(),
            exclude_types=[1, 2],
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        model_sel = PolarModel(self.dd0, ft_sel, self.type_mapping)
        jit_md = torch.jit.script(model_sel)
        torch.jit.save(jit_md, self.file_path)
        load_md = DeepPolar(self.file_path)

        atype = self.atype.numpy()  # [0, 0, 0, 1, 1]
        coord = self.coord.reshape(1, self.natoms, 3).numpy()
        cell = self.cell.reshape(1, 9).numpy()

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
        # Original selected atom indices: [0, 1, 2] (type-0 atoms)
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

        ft_sel = PolarFittingNet(
            self.nt,
            self.dd0.dim_out,
            embedding_width=self.dd0.get_dim_emb(),
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=self.dd0.mixed_types(),
            exclude_types=[1, 2],
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        model_sel = PolarModel(self.dd0, ft_sel, self.type_mapping)
        jit_md = torch.jit.script(model_sel)
        torch.jit.save(jit_md, self.file_path)
        load_md = DeepPolar(self.file_path)

        # Shuffle atoms so selected type-0 atoms are non-contiguous
        # atype=[0,0,0,1,1] → shuffled idx → atype=[0,0,1,1,0]
        idx_perm = [1, 0, 4, 3, 2]
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
                coord_sf.reshape(1, -1).astype(np.float64),
            )
            np.save(
                os.path.join(set_dir, "box.npy"),
                cell.reshape(1, -1).astype(np.float64),
            )
            # Labels: nsel atoms in shuffled atom order (nsel format)
            np.save(
                os.path.join(set_dir, "atomic_polarizability.npy"),
                ref_sf_sel.reshape(1, -1).astype(np.float32),
            )

            data = DeepmdData(
                tmpdir,
                set_prefix="set",
                shuffle_test=False,
                type_map=load_md.get_type_map(),
                sort_atoms=False,
            )
            data.add(
                "atomic_polarizability",
                9,
                atomic=True,
                must=True,
                high_prec=False,
                type_sel=load_md.get_sel_type(),
                output_natoms_for_type_sel=True,
            )
            test_data = data.get_test()

            # Loaded label shape: [1, natoms*9]. Filter to sel atoms.
            label_sel = test_data["atom_polarizability"].reshape(1, self.natoms, 9)[
                :, sel_mask_sf, :
            ]  # [1, nsel, 9]

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
