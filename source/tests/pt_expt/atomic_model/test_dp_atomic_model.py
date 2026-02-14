# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.atomic_model import DPAtomicModel as DPDPAtomicModel
from deepmd.dpmodel.descriptor import DescrptSeA as DPDescrptSeA
from deepmd.dpmodel.fitting import InvarFitting as DPInvarFitting
from deepmd.pt_expt.atomic_model import (
    DPAtomicModel,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    InvarFitting,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...pt.model.test_env_mat import (
    TestCaseSingleFrameWithNlist,
    TestCaseSingleFrameWithNlistWithVirtual,
)
from ...seed import (
    GLOBAL_SEED,
)


class TestDPAtomicModel(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def test_self_consistency(self) -> None:
        """Test that pt_expt atomic model serialize/deserialize preserves behavior."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(self.device)
        type_map = ["foo", "bar"]

        # test the case of exclusion
        for atom_excl, pair_excl in itertools.product([[], [1]], [[], [[0, 1]]]):
            md0 = DPAtomicModel(
                ds,
                ft,
                type_map=type_map,
            ).to(self.device)
            md0.reinit_atom_exclude(atom_excl)
            md0.reinit_pair_exclude(pair_excl)
            md1 = DPAtomicModel.deserialize(md0.serialize()).to(self.device)

            # Test forward pass
            args = [
                torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
                torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
                torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
            ]
            ret0 = md0.forward_common_atomic(*args)
            ret1 = md1.forward_common_atomic(*args)
            np.testing.assert_allclose(
                ret0["energy"].detach().cpu().numpy(),
                ret1["energy"].detach().cpu().numpy(),
            )

    def test_dp_consistency(self) -> None:
        """Test numerical consistency between dpmodel and pt_expt atomic models."""
        nf, nloc, nnei = self.nlist.shape
        ds = DPDescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        )
        ft = DPInvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        )
        type_map = ["foo", "bar"]
        md0 = DPDPAtomicModel(ds, ft, type_map=type_map)
        md1 = DPAtomicModel.deserialize(md0.serialize()).to(self.device)

        # dpmodel uses numpy arrays
        args0 = [self.coord_ext, self.atype_ext, self.nlist]
        # pt_expt uses torch tensors
        args1 = [
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
        ]
        ret0 = md0.forward_common_atomic(*args0)
        ret1 = md1.forward_common_atomic(*args1)
        np.testing.assert_allclose(
            ret0["energy"],
            ret1["energy"].detach().cpu().numpy(),
        )

    def test_exportable(self) -> None:
        """Test that pt_expt atomic model can be exported with torch.export."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
        ).to(self.device)
        ft = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(self.device)
        type_map = ["foo", "bar"]
        md0 = DPAtomicModel(ds, ft, type_map=type_map).to(self.device)
        md0 = md0.eval()

        # Prepare inputs for export
        coord = torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device)
        atype = torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=torch.int64, device=self.device)

        # Test forward pass
        ret0 = md0(coord, atype, nlist)
        self.assertIn("energy", ret0)

        # Test torch.export
        # Use strict=False for now to handle dynamic shapes
        exported = torch.export.export(
            md0,
            (coord, atype, nlist),
            strict=False,
        )
        self.assertIsNotNone(exported)

        # Test exported model produces same output
        ret1 = exported.module()(coord, atype, nlist)
        np.testing.assert_allclose(
            ret0["energy"].detach().cpu().numpy(),
            ret1["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_excl_consistency(self) -> None:
        """Test that exclusion masks work correctly after serialize/deserialize."""
        type_map = ["foo", "bar"]

        # test the case of exclusion
        for atom_excl, pair_excl in itertools.product([[], [1]], [[], [[0, 1]]]):
            ds = DescrptSeA(
                self.rcut,
                self.rcut_smth,
                self.sel,
            ).to(self.device)
            ft = InvarFitting(
                "energy",
                self.nt,
                ds.get_dim_out(),
                1,
                mixed_types=ds.mixed_types(),
                seed=GLOBAL_SEED,
            ).to(self.device)
            md0 = DPAtomicModel(
                ds,
                ft,
                type_map=type_map,
            ).to(self.device)
            md1 = DPAtomicModel.deserialize(md0.serialize()).to(self.device)

            md0.reinit_atom_exclude(atom_excl)
            md0.reinit_pair_exclude(pair_excl)
            # hacking!
            md1.descriptor.reinit_exclude(pair_excl)
            md1.fitting.reinit_exclude(atom_excl)

            # check energy consistency
            args = [
                torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
                torch.tensor(self.atype_ext, dtype=torch.int64, device=self.device),
                torch.tensor(self.nlist, dtype=torch.int64, device=self.device),
            ]
            ret0 = md0.forward_common_atomic(*args)
            ret1 = md1.forward_common_atomic(*args)
            np.testing.assert_allclose(
                ret0["energy"].detach().cpu().numpy(),
                ret1["energy"].detach().cpu().numpy(),
            )

            # check output def
            out_names = [vv.name for vv in md0.atomic_output_def().get_data().values()]
            self.assertEqual(out_names, ["energy", "mask"])
            if atom_excl != []:
                for ii in md0.atomic_output_def().get_data().values():
                    if ii.name == "mask":
                        self.assertEqual(ii.shape, [1])
                        self.assertFalse(ii.reducible)
                        self.assertFalse(ii.r_differentiable)
                        self.assertFalse(ii.c_differentiable)

            # check mask
            if atom_excl == []:
                pass
            elif atom_excl == [1]:
                self.assertIn("mask", ret0.keys())
                expected = np.array([1, 1, 0], dtype=int)
                expected = np.concatenate(
                    [expected, expected[self.perm[: self.nloc]]]
                ).reshape(2, 3)
                np.testing.assert_array_equal(
                    ret0["mask"].detach().cpu().numpy(), expected
                )
            else:
                raise ValueError(f"not expected atom_excl {atom_excl}")


class TestDPAtomicModelVirtualConsistency(unittest.TestCase):
    def setUp(self) -> None:
        self.case0 = TestCaseSingleFrameWithNlist()
        self.case1 = TestCaseSingleFrameWithNlistWithVirtual()
        self.case0.setUp()
        self.case1.setUp()
        self.device = env.DEVICE

    def test_virtual_consistency(self) -> None:
        nf, _, _ = self.case0.nlist.shape
        ds = DescrptSeA(
            self.case0.rcut,
            self.case0.rcut_smth,
            self.case0.sel,
        )
        ft = InvarFitting(
            "energy",
            self.case0.nt,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        )
        type_map = ["foo", "bar"]
        md1 = DPAtomicModel(ds, ft, type_map=type_map).to(self.device)

        args0 = [
            torch.tensor(self.case0.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.case0.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.case0.nlist, dtype=torch.int64, device=self.device),
        ]
        args1 = [
            torch.tensor(self.case1.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.case1.atype_ext, dtype=torch.int64, device=self.device),
            torch.tensor(self.case1.nlist, dtype=torch.int64, device=self.device),
        ]

        ret0 = md1.forward_common_atomic(*args0)
        ret1 = md1.forward_common_atomic(*args1)

        for dd in range(self.case0.nf):
            np.testing.assert_allclose(
                ret0["energy"][dd].detach().cpu().numpy(),
                ret1["energy"][dd, self.case1.get_real_mapping[dd], :]
                .detach()
                .cpu()
                .numpy(),
            )
        expected_mask = np.array(
            [
                [1, 0, 1, 1],
                [1, 1, 0, 1],
            ]
        )
        np.testing.assert_equal(ret1["mask"].detach().cpu().numpy(), expected_mask)
