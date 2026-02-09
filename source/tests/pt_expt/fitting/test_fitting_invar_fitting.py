# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import (
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
)
from ...seed import (
    GLOBAL_SEED,
)


class TestInvarFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def test_self_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

        for (
            mixed_types,
            od,
            nfp,
            nap,
            et,
        ) in itertools.product(
            [True, False],
            [1, 2],
            [0, 3],
            [0, 4],
            [[], [0], [1]],
        ):
            ifn0 = InvarFitting(
                "energy",
                self.nt,
                ds.dim_out,
                od,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
                exclude_types=et,
            ).to(self.device)
            ifn1 = InvarFitting.deserialize(ifn0.serialize()).to(self.device)
            if nfp > 0:
                ifp = torch.from_numpy(rng.normal(size=(self.nf, nfp))).to(self.device)
            else:
                ifp = None
            if nap > 0:
                iap = torch.from_numpy(rng.normal(size=(self.nf, self.nloc, nap))).to(
                    self.device
                )
            else:
                iap = None
            ret0 = ifn0(
                torch.from_numpy(dd[0]).to(self.device),
                torch.from_numpy(atype).to(self.device),
                fparam=ifp,
                aparam=iap,
            )
            ret1 = ifn1(
                torch.from_numpy(dd[0]).to(self.device),
                torch.from_numpy(atype).to(self.device),
                fparam=ifp,
                aparam=iap,
            )
            np.testing.assert_allclose(
                ret0["energy"].detach().cpu().numpy(),
                ret1["energy"].detach().cpu().numpy(),
            )
            sel_set = set(ifn0.get_sel_type())
            exclude_set = set(et)
            self.assertEqual(sel_set | exclude_set, set(range(self.nt)))
            self.assertEqual(sel_set & exclude_set, set())

    def test_mask(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]
        od = 2
        mixed_types = True
        # exclude type 1
        et = [1]
        ifn0 = InvarFitting(
            "energy",
            self.nt,
            ds.dim_out,
            od,
            mixed_types=mixed_types,
            exclude_types=et,
        ).to(self.device)
        ret0 = ifn0(
            torch.from_numpy(dd[0]).to(self.device),
            torch.from_numpy(atype).to(self.device),
        )
        # atom index 2 is of type 1 that is excluded
        zero_idx = 2
        np.testing.assert_allclose(
            ret0["energy"][0, zero_idx, :].detach().cpu().numpy(),
            np.zeros_like(ret0["energy"][0, zero_idx, :].detach().cpu().numpy()),
        )
        zero_idx = 0
        np.testing.assert_allclose(
            ret0["energy"][1, zero_idx, :].detach().cpu().numpy(),
            np.zeros_like(ret0["energy"][1, zero_idx, :].detach().cpu().numpy()),
        )

    def test_self_exception(
        self,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

        for (
            mixed_types,
            od,
            nfp,
            nap,
        ) in itertools.product(
            [True, False],
            [1, 2],
            [0, 3],
            [0, 4],
        ):
            ifn0 = InvarFitting(
                "energy",
                self.nt,
                ds.dim_out,
                od,
                numb_fparam=nfp,
                numb_aparam=nap,
                mixed_types=mixed_types,
            ).to(self.device)

            if nfp > 0:
                ifp = torch.from_numpy(rng.normal(size=(self.nf, nfp))).to(self.device)
            else:
                ifp = None
            if nap > 0:
                iap = torch.from_numpy(rng.normal(size=(self.nf, self.nloc, nap))).to(
                    self.device
                )
            else:
                iap = None
            with self.assertRaises(ValueError) as context:
                ret0 = ifn0(
                    torch.from_numpy(dd[0][:, :, :-2]).to(self.device),
                    torch.from_numpy(atype).to(self.device),
                    fparam=ifp,
                    aparam=iap,
                )
                self.assertIn("input descriptor", str(context.exception))

            if nfp > 0:
                ifp = torch.from_numpy(rng.normal(size=(self.nf, nfp - 1))).to(
                    self.device
                )
                with self.assertRaises(ValueError) as context:
                    ret0 = ifn0(
                        torch.from_numpy(dd[0]).to(self.device),
                        torch.from_numpy(atype).to(self.device),
                        fparam=ifp,
                        aparam=iap,
                    )
                    self.assertIn("input fparam", str(context.exception))

            if nap > 0:
                iap = torch.from_numpy(
                    rng.normal(size=(self.nf, self.nloc, nap - 1))
                ).to(self.device)
                with self.assertRaises(ValueError) as context:
                    ifn0(
                        torch.from_numpy(dd[0]).to(self.device),
                        torch.from_numpy(atype).to(self.device),
                        fparam=ifp,
                        aparam=iap,
                    )
                    self.assertIn("input aparam", str(context.exception))

    def test_get_set(self) -> None:
        ifn0 = InvarFitting(
            "energy",
            self.nt,
            3,
            1,
        ).to(self.device)
        rng = np.random.default_rng(GLOBAL_SEED)
        foo = rng.normal([3, 4])
        for ii in [
            "bias_atom_e",
            "fparam_avg",
            "fparam_inv_std",
            "aparam_avg",
            "aparam_inv_std",
        ]:
            ifn0[ii] = torch.from_numpy(foo).to(self.device)
            np.testing.assert_allclose(
                foo, ifn0[ii].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
            )

    def test_torch_export_simple(self) -> None:
        """Test that InvarFitting can be exported with torch.export."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        rng = np.random.default_rng(GLOBAL_SEED)

        ifn = InvarFitting(
            "energy",
            self.nt,
            ds.dim_out,
            1,
            numb_fparam=0,
            numb_aparam=0,
            mixed_types=True,
        ).to(self.device)

        # Prepare inputs
        descriptor = torch.from_numpy(
            rng.standard_normal((self.nf, self.nloc, ds.dim_out))
        ).to(self.device)
        atype = torch.from_numpy(self.atype_ext[:, :nloc]).to(self.device)

        # Test forward pass works
        ret = ifn(descriptor, atype)
        self.assertIn("energy", ret)

        # Test torch.export
        exported = torch.export.export(
            ifn,
            (descriptor, atype),
            kwargs={},
            strict=False,  # Use strict=False for now to handle dynamic shapes
        )
        self.assertIsNotNone(exported)

        # Test exported model produces same output
        ret_exported = exported.module()(descriptor, atype)
        np.testing.assert_allclose(
            ret["energy"].detach().cpu().numpy(),
            ret_exported["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_torch_export_with_fparam(self) -> None:
        """Test that InvarFitting with fparam can be exported."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        rng = np.random.default_rng(GLOBAL_SEED)

        ifn = InvarFitting(
            "energy",
            self.nt,
            ds.dim_out,
            1,
            numb_fparam=3,
            numb_aparam=0,
            mixed_types=True,
        ).to(self.device)

        # Prepare inputs
        descriptor = torch.from_numpy(
            rng.normal(size=(self.nf, self.nloc, ds.dim_out))
        ).to(self.device)
        atype = torch.from_numpy(self.atype_ext[:, :nloc]).to(self.device)
        fparam = torch.from_numpy(rng.normal(size=(self.nf, 3))).to(self.device)

        # Test forward pass works
        ret = ifn(descriptor, atype, fparam=fparam)
        self.assertIn("energy", ret)

        # Test torch.export
        exported = torch.export.export(
            ifn,
            (descriptor, atype),
            kwargs={"fparam": fparam},
            strict=False,
        )
        self.assertIsNotNone(exported)

        # Test exported model produces same output
        ret_exported = exported.module()(descriptor, atype, fparam=fparam)
        np.testing.assert_allclose(
            ret["energy"].detach().cpu().numpy(),
            ret_exported["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
