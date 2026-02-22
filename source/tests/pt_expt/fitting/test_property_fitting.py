# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    PropertyFittingNet,
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


class TestPropertyFittingNet(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def test_self_consistency(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

        for nfp, nap in [(0, 0), (3, 0), (0, 4), (3, 4)]:
            fn0 = PropertyFittingNet(
                self.nt,
                ds.dim_out,
                task_dim=3,
                numb_fparam=nfp,
                numb_aparam=nap,
            ).to(self.device)
            fn1 = PropertyFittingNet.deserialize(fn0.serialize()).to(self.device)
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
            ret0 = fn0(
                torch.from_numpy(dd[0]).to(self.device),
                torch.from_numpy(atype).to(self.device),
                fparam=ifp,
                aparam=iap,
            )
            ret1 = fn1(
                torch.from_numpy(dd[0]).to(self.device),
                torch.from_numpy(atype).to(self.device),
                fparam=ifp,
                aparam=iap,
            )
            np.testing.assert_allclose(
                ret0["property"].detach().cpu().numpy(),
                ret1["property"].detach().cpu().numpy(),
            )

    def test_serialize_has_correct_type(self) -> None:
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        fn = PropertyFittingNet(
            self.nt,
            ds.dim_out,
            task_dim=3,
        ).to(self.device)
        serialized = fn.serialize()
        self.assertEqual(serialized["type"], "property")
        fn2 = PropertyFittingNet.deserialize(serialized).to(self.device)
        self.assertIsInstance(fn2, PropertyFittingNet)

    def test_torch_export_simple(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        rng = np.random.default_rng(GLOBAL_SEED)

        fn = PropertyFittingNet(
            self.nt,
            ds.dim_out,
            task_dim=3,
            numb_fparam=0,
            numb_aparam=0,
        ).to(self.device)

        descriptor = torch.from_numpy(
            rng.standard_normal((self.nf, self.nloc, ds.dim_out))
        ).to(self.device)
        atype = torch.from_numpy(self.atype_ext[:, :nloc]).to(self.device)

        ret = fn(descriptor, atype)
        self.assertIn("property", ret)

        exported = torch.export.export(
            fn,
            (descriptor, atype),
            kwargs={},
            strict=False,
        )
        self.assertIsNotNone(exported)

        ret_exported = exported.module()(descriptor, atype)
        np.testing.assert_allclose(
            ret["property"].detach().cpu().numpy(),
            ret_exported["property"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
