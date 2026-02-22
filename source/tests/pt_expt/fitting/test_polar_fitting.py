# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    PolarFitting,
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


class TestPolarFitting(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def test_self_consistency(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

        embedding_width = ds.get_dim_emb()

        for nfp, nap in [(0, 0), (3, 0), (0, 4), (3, 4)]:
            fn0 = PolarFitting(
                self.nt,
                ds.dim_out,
                embedding_width,
                numb_fparam=nfp,
                numb_aparam=nap,
            ).to(self.device)
            fn1 = PolarFitting.deserialize(fn0.serialize()).to(self.device)
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
                gr=torch.from_numpy(dd[1]).to(self.device),
                fparam=ifp,
                aparam=iap,
            )
            ret1 = fn1(
                torch.from_numpy(dd[0]).to(self.device),
                torch.from_numpy(atype).to(self.device),
                gr=torch.from_numpy(dd[1]).to(self.device),
                fparam=ifp,
                aparam=iap,
            )
            np.testing.assert_allclose(
                ret0["polarizability"].detach().cpu().numpy(),
                ret1["polarizability"].detach().cpu().numpy(),
            )

    def test_serialize_has_correct_type(self) -> None:
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        embedding_width = ds.get_dim_emb()
        fn = PolarFitting(
            self.nt,
            ds.dim_out,
            embedding_width,
        ).to(self.device)
        serialized = fn.serialize()
        self.assertEqual(serialized["type"], "polar")
        fn2 = PolarFitting.deserialize(serialized).to(self.device)
        self.assertIsInstance(fn2, PolarFitting)

    def test_torch_export_simple(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        embedding_width = ds.get_dim_emb()

        fn = PolarFitting(
            self.nt,
            ds.dim_out,
            embedding_width,
            numb_fparam=0,
            numb_aparam=0,
        ).to(self.device)

        descriptor = torch.from_numpy(dd[0]).to(self.device)
        atype = torch.from_numpy(self.atype_ext[:, :nloc]).to(self.device)
        gr = torch.from_numpy(dd[1]).to(self.device)

        ret = fn(descriptor, atype, gr=gr)
        self.assertIn("polarizability", ret)

        exported = torch.export.export(
            fn,
            (descriptor, atype),
            kwargs={"gr": gr},
            strict=False,
        )
        self.assertIsNotNone(exported)

        ret_exported = exported.module()(descriptor, atype, gr=gr)
        np.testing.assert_allclose(
            ret["polarizability"].detach().cpu().numpy(),
            ret_exported["polarizability"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
