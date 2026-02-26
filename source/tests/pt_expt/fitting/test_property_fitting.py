# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

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


class TestPropertyFittingNet(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("nfp", [0, 3])  # numb_fparam
    @pytest.mark.parametrize("nap", [0, 4])  # numb_aparam
    def test_self_consistency(self, nfp, nap) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        dd = ds.call(self.coord_ext, self.atype_ext, self.nlist)
        atype = self.atype_ext[:, :nloc]

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
        assert serialized["type"] == "property"
        fn2 = PropertyFittingNet.deserialize(serialized).to(self.device)
        assert isinstance(fn2, PropertyFittingNet)

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
        assert "property" in ret

        exported = torch.export.export(
            fn,
            (descriptor, atype),
            kwargs={},
            strict=False,
        )
        assert exported is not None

        ret_exported = exported.module()(descriptor, atype)
        np.testing.assert_allclose(
            ret["property"].detach().cpu().numpy(),
            ret_exported["property"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_make_fx(self) -> None:
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        rng = np.random.default_rng(GLOBAL_SEED)

        fn0 = (
            PropertyFittingNet(
                self.nt,
                ds.dim_out,
                task_dim=3,
                precision="float64",
            )
            .to(self.device)
            .eval()
        )

        descriptor = torch.from_numpy(
            rng.standard_normal((self.nf, self.nloc, ds.dim_out))
        ).to(self.device)
        atype = torch.from_numpy(self.atype_ext[:, :nloc]).to(self.device)

        def fn(descriptor, atype):
            descriptor = descriptor.detach().requires_grad_(True)
            ret = fn0(descriptor, atype)["property"]
            grad = torch.autograd.grad(ret.sum(), descriptor, create_graph=False)[0]
            return ret, grad

        ret_eager, grad_eager = fn(descriptor, atype)
        traced = make_fx(fn)(descriptor, atype)
        ret_traced, grad_traced = traced(descriptor, atype)
        np.testing.assert_allclose(
            ret_eager.detach().cpu().numpy(),
            ret_traced.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            grad_eager.detach().cpu().numpy(),
            grad_traced.detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
