# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    EnergyFittingNet,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...common.test_mixins import (
    TestCaseSingleFrameWithNlist,
)
from ...seed import (
    GLOBAL_SEED,
)


class TestEnergyFittingNet(unittest.TestCase, TestCaseSingleFrameWithNlist):
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

        for nfp, nap in [(0, 0), (3, 0), (0, 4), (3, 4)]:
            efn0 = EnergyFittingNet(
                self.nt,
                ds.dim_out,
                numb_fparam=nfp,
                numb_aparam=nap,
            ).to(self.device)
            efn1 = EnergyFittingNet.deserialize(efn0.serialize()).to(self.device)
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
            ret0 = efn0(
                torch.from_numpy(dd[0]).to(self.device),
                torch.from_numpy(atype).to(self.device),
                fparam=ifp,
                aparam=iap,
            )
            ret1 = efn1(
                torch.from_numpy(dd[0]).to(self.device),
                torch.from_numpy(atype).to(self.device),
                fparam=ifp,
                aparam=iap,
            )
            np.testing.assert_allclose(
                ret0["energy"].detach().cpu().numpy(),
                ret1["energy"].detach().cpu().numpy(),
            )

    def test_serialize_has_correct_type(self) -> None:
        """Test that EnergyFittingNet serializes with type='ener' not 'invar'."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)

        efn = EnergyFittingNet(
            self.nt,
            ds.dim_out,
        ).to(self.device)
        serialized = efn.serialize()

        # Check that the type is 'ener' not 'invar'
        self.assertEqual(serialized["type"], "ener")

        # Check that it can be deserialized
        efn2 = EnergyFittingNet.deserialize(serialized).to(self.device)
        self.assertIsInstance(efn2, EnergyFittingNet)

    def test_make_fx(self) -> None:
        from ..export_helpers import (
            export_save_load_and_compare,
        )

        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        rng = np.random.default_rng(GLOBAL_SEED)

        for nfp, nap in [(0, 0), (3, 4)]:
            efn = (
                EnergyFittingNet(
                    self.nt,
                    ds.dim_out,
                    numb_fparam=nfp,
                    numb_aparam=nap,
                    precision="float64",
                )
                .to(self.device)
                .eval()
            )

            descriptor = torch.from_numpy(
                rng.standard_normal((self.nf, self.nloc, ds.dim_out))
            ).to(self.device)
            atype = torch.from_numpy(self.atype_ext[:, :nloc]).to(self.device)
            fparam = (
                torch.from_numpy(rng.standard_normal((self.nf, nfp))).to(self.device)
                if nfp > 0
                else None
            )
            aparam = (
                torch.from_numpy(rng.standard_normal((self.nf, self.nloc, nap))).to(
                    self.device
                )
                if nap > 0
                else None
            )

            def fn(descriptor, atype, fparam, aparam):
                descriptor = descriptor.detach().requires_grad_(True)
                ret = efn(descriptor, atype, fparam=fparam, aparam=aparam)["energy"]
                grad = torch.autograd.grad(ret.sum(), descriptor, create_graph=False)[0]
                return ret, grad

            ret_eager, grad_eager = fn(descriptor, atype, fparam, aparam)
            traced = make_fx(fn)(descriptor, atype, fparam, aparam)
            ret_traced, grad_traced = traced(descriptor, atype, fparam, aparam)
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

            # --- symbolic trace + export + .pte round-trip ---
            nframes_dim = torch.export.Dim("nframes", min=1)
            dynamic_shapes = (
                {0: nframes_dim},  # descriptor
                {0: nframes_dim},  # atype
                {0: nframes_dim} if fparam is not None else None,  # fparam
                {0: nframes_dim} if aparam is not None else None,  # aparam
            )
            inputs = (descriptor, atype, fparam, aparam)
            loaded = export_save_load_and_compare(
                fn,
                inputs,
                (ret_eager, grad_eager),
                dynamic_shapes,
                rtol=1e-10,
                atol=1e-10,
            )

            # Different shapes (nf=1 slice)
            inputs_1f = tuple(t[0:1] if t is not None else None for t in inputs)
            ret_1f, grad_1f = fn(*inputs_1f)
            ret_loaded_1f, grad_loaded_1f = loaded(*inputs_1f)
            np.testing.assert_allclose(
                ret_1f.detach().cpu().numpy(),
                ret_loaded_1f.detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
            )
            np.testing.assert_allclose(
                grad_1f.detach().cpu().numpy(),
                grad_loaded_1f.detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
            )

    def test_torch_export_simple(self) -> None:
        """Test that EnergyFittingNet can be exported with torch.export."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        rng = np.random.default_rng(GLOBAL_SEED)

        efn = EnergyFittingNet(
            self.nt,
            ds.dim_out,
            numb_fparam=0,
            numb_aparam=0,
        ).to(self.device)

        # Prepare inputs
        descriptor = torch.from_numpy(
            rng.standard_normal((self.nf, self.nloc, ds.dim_out))
        ).to(self.device)
        atype = torch.from_numpy(self.atype_ext[:, :nloc]).to(self.device)

        # Test forward pass works
        ret = efn(descriptor, atype)
        self.assertIn("energy", ret)

        # Test torch.export
        exported = torch.export.export(
            efn,
            (descriptor, atype),
            kwargs={},
            strict=False,
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

    def test_torch_export_with_aparam(self) -> None:
        """Test that EnergyFittingNet with aparam can be exported."""
        nf, nloc, nnei = self.nlist.shape
        ds = DescrptSeA(self.rcut, self.rcut_smth, self.sel)
        rng = np.random.default_rng(GLOBAL_SEED)

        efn = EnergyFittingNet(
            self.nt,
            ds.dim_out,
            numb_fparam=0,
            numb_aparam=4,
        ).to(self.device)

        # Prepare inputs
        descriptor = torch.from_numpy(
            rng.normal(size=(self.nf, self.nloc, ds.dim_out))
        ).to(self.device)
        atype = torch.from_numpy(self.atype_ext[:, :nloc]).to(self.device)
        aparam = torch.from_numpy(rng.normal(size=(self.nf, self.nloc, 4))).to(
            self.device
        )

        # Test forward pass works
        ret = efn(descriptor, atype, aparam=aparam)
        self.assertIn("energy", ret)

        # Test torch.export
        exported = torch.export.export(
            efn,
            (descriptor, atype),
            kwargs={"aparam": aparam},
            strict=False,
        )
        self.assertIsNotNone(exported)

        # Test exported model produces same output
        ret_exported = exported.module()(descriptor, atype, aparam=aparam)
        np.testing.assert_allclose(
            ret["energy"].detach().cpu().numpy(),
            ret_exported["energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
