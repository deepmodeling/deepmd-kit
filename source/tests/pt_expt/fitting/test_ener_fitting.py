# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest
from unittest import (
    mock,
)

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting.ener_fitting import EnergyFittingNet as EnergyFittingNetDP
from deepmd.pt.cxx_op import (
    ENABLE_CUSTOMIZED_OP,
)
from deepmd.pt_expt.fitting import (
    EnergyFittingNet,
    ener_fitting,
)
from deepmd.pt_expt.kernels.graph_fitting import (
    op_available,
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
from ..export_helpers import (
    export_save_load_and_compare,
)


@pytest.mark.skipif(
    not ENABLE_CUSTOMIZED_OP or not op_available(),
    reason="the fused fitting operator is unavailable",
)
@pytest.mark.parametrize("activation", ["tanh", "silu"])
@pytest.mark.parametrize("reference_mode", ["none", "vacuum", "folded"])
def test_fused_node_gate_matches_portable(
    monkeypatch: pytest.MonkeyPatch, activation: str, reference_mode: str
) -> None:
    """The fused readout preserves gates and their gradients across vacuum folding."""
    monkeypatch.setenv("DP_CUDA_INFER", "1")
    fitting = (
        EnergyFittingNet(
            ntypes=2,
            dim_descrpt=12,
            neuron=[16, 16],
            resnet_dt=False,
            activation_function=activation,
            precision="float32",
            mixed_types=True,
            vacuum_ref=reference_mode != "none",
            seed=3,
        )
        .to(env.DEVICE)
        .eval()
    )
    generator = torch.Generator(device=env.DEVICE).manual_seed(5)
    descriptor = torch.randn(
        8, 12, generator=generator, dtype=torch.float32, device=env.DEVICE
    ).requires_grad_(True)
    atype = torch.arange(8, dtype=torch.int64, device=env.DEVICE) % 2
    gate = torch.tensor(
        [0.0, 1.0, 0.2, 0.65, 0.9, 0.0, 0.4, 1.0],
        dtype=torch.float64,
        device=env.DEVICE,
        requires_grad=True,
    )
    vacuum = (
        torch.randn(2, 12, generator=generator, dtype=torch.float32, device=env.DEVICE)
        if reference_mode != "none"
        else None
    )
    with torch.no_grad():
        fitting.bias_atom_e[:, 0].copy_(fitting.bias_atom_e.new_tensor([1.25, -0.75]))
    if reference_mode == "folded":
        fitting.fold_vacuum_reference(vacuum)
        vacuum = None

    kwargs = {"vacuum_descriptor": vacuum, "node_gate": gate}
    reference = EnergyFittingNetDP.call_graph(fitting, descriptor, atype, **kwargs)[
        "energy"
    ]
    with mock.patch.object(
        ener_fitting, "graph_fitting", wraps=ener_fitting.graph_fitting
    ) as fused:
        actual = fitting.call_graph(descriptor, atype, **kwargs)["energy"]
    fused.assert_called_once()
    torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
    closed = gate == 0.0
    bias = fitting.readout_reference().to(actual.dtype)[atype]
    torch.testing.assert_close(actual[closed], bias[closed], atol=0.0, rtol=0.0)

    cotangent = torch.linspace(-1.0, 1.0, 8, device=env.DEVICE)[:, None]
    expected_gradients = torch.autograd.grad(
        (reference * cotangent).sum(), (descriptor, gate)
    )
    actual_gradients = torch.autograd.grad(
        (actual * cotangent).sum(), (descriptor, gate)
    )
    torch.testing.assert_close(
        actual_gradients, expected_gradients, atol=2e-5, rtol=2e-5
    )
    assert actual_gradients[1].abs().max() > 1e-6


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
            export_save_load_and_compare(
                fn,
                inputs,
                (ret_eager, grad_eager),
                dynamic_shapes,
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
