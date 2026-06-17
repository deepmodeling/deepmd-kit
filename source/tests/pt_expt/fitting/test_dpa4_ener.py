# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.fitting.dpa4_ener import (
    SeZMEnergyFittingNet as SeZMEnergyFittingNetDP,
)
from deepmd.pt_expt.fitting.dpa4_ener import (
    SeZMEnergyFittingNet,
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

DIM_DESCRPT = 12


class TestSeZMEnergyFittingNet(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE
        rng = np.random.default_rng(GLOBAL_SEED)
        self.descriptor = rng.normal(size=(self.nf, self.nloc, DIM_DESCRPT))
        self.atype = self.atype_ext[:, : self.nloc]

    @pytest.mark.parametrize("neuron", [[0], [16, 16]])  # auto-width / hidden layers
    @pytest.mark.parametrize("mixed_types", [True, False])  # type-mixed branches
    def test_self_consistency_and_dpmodel(self, neuron, mixed_types) -> None:
        ft0 = SeZMEnergyFittingNet(
            self.nt,
            DIM_DESCRPT,
            neuron=neuron,
            mixed_types=mixed_types,
            precision="float64",
            seed=GLOBAL_SEED,
        ).to(self.device)
        ft1 = SeZMEnergyFittingNet.deserialize(ft0.serialize()).to(self.device)
        ft_dp = SeZMEnergyFittingNetDP.deserialize(ft0.serialize())

        descriptor_t = torch.from_numpy(self.descriptor).to(self.device)
        atype_t = torch.from_numpy(self.atype).to(self.device)
        ret0 = ft0(descriptor_t, atype_t)["energy"].detach().cpu().numpy()
        ret1 = ft1(descriptor_t, atype_t)["energy"].detach().cpu().numpy()
        ret_dp = ft_dp.call(self.descriptor, self.atype)["energy"]
        np.testing.assert_allclose(ret0, ret1, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(ret0, ret_dp, rtol=1e-12, atol=1e-14)

    @pytest.mark.parametrize("neuron", [[0], [16, 16]])  # auto-width / hidden layers
    def test_trainable_parameters(self, neuron) -> None:
        ft = SeZMEnergyFittingNet(
            self.nt,
            DIM_DESCRPT,
            neuron=neuron,
            precision="float64",
            seed=GLOBAL_SEED,
        ).to(self.device)
        params = list(ft.parameters())
        assert len(params) > 0
        names = [name for name, _ in ft.named_parameters()]
        assert any("hidden_layers" in name for name in names)
        assert any("output_layer" in name for name in names)

    def test_serialize_type(self) -> None:
        ft = SeZMEnergyFittingNet(
            self.nt, DIM_DESCRPT, neuron=[16], precision="float64", seed=GLOBAL_SEED
        ).to(self.device)
        assert ft.serialize()["type"] == "sezm_ener"

    def test_make_fx(self) -> None:
        ft = (
            SeZMEnergyFittingNet(
                self.nt,
                DIM_DESCRPT,
                neuron=[16, 16],
                precision="float64",
                seed=GLOBAL_SEED,
            )
            .to(self.device)
            .eval()
        )
        descriptor_t = torch.from_numpy(self.descriptor).to(self.device)
        atype_t = torch.from_numpy(self.atype).to(self.device)

        def fn(descriptor, atype):
            descriptor = descriptor.detach().requires_grad_(True)
            ret = ft(descriptor, atype)["energy"]
            grad = torch.autograd.grad(ret.sum(), descriptor, create_graph=False)[0]
            return ret, grad

        traced = make_fx(fn)(descriptor_t, atype_t)
        ret_t, grad_t = traced(descriptor_t, atype_t)
        ret_e, grad_e = fn(descriptor_t, atype_t)
        np.testing.assert_allclose(
            ret_t.detach().cpu().numpy(), ret_e.detach().cpu().numpy()
        )
        np.testing.assert_allclose(
            grad_t.detach().cpu().numpy(), grad_e.detach().cpu().numpy()
        )
