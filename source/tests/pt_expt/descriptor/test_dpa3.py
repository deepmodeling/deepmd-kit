# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DPDescrptDPA3
from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.pt_expt.descriptor.dpa3 import (
    DescrptDPA3,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)

from ...pt.model.test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from ...pt.model.test_mlp import (
    get_tols,
)
from ...seed import (
    GLOBAL_SEED,
)


class TestDescrptDPA3(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("ua", [True, False])  # update_angle
    @pytest.mark.parametrize("ruri", ["norm", "const"])  # update_residual_init
    @pytest.mark.parametrize("acr", [0, 1])  # a_compress_rate
    @pytest.mark.parametrize("acer", [1, 2])  # a_compress_e_rate
    @pytest.mark.parametrize("acus", [True, False])  # a_compress_use_split
    @pytest.mark.parametrize("nme", [1, 2])  # n_multi_edge_message
    def test_consistency(self, ua, ruri, acr, acer, acus, nme) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        # fixed parameters
        rus = "res_residual"  # update_style
        prec = "float64"  # precision
        ect = False  # use_econf_tebd

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        if prec == "float64":
            atol = 1e-8  # marginal test cases

        repflow = RepFlowArgs(
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            a_compress_rate=acr,
            a_compress_e_rate=acer,
            a_compress_use_split=acus,
            n_multi_edge_message=nme,
            axis_neuron=4,
            update_angle=ua,
            update_style=rus,
            update_residual_init=ruri,
            smooth_edge_update=True,
        )

        dd0 = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision=prec,
            use_econf_tebd=ect,
            type_map=["O", "H"] if ect else None,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        rd0, _, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
            torch.tensor(self.mapping, dtype=int, device=self.device),
        )
        # serialization round-trip
        dd1 = DescrptDPA3.deserialize(dd0.serialize())
        rd1, _, _, _, _ = dd1(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
            torch.tensor(self.mapping, dtype=int, device=self.device),
        )
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd1.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        # dp impl
        dd2 = DPDescrptDPA3.deserialize(dd0.serialize())
        rd2, _, _, _, _ = dd2.call(
            self.coord_ext, self.atype_ext, self.nlist, self.mapping
        )
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd2,
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_exportable(self, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]

        repflow = RepFlowArgs(
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            update_angle=True,
            update_style="res_residual",
            update_residual_init="const",
            smooth_edge_update=True,
        )

        dd0 = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0 = dd0.eval()
        inputs = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
            torch.tensor(self.mapping, dtype=int, device=self.device),
        )
        torch.export.export(dd0, inputs)

    @pytest.mark.parametrize("ruri", ["norm", "const"])  # update_residual_init
    @pytest.mark.parametrize("acus", [True, False])  # a_compress_use_split
    @pytest.mark.parametrize("nme", [1, 2])  # n_multi_edge_message
    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_make_fx(self, ruri, acus, nme, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        if prec == "float64":
            atol = 1e-8

        repflow = RepFlowArgs(
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            a_compress_use_split=acus,
            n_multi_edge_message=nme,
            axis_neuron=4,
            update_angle=True,
            update_style="res_residual",
            update_residual_init=ruri,
            smooth_edge_update=True,
        )

        dd0 = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0 = dd0.eval()
        coord_ext = torch.tensor(self.coord_ext, dtype=dtype, device=self.device)
        atype_ext = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)
        mapping = torch.tensor(self.mapping, dtype=int, device=self.device)

        def fn(coord_ext, atype_ext, nlist, mapping):
            coord_ext = coord_ext.detach().requires_grad_(True)
            rd = dd0(coord_ext, atype_ext, nlist, mapping)[0]
            grad = torch.autograd.grad(rd.sum(), coord_ext, create_graph=False)[0]
            return rd, grad

        rd_eager, grad_eager = fn(coord_ext, atype_ext, nlist, mapping)
        traced = make_fx(fn)(coord_ext, atype_ext, nlist, mapping)
        rd_traced, grad_traced = traced(coord_ext, atype_ext, nlist, mapping)
        np.testing.assert_allclose(
            rd_eager.detach().cpu().numpy(),
            rd_traced.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            grad_eager.detach().cpu().numpy(),
            grad_traced.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
