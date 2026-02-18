# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DPDescrptDPA2
from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.pt_expt.descriptor.dpa2 import (
    DescrptDPA2,
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


class TestDescrptDPA2(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("riti", ["concat", "strip"])  # repinit_tebd_input_mode
    @pytest.mark.parametrize("rp1c", [True, False])  # repformer_update_g1_has_conv
    @pytest.mark.parametrize("rp1d", [True, False])  # repformer_update_g1_has_drrd
    @pytest.mark.parametrize("rp1g", [True, False])  # repformer_update_g1_has_grrg
    @pytest.mark.parametrize("rp2a", [True, False])  # repformer_update_g2_has_attn
    @pytest.mark.parametrize(
        "rus", ["res_avg", "res_residual"]
    )  # repformer_update_style
    @pytest.mark.parametrize("prec", ["float64"])  # precision
    @pytest.mark.parametrize("ect", [False, True])  # use_econf_tebd
    @pytest.mark.parametrize("ns", [False, True])  # new sub-structures
    def test_consistency(
        self, riti, rp1c, rp1d, rp1g, rp2a, rus, prec, ect, ns
    ) -> None:
        if ns and not rp1d and not rp1g:
            pytest.skip("invalid parameter combination")

        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        davg_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd = 0.1 + np.abs(dstd)
        dstd_2 = 0.1 + np.abs(dstd_2)

        # fixed parameters
        riz = True  # repinit_set_davg_zero
        rp1a = False  # repformer_update_g1_has_attn
        rp2g = False  # repformer_update_g2_has_g1g1
        rph = False  # repformer_update_h2
        rp2gate = True  # repformer_attn2_has_gate
        rpz = True  # repformer_set_davg_zero
        sm = True  # smooth

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        if prec == "float64":
            atol = 1e-8  # marginal test cases

        repinit = RepinitArgs(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            nsel=self.sel_mix,
            tebd_input_mode=riti,
            set_davg_zero=riz,
        )
        repformer = RepformerArgs(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth,
            nsel=nnei // 2,
            nlayers=3,
            g1_dim=20,
            g2_dim=10,
            axis_neuron=4,
            update_g1_has_conv=rp1c,
            update_g1_has_drrd=rp1d,
            update_g1_has_grrg=rp1g,
            update_g1_has_attn=rp1a,
            update_g2_has_g1g1=rp2g,
            update_g2_has_attn=rp2a,
            update_h2=rph,
            attn1_hidden=20,
            attn1_nhead=2,
            attn2_hidden=10,
            attn2_nhead=2,
            attn2_has_gate=rp2gate,
            update_style=rus,
            set_davg_zero=rpz,
            use_sqrt_nnei=ns,
            g1_out_conv=ns,
            g1_out_mlp=ns,
        )

        dd0 = DescrptDPA2(
            self.nt,
            repinit=repinit,
            repformer=repformer,
            smooth=sm,
            exclude_types=[],
            add_tebd_to_repinit_out=False,
            precision=prec,
            use_econf_tebd=ect,
            type_map=["O", "H"] if ect else None,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repinit.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repinit.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0.repformers.mean = torch.tensor(davg_2, dtype=dtype, device=self.device)
        dd0.repformers.stddev = torch.tensor(dstd_2, dtype=dtype, device=self.device)
        rd0, _, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
            torch.tensor(self.mapping, dtype=int, device=self.device),
        )
        # serialization round-trip
        dd1 = DescrptDPA2.deserialize(dd0.serialize())
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
        dd2 = DPDescrptDPA2.deserialize(dd0.serialize())
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
        davg_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd = 0.1 + np.abs(dstd)
        dstd_2 = 0.1 + np.abs(dstd_2)

        dtype = PRECISION_DICT[prec]

        repinit = RepinitArgs(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            nsel=self.sel_mix,
            tebd_input_mode="concat",
            set_davg_zero=True,
        )
        repformer = RepformerArgs(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth,
            nsel=nnei // 2,
            nlayers=3,
            g1_dim=20,
            g2_dim=10,
            axis_neuron=4,
            update_g1_has_conv=True,
            update_g1_has_drrd=True,
            update_g1_has_grrg=True,
            update_g1_has_attn=False,
            update_g2_has_g1g1=False,
            update_g2_has_attn=True,
            update_h2=False,
            attn1_hidden=20,
            attn1_nhead=2,
            attn2_hidden=10,
            attn2_nhead=2,
            attn2_has_gate=True,
            update_style="res_avg",
            set_davg_zero=True,
            use_sqrt_nnei=True,
            g1_out_conv=True,
            g1_out_mlp=True,
        )

        dd0 = DescrptDPA2(
            self.nt,
            repinit=repinit,
            repformer=repformer,
            smooth=True,
            exclude_types=[],
            add_tebd_to_repinit_out=False,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repinit.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repinit.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0.repformers.mean = torch.tensor(davg_2, dtype=dtype, device=self.device)
        dd0.repformers.stddev = torch.tensor(dstd_2, dtype=dtype, device=self.device)
        dd0 = dd0.eval()
        inputs = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
            torch.tensor(self.mapping, dtype=int, device=self.device),
        )
        torch.export.export(dd0, inputs)

    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_make_fx(self, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        davg_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd = 0.1 + np.abs(dstd)
        dstd_2 = 0.1 + np.abs(dstd_2)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        if prec == "float64":
            atol = 1e-8

        repinit = RepinitArgs(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            nsel=self.sel_mix,
            tebd_input_mode="concat",
            set_davg_zero=True,
        )
        repformer = RepformerArgs(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth,
            nsel=nnei // 2,
            nlayers=3,
            g1_dim=20,
            g2_dim=10,
            axis_neuron=4,
            update_g1_has_conv=True,
            update_g1_has_drrd=True,
            update_g1_has_grrg=True,
            update_g1_has_attn=False,
            update_g2_has_g1g1=False,
            update_g2_has_attn=True,
            update_h2=False,
            attn1_hidden=20,
            attn1_nhead=2,
            attn2_hidden=10,
            attn2_nhead=2,
            attn2_has_gate=True,
            update_style="res_avg",
            set_davg_zero=True,
            use_sqrt_nnei=True,
            g1_out_conv=True,
            g1_out_mlp=True,
        )

        dd0 = DescrptDPA2(
            self.nt,
            repinit=repinit,
            repformer=repformer,
            smooth=True,
            exclude_types=[],
            add_tebd_to_repinit_out=False,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repinit.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repinit.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0.repformers.mean = torch.tensor(davg_2, dtype=dtype, device=self.device)
        dd0.repformers.stddev = torch.tensor(dstd_2, dtype=dtype, device=self.device)
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
