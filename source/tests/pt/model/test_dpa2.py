# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DPDescrptDPA2
from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.pt.model.descriptor.dpa2 import (
    DescrptDPA2,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDescrptDPA2(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        davg_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd = 0.1 + np.abs(dstd)
        dstd_2 = 0.1 + np.abs(dstd_2)

        for (
            riti,
            riz,
            rp1c,
            rp1d,
            rp1g,
            rp1a,
            rp2g,
            rp2a,
            rph,
            rp2gate,
            rus,
            rpz,
            sm,
            prec,
            ect,
            ns,
        ) in itertools.product(
            ["concat", "strip"],  # repinit_tebd_input_mode
            [
                True,
            ],  # repinit_set_davg_zero
            [True, False],  # repformer_update_g1_has_conv
            [True, False],  # repformer_update_g1_has_drrd
            [True, False],  # repformer_update_g1_has_grrg
            [
                False,
            ],  # repformer_update_g1_has_attn
            [
                False,
            ],  # repformer_update_g2_has_g1g1
            [True, False],  # repformer_update_g2_has_attn
            [
                False,
            ],  # repformer_update_h2
            [
                True,
            ],  # repformer_attn2_has_gate
            ["res_avg", "res_residual"],  # repformer_update_style
            [
                True,
            ],  # repformer_set_davg_zero
            [
                True,
            ],  # smooth
            ["float64"],  # precision
            [False, True],  # use_econf_tebd
            [
                False,
                True,
            ],  # new sub-structures (use_sqrt_nnei, g1_out_conv, g1_out_mlp)
        ):
            if ns and not rp1d and not rp1g:
                continue
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            if prec == "float64":
                atol = 1e-8  # marginal GPU test cases...

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

            # dpa2 new impl
            dd0 = DescrptDPA2(
                self.nt,
                repinit=repinit,
                repformer=repformer,
                # kwargs for descriptor
                smooth=sm,
                exclude_types=[],
                add_tebd_to_repinit_out=False,
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repinit.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.repinit.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            dd0.repformers.mean = torch.tensor(davg_2, dtype=dtype, device=env.DEVICE)
            dd0.repformers.stddev = torch.tensor(dstd_2, dtype=dtype, device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptDPA2.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
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

    def test_jit(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        davg_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd = 0.1 + np.abs(dstd)

        for (
            riti,
            riz,
            rp1c,
            rp1d,
            rp1g,
            rp1a,
            rp2g,
            rp2a,
            rph,
            rp2gate,
            rus,
            rpz,
            sm,
            prec,
            ect,
            ns,
        ) in itertools.product(
            ["concat", "strip"],  # repinit_tebd_input_mode
            [
                True,
            ],  # repinit_set_davg_zero
            [
                True,
            ],  # repformer_update_g1_has_conv
            [
                True,
            ],  # repformer_update_g1_has_drrd
            [
                True,
            ],  # repformer_update_g1_has_grrg
            [
                True,
            ],  # repformer_update_g1_has_attn
            [
                True,
            ],  # repformer_update_g2_has_g1g1
            [
                True,
            ],  # repformer_update_g2_has_attn
            [
                False,
            ],  # repformer_update_h2
            [
                True,
            ],  # repformer_attn2_has_gate
            ["res_avg", "res_residual"],  # repformer_update_style
            [
                True,
            ],  # repformer_set_davg_zero
            [
                True,
            ],  # smooth
            ["float64"],  # precision
            [False, True],  # use_econf_tebd
            [True],  # new sub-structures (use_sqrt_nnei, g1_out_conv, g1_out_mlp)
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)

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

            # dpa2 new impl
            dd0 = DescrptDPA2(
                self.nt,
                repinit=repinit,
                repformer=repformer,
                # kwargs for descriptor
                smooth=sm,
                exclude_types=[],
                add_tebd_to_repinit_out=False,
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repinit.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.repinit.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            dd0.repformers.mean = torch.tensor(davg_2, dtype=dtype, device=env.DEVICE)
            dd0.repformers.stddev = torch.tensor(dstd_2, dtype=dtype, device=env.DEVICE)
            model = torch.jit.script(dd0)
