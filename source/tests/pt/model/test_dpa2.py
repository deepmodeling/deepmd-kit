# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa2 import DescrptDPA2 as DPDescrptDPA2
from deepmd.pt.model.descriptor.dpa2 import (
    DescrptDPA2,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    PRECISION_DICT,
)

from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)
from .test_mlp import (
    get_tols,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION


class TestDescrptDPA2(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self):
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ):
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
        ) in itertools.product(
            ["concat", "strip"],  # repinit_tebd_input_mode
            [
                True,
            ],  # repinit_set_davg_zero
            [True, False],  # repformer_update_g1_has_conv
            [True, False],  # repformer_update_g1_has_drrd
            [True, False],  # repformer_update_g1_has_grrg
            [True, False],  # repformer_update_g1_has_attn
            [True, False],  # repformer_update_g2_has_g1g1
            [True, False],  # repformer_update_g2_has_attn
            [
                False,
            ],  # repformer_update_h2
            [True, False],  # repformer_attn2_has_gate
            ["res_avg", "res_residual"],  # repformer_update_style
            [
                True,
            ],  # repformer_set_davg_zero
            [True, False],  # smooth
            ["float64"],  # precision
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)

            # dpa2 new impl
            dd0 = DescrptDPA2(
                self.nt,
                repinit_rcut=self.rcut,
                repinit_rcut_smth=self.rcut_smth,
                repinit_nsel=self.sel_mix,
                repformer_rcut=self.rcut / 2,
                repformer_rcut_smth=self.rcut_smth,
                repformer_nsel=nnei // 2,
                # kwargs for repinit
                repinit_tebd_input_mode=riti,
                repinit_set_davg_zero=riz,
                # kwargs for repformer
                repformer_nlayers=3,
                repformer_g1_dim=20,
                repformer_g2_dim=10,
                repformer_axis_neuron=4,
                repformer_update_g1_has_conv=rp1c,
                repformer_update_g1_has_drrd=rp1d,
                repformer_update_g1_has_grrg=rp1g,
                repformer_update_g1_has_attn=rp1a,
                repformer_update_g2_has_g1g1=rp2g,
                repformer_update_g2_has_attn=rp2a,
                repformer_update_h2=rph,
                repformer_attn1_hidden=20,
                repformer_attn1_nhead=2,
                repformer_attn2_hidden=10,
                repformer_attn2_nhead=2,
                repformer_attn2_has_gate=rp2gate,
                repformer_update_style=rus,
                repformer_set_davg_zero=rpz,
                # kwargs for descriptor
                smooth=sm,
                exclude_types=[],
                add_tebd_to_repinit_out=False,
                precision=prec,
                old_impl=False,
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
            # old impl
            if prec == "float64" and rus == "res_avg":
                dd3 = DescrptDPA2(
                    self.nt,
                    repinit_rcut=self.rcut,
                    repinit_rcut_smth=self.rcut_smth,
                    repinit_nsel=self.sel_mix,
                    repformer_rcut=self.rcut / 2,
                    repformer_rcut_smth=self.rcut_smth,
                    repformer_nsel=nnei // 2,
                    # kwargs for repinit
                    repinit_tebd_input_mode=riti,
                    repinit_set_davg_zero=riz,
                    # kwargs for repformer
                    repformer_nlayers=3,
                    repformer_g1_dim=20,
                    repformer_g2_dim=10,
                    repformer_axis_neuron=4,
                    repformer_update_g1_has_conv=rp1c,
                    repformer_update_g1_has_drrd=rp1d,
                    repformer_update_g1_has_grrg=rp1g,
                    repformer_update_g1_has_attn=rp1a,
                    repformer_update_g2_has_g1g1=rp2g,
                    repformer_update_g2_has_attn=rp2a,
                    repformer_update_h2=rph,
                    repformer_attn1_hidden=20,
                    repformer_attn1_nhead=2,
                    repformer_attn2_hidden=10,
                    repformer_attn2_nhead=2,
                    repformer_attn2_has_gate=rp2gate,
                    repformer_update_style="res_avg",
                    repformer_set_davg_zero=rpz,
                    # kwargs for descriptor
                    smooth=sm,
                    exclude_types=[],
                    add_tebd_to_repinit_out=False,
                    precision=prec,
                    old_impl=True,
                ).to(env.DEVICE)
                dd0_state_dict = dd0.state_dict()
                dd3_state_dict = dd3.state_dict()
                for i in list(dd0_state_dict.keys()):
                    if ".bias" in i and (
                        ".linear1." in i or ".linear2." in i or ".head_map." in i
                    ):
                        dd0_state_dict[i] = dd0_state_dict[i].unsqueeze(0)
                    if ".attn2_lm.matrix" in i:
                        dd0_state_dict[
                            i.replace(".attn2_lm.matrix", ".attn2_lm.weight")
                        ] = dd0_state_dict.pop(i)

                dd3.load_state_dict(dd0_state_dict)
                rd3, _, _, _, _ = dd3(
                    torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                    torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                    torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                    torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
                )
                np.testing.assert_allclose(
                    rd0.detach().cpu().numpy(),
                    rd3.detach().cpu().numpy(),
                    rtol=rtol,
                    atol=atol,
                )

    def test_jit(
        self,
    ):
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
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)

            # dpa2 new impl
            dd0 = DescrptDPA2(
                self.nt,
                repinit_rcut=self.rcut,
                repinit_rcut_smth=self.rcut_smth,
                repinit_nsel=self.sel_mix,
                repformer_rcut=self.rcut / 2,
                repformer_rcut_smth=self.rcut_smth,
                repformer_nsel=nnei // 2,
                # kwargs for repinit
                repinit_tebd_input_mode=riti,
                repinit_set_davg_zero=riz,
                # kwargs for repformer
                repformer_nlayers=3,
                repformer_g1_dim=20,
                repformer_g2_dim=10,
                repformer_axis_neuron=4,
                repformer_update_g1_has_conv=rp1c,
                repformer_update_g1_has_drrd=rp1d,
                repformer_update_g1_has_grrg=rp1g,
                repformer_update_g1_has_attn=rp1a,
                repformer_update_g2_has_g1g1=rp2g,
                repformer_update_g2_has_attn=rp2a,
                repformer_update_h2=rph,
                repformer_attn1_hidden=20,
                repformer_attn1_nhead=2,
                repformer_attn2_hidden=10,
                repformer_attn2_nhead=2,
                repformer_attn2_has_gate=rp2gate,
                repformer_update_style=rus,
                repformer_set_davg_zero=rpz,
                # kwargs for descriptor
                smooth=sm,
                exclude_types=[],
                add_tebd_to_repinit_out=False,
                precision=prec,
                old_impl=False,
            ).to(env.DEVICE)

            dd0.repinit.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.repinit.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            dd0.repformers.mean = torch.tensor(davg_2, dtype=dtype, device=env.DEVICE)
            dd0.repformers.stddev = torch.tensor(dstd_2, dtype=dtype, device=env.DEVICE)
            model = torch.jit.script(dd0)
