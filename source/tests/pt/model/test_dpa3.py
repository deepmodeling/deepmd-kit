# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.pt.model.descriptor import (
    DescrptDPA3,
    DescrptHybrid,
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


def _repflow_args() -> RepFlowArgs:
    return RepFlowArgs(
        n_dim=8,
        e_dim=6,
        a_dim=4,
        nlayers=1,
        e_rcut=4.0,
        e_rcut_smth=0.5,
        e_sel=12,
        a_rcut=3.5,
        a_rcut_smth=0.5,
        a_sel=8,
        axis_neuron=4,
        update_angle=False,
    )


class TestDescrptDPA3(unittest.TestCase, TestCaseSingleFrameWithNlist):
    def setUp(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)

    def test_consistency(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for (
            ua,
            rus,
            ruri,
            acr,
            acer,
            acus,
            nme,
            prec,
            ect,
            cs_mode,
            seq_upd,
        ) in itertools.product(
            [True, False],  # update_angle
            ["res_residual"],  # update_style
            ["norm", "const"],  # update_residual_init
            [0, 1],  # a_compress_rate
            [1, 2],  # a_compress_e_rate
            [True, False],  # a_compress_use_split
            [1, 2],  # n_multi_edge_message
            ["float64"],  # precision
            [False],  # use_econf_tebd
            ["no_chg_spin", "explicit_chg_spin", "default_chg_spin"],
            [False, True],  # sequential_update
        ):
            # sequential_update only works with update_angle=True
            if seq_upd and not ua:
                continue
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            if prec == "float64":
                atol = 1e-8  # marginal GPU test cases...

            add_chg_spin = cs_mode != "no_chg_spin"
            default_chg_spin = [5.0, 1.0] if cs_mode == "default_chg_spin" else None
            # Descriptor.forward does not apply default_chg_spin fallback
            # (that lives in dp_atomic_model). When add_chg_spin_ebd is on,
            # tests must always pass an explicit charge_spin tensor.
            need_cs_input = add_chg_spin

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
                sequential_update=seq_upd,
            )

            # dpa3 new impl
            dd0 = DescrptDPA3(
                self.nt,
                repflow=repflow,
                # kwargs for descriptor
                exclude_types=[],
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                add_chg_spin_ebd=add_chg_spin,
                default_chg_spin=default_chg_spin,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)

            # Prepare charge_spin per mode.
            charge_spin = None
            if need_cs_input:
                charge_spin = torch.tensor(
                    [[5, 1]], dtype=dtype, device=env.DEVICE
                ).expand(nf, -1)

            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
                charge_spin=charge_spin,
            )
            # serialization
            dd1 = DescrptDPA3.deserialize(dd0.serialize())
            rd1, _, _, _, _ = dd1(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
                charge_spin=charge_spin,
            )
            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
            # Cross-backend (dpmodel vs pt) numeric consistency for
            # add_chg_spin_ebd is covered by
            # source/tests/consistent/descriptor/test_dpa3.py.

            # default_chg_spin should match explicit when value is the same.
            if cs_mode == "default_chg_spin":
                dd_explicit = DescrptDPA3(
                    self.nt,
                    repflow=repflow,
                    exclude_types=[],
                    precision=prec,
                    use_econf_tebd=ect,
                    type_map=["O", "H"] if ect else None,
                    add_chg_spin_ebd=True,
                    default_chg_spin=None,
                    seed=GLOBAL_SEED,
                ).to(env.DEVICE)
                dd_explicit.repflows.mean = torch.tensor(
                    davg, dtype=dtype, device=env.DEVICE
                )
                dd_explicit.repflows.stddev = torch.tensor(
                    dstd, dtype=dtype, device=env.DEVICE
                )
                cs = torch.tensor([[5, 1]], dtype=dtype, device=env.DEVICE).expand(
                    nf, -1
                )
                rd_explicit, _, _, _, _ = dd_explicit(
                    torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                    torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                    torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                    torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
                    charge_spin=cs,
                )
                np.testing.assert_allclose(
                    rd0.detach().cpu().numpy(),
                    rd_explicit.detach().cpu().numpy(),
                    rtol=rtol,
                    atol=atol,
                )

    def test_hybrid_default_chg_spin_semantics(self) -> None:
        def make_dpa3(default_chg_spin: list[float] | None) -> DescrptDPA3:
            return DescrptDPA3(
                self.nt,
                repflow=_repflow_args(),
                precision="float64",
                add_chg_spin_ebd=True,
                default_chg_spin=default_chg_spin,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

        shared_default = DescrptHybrid(
            list=[make_dpa3([5.0, 1.0]), make_dpa3([5.0, 1.0])]
        )
        self.assertTrue(shared_default.has_default_chg_spin())
        torch.testing.assert_close(
            shared_default.get_default_chg_spin(),
            torch.tensor([5.0, 1.0], dtype=torch.float64, device=env.DEVICE),
        )

        missing_default = DescrptHybrid(list=[make_dpa3([5.0, 1.0]), make_dpa3(None)])
        self.assertFalse(missing_default.has_default_chg_spin())
        self.assertIsNone(missing_default.get_default_chg_spin())

        mismatched_default = DescrptHybrid(
            list=[make_dpa3([5.0, 1.0]), make_dpa3([6.0, 1.0])]
        )
        self.assertFalse(mismatched_default.has_default_chg_spin())
        self.assertIsNone(mismatched_default.get_default_chg_spin())

    def test_jit(
        self,
    ) -> None:
        rng = np.random.default_rng(100)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        for (
            ua,
            rus,
            ruri,
            acr,
            acer,
            acus,
            nme,
            prec,
            ect,
            seq_upd,
        ) in itertools.product(
            [True],  # update_angle
            ["res_residual"],  # update_style
            ["const"],  # update_residual_init
            [0, 1],  # a_compress_rate
            [2],  # a_compress_e_rate
            [True],  # a_compress_use_split
            [1, 2],  # n_multi_edge_message
            ["float64"],  # precision
            [False],  # use_econf_tebd
            [False, True],  # sequential_update
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)

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
                sequential_update=seq_upd,
            )

            # dpa3 new impl
            dd0 = DescrptDPA3(
                self.nt,
                repflow=repflow,
                # kwargs for descriptor
                exclude_types=[],
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            model = torch.jit.script(dd0)
