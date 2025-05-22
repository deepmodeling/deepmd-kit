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


class TestDescrptDPA3LocMapping(unittest.TestCase, TestCaseSingleFrameWithNlist):
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
            nme,
            prec,
            ect,
            optim,
        ) in itertools.product(
            [True, False],  # update_angle
            ["res_residual"],  # update_style
            ["norm", "const"],  # update_residual_init
            [0, 1],  # a_compress_rate
            [1, 2],  # n_multi_edge_message
            ["float64"],  # precision
            [False],  # use_econf_tebd
            [True, False],  # optim_update
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            if prec == "float64":
                atol = 1e-8  # marginal GPU test cases...

            repflow = RepFlowArgs(
                n_dim=20,
                e_dim=10,
                a_dim=10,
                nlayers=3,
                e_rcut=self.rcut,
                e_rcut_smth=self.rcut_smth,
                e_sel=nnei,
                a_rcut=self.rcut - 0.1,
                a_rcut_smth=self.rcut_smth,
                a_sel=nnei,
                a_compress_rate=acr,
                n_multi_edge_message=nme,
                axis_neuron=4,
                update_angle=ua,
                update_style=rus,
                update_residual_init=ruri,
                optim_update=optim,
                smooth_edge_update=True,
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
                use_loc_mapping=False,
            ).to(env.DEVICE)

            # dpa3 using local mapping
            dd1 = DescrptDPA3(
                self.nt,
                repflow=repflow,
                # kwargs for descriptor
                exclude_types=[],
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
                use_loc_mapping=True,
            ).to(env.DEVICE)

            coord_ext = np.concatenate([self.coord_ext[:1], self.coord_ext[:1]], axis=0)
            atype_ext = np.concatenate([self.atype_ext[:1], self.atype_ext[:1]], axis=0)
            nlist = np.concatenate([self.nlist[:1], self.nlist[:1]], axis=0)
            mapping = np.concatenate([self.mapping[:1], self.mapping[:1]], axis=0)

            dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                torch.tensor(coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(nlist, dtype=int, device=env.DEVICE),
                torch.tensor(mapping, dtype=int, device=env.DEVICE),
            )

            dd1.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd1.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd1, _, _, _, _ = dd1(
                torch.tensor(coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(nlist, dtype=int, device=env.DEVICE),
                torch.tensor(mapping, dtype=int, device=env.DEVICE),
            )

            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )

    def test_consistency_nosel(
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
            nme,
            prec,
            ect,
            optim,
        ) in itertools.product(
            [True, False],  # update_angle
            ["res_residual"],  # update_style
            ["norm", "const"],  # update_residual_init
            [0, 1],  # a_compress_rate
            [1, 2],  # n_multi_edge_message
            ["float64"],  # precision
            [False],  # use_econf_tebd
            [True, False],  # optim_update
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            if prec == "float64":
                atol = 1e-8  # marginal GPU test cases...

            repflow = RepFlowArgs(
                n_dim=20,
                e_dim=10,
                a_dim=10,
                nlayers=3,
                e_rcut=self.rcut,
                e_rcut_smth=self.rcut_smth,
                e_sel=nnei,
                a_rcut=self.rcut - 0.1,
                a_rcut_smth=self.rcut_smth,
                a_sel=nnei,
                a_compress_rate=acr,
                n_multi_edge_message=nme,
                axis_neuron=4,
                update_angle=ua,
                update_style=rus,
                update_residual_init=ruri,
                optim_update=optim,
                smooth_edge_update=True,
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
                use_loc_mapping=False,
            ).to(env.DEVICE)

            # dpa3 using local mapping
            dd1 = DescrptDPA3(
                self.nt,
                repflow=repflow,
                # kwargs for descriptor
                exclude_types=[],
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
                use_loc_mapping=True,
            ).to(env.DEVICE)

            coord_ext = np.concatenate([self.coord_ext[:1], self.coord_ext[:1]], axis=0)
            atype_ext = np.concatenate([self.atype_ext[:1], self.atype_ext[:1]], axis=0)
            nlist = np.concatenate([self.nlist[:1], self.nlist[:1]], axis=0)
            mapping = np.concatenate([self.mapping[:1], self.mapping[:1]], axis=0)

            dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd0, _, _, _, _ = dd0(
                torch.tensor(coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(nlist, dtype=int, device=env.DEVICE),
                torch.tensor(mapping, dtype=int, device=env.DEVICE),
            )

            dd1.repflows.mean = torch.tensor(davg, dtype=dtype, device=env.DEVICE)
            dd1.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=env.DEVICE)
            rd1, _, _, _, _ = dd1(
                torch.tensor(coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(nlist, dtype=int, device=env.DEVICE),
                torch.tensor(mapping, dtype=int, device=env.DEVICE),
            )

            np.testing.assert_allclose(
                rd0.detach().cpu().numpy(),
                rd1.detach().cpu().numpy(),
                rtol=rtol,
                atol=atol,
            )
