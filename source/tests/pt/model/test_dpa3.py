# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import torch

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DPDescrptDPA3
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
        ):
            dtype = PRECISION_DICT[prec]
            rtol, atol = get_tols(prec)
            if prec == "float64":
                atol = 1e-8  # marginal GPU test cases...

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
            rd0, _, _, _, _ = dd0(
                torch.tensor(self.coord_ext, dtype=dtype, device=env.DEVICE),
                torch.tensor(self.atype_ext, dtype=int, device=env.DEVICE),
                torch.tensor(self.nlist, dtype=int, device=env.DEVICE),
                torch.tensor(self.mapping, dtype=int, device=env.DEVICE),
            )
            # serialization
            dd1 = DescrptDPA3.deserialize(dd0.serialize())
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
