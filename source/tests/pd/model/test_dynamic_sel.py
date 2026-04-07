# SPDX-License-Identifier: LGPL-3.0-or-later
import itertools
import unittest

import numpy as np
import paddle

from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.pd.model.descriptor import (
    DescrptDPA3,
)
from deepmd.pd.utils import (
    env,
)
from deepmd.pd.utils.env import (
    PRECISION_DICT,
)

from ...seed import (
    GLOBAL_SEED,
)
from .test_env_mat import (
    TestCaseSingleFrameWithNlist,
)

dtype = env.GLOBAL_PD_FLOAT_PRECISION


class TestDescrptDPA3DynamicSel(unittest.TestCase, TestCaseSingleFrameWithNlist):
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
            # rtol, atol = get_tols(prec)
            rtol, atol = 1e-5, 1e-7
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
                sel_reduce_factor=1.0,  # test consistent when sel_reduce_factor == 1.0
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

            repflow.use_dynamic_sel = True

            # dpa3 new impl
            dd1 = DescrptDPA3(
                self.nt,
                repflow=repflow,
                # kwargs for descriptor
                exclude_types=[],
                precision=prec,
                use_econf_tebd=ect,
                type_map=["O", "H"] if ect else None,
                seed=GLOBAL_SEED,
            ).to(env.DEVICE)

            dd0.repflows.mean = paddle.to_tensor(davg, dtype=dtype).to(
                device=env.DEVICE
            )
            dd0.repflows.stddev = paddle.to_tensor(dstd, dtype=dtype).to(
                device=env.DEVICE
            )
            rd0, _, _, _, _ = dd0(
                paddle.to_tensor(self.coord_ext, dtype=dtype).to(device=env.DEVICE),
                paddle.to_tensor(self.atype_ext, dtype=paddle.int64).to(
                    device=env.DEVICE
                ),
                paddle.to_tensor(self.nlist, dtype=paddle.int64).to(device=env.DEVICE),
                paddle.to_tensor(self.mapping, dtype=paddle.int64).to(
                    device=env.DEVICE
                ),
            )
            # serialization
            dd1.repflows.mean = paddle.to_tensor(davg, dtype=dtype).to(
                device=env.DEVICE
            )
            dd1.repflows.stddev = paddle.to_tensor(dstd, dtype=dtype).to(
                device=env.DEVICE
            )
            rd1, _, _, _, _ = dd1(
                paddle.to_tensor(self.coord_ext, dtype=dtype).to(device=env.DEVICE),
                paddle.to_tensor(self.atype_ext, dtype=paddle.int64).to(
                    device=env.DEVICE
                ),
                paddle.to_tensor(self.nlist, dtype=paddle.int64).to(device=env.DEVICE),
                paddle.to_tensor(self.mapping, dtype=paddle.int64).to(
                    device=env.DEVICE
                ),
            )
            np.testing.assert_allclose(
                rd0.numpy(),
                rd1.numpy(),
                rtol=rtol,
                atol=atol,
            )
